package deepgram

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/emit"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxSpeechChars  = 2000
	maxSpeechBytes  = 256 << 20
	speechReadChunk = 16 << 10
	closeGrace      = 5 * time.Second
	ackTimeout      = 30 * time.Second
	flushWindow     = time.Minute
	flushLimit      = 20
	flushMessage    = `{"type":"Flush"}`
	closeMessage    = `{"type":"Close"}`
)

// SpeechModel synthesizes speech with Aura voices; the model id is the voice, for example "aura-2-thalia-en", and SpeechRequest.Voice overrides it. Text is limited to 2000 characters per request.
// StreamSpeech splits text into sentences over the speak WebSocket, which supports linear16, mulaw and alaw only; a text iterator that never returns keeps one goroutine blocked.
type SpeechModel struct {
	client *transport.Client
	id     string
}

type speakMessage struct {
	Type string `json:"type,omitzero"`
	Text string `json:"text,omitzero"`
}

type speakEvent struct {
	Type        string `json:"type"`
	Variant     string `json:"variant"`
	Description string `json:"description"`
	Code        string `json:"code"`
}

type audioFormat struct {
	encoding  string
	container string
	mediaType string
}

func (m *SpeechModel) Provider() string { return providerName }

func (m *SpeechModel) ModelID() string { return m.id }

func (m *SpeechModel) GenerateSpeech(ctx context.Context, c *ai.SpeechCall) (*ai.SpeechResponse, error) {
	if n := utf8.RuneCountInString(c.Text); n > maxSpeechChars {
		return nil, fmt.Errorf("deepgram: speech text has %d characters, the limit is %d; use StreamSpeech, which splits the text", n, maxSpeechChars)
	}
	opts, err := mediaio.DecodeOptions[SpeechOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	f, err := restFormat(c.Format, opts)
	if err != nil {
		return nil, err
	}
	q := m.speechQuery(c.Voice, f, opts, c.Speed)
	body, err := transport.EncodeJSON(speakMessage{Text: c.Text})
	if err != nil {
		return nil, fmt.Errorf("deepgram: encode request: %w", err)
	}
	resp, err := m.client.Do(ctx, transport.Request{Path: "/speak?" + q.Encode(), Raw: strings.NewReader(string(body)), ContentType: "application/json", Header: acceptAudio(c.Headers)})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxSpeechBytes+1))
	if err != nil {
		return nil, fmt.Errorf("deepgram: read audio: %w", err)
	}
	if len(data) > maxSpeechBytes {
		return nil, fmt.Errorf("deepgram: audio exceeds the %d MB limit", maxSpeechBytes>>20)
	}
	mediaType := resp.Header.Get("Content-Type")
	if mediaType == "" {
		mediaType = f.mediaType
	}
	out := &ai.SpeechResponse{Audio: ai.File{MediaType: mediaType, Data: data}, Header: resp.Header}
	if c.Instructions != "" {
		out.Warnings = append(out.Warnings, ai.Warning{Message: "deepgram: speech instructions are not supported and were ignored"})
	}
	return out, nil
}

func (m *SpeechModel) speechQuery(voice string, f audioFormat, opts SpeechOptions, speed *float64) query {
	q := newQuery()
	q.set("model", cmp.Or(voice, m.id))
	q.set("encoding", f.encoding)
	q.set("container", f.container)
	q.setInt("sample_rate", opts.SampleRate)
	q.setInt("bit_rate", opts.BitRate)
	q.setFloat("speed", speed)
	q.addAll("tag", opts.Tags)
	q.setFlag("mip_opt_out", opts.MIPOptOut)
	return q
}

func (m *SpeechModel) StreamSpeech(ctx context.Context, c *ai.SpeechStreamCall) (*ai.SpeechStreamResponse, error) {
	opts, err := mediaio.DecodeOptions[SpeechOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	f, err := streamFormat(c.Format, opts)
	if err != nil {
		return nil, err
	}
	q := m.speechQuery(c.Voice, f, SpeechOptions{SampleRate: opts.SampleRate, MIPOptOut: opts.MIPOptOut}, c.Speed)
	q.Del("container")
	ws, header, err := dial(ctx, m.client, "/speak", q.Values, c.Headers)
	if err != nil {
		return nil, err
	}
	s := &speakStream{ws: ws, text: c.Text}
	s.flushed = make(chan struct{}, 1)
	return &ai.SpeechStreamResponse{Audio: s.audio(ctx), MediaType: f.mediaType, Header: header}, nil
}

type speakStream struct {
	ws      *websocket.Conn
	text    iter.Seq[string]
	sent    atomic.Int64
	flushed chan struct{}
	acks    atomic.Int64
	warns   atomic.Int64
	flushes []time.Time
}

func (s *speakStream) audio(parent context.Context) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		yield = emit.New(yield)
		ctx, cancel := context.WithCancel(parent)
		failed := make(chan error, 1)
		defer func() {
			cancel()
			if err := s.ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				yield(nil, fmt.Errorf("deepgram: close speech stream: %w", err))
			}
		}()
		go func() {
			if err := s.write(ctx, cancel); err != nil {
				failed <- err
				cancel()
			}
		}()
		for {
			kind, data, err := s.ws.Read(ctx)
			if err != nil {
				select {
				case werr := <-failed:
					yield(nil, werr)
				default:
					switch {
					case parent.Err() != nil:
						yield(nil, fmt.Errorf("deepgram: read speech stream: %w", parent.Err()))
					case !s.finished(err):
						yield(nil, fmt.Errorf("deepgram: read speech stream: %w", err))
					}
				}
				return
			}
			if kind == websocket.MessageBinary {
				if !yield(data, nil) {
					return
				}
				continue
			}
			if err := s.handle(data); err != nil {
				yield(nil, err)
				return
			}
		}
	}
}

func (s *speakStream) finished(err error) bool {
	status := websocket.CloseStatus(err)
	return status == websocket.StatusNormalClosure || status == websocket.StatusNoStatusRcvd
}

func (s *speakStream) handle(data []byte) error {
	var ev speakEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return fmt.Errorf("deepgram: decode speech message: %w", err)
	}
	switch ev.Type {
	case "Flushed":
		s.acks.Add(1)
		select {
		case s.flushed <- struct{}{}:
		default:
		}
	case "Warning":
		s.warns.Add(1)
		select {
		case s.flushed <- struct{}{}:
		default:
		}
	case "Error":
		return transport.ErrorFromEvent(providerName, cmp.Or(ev.Variant, ev.Code), ev.Description)
	}
	return nil
}

func (s *speakStream) write(ctx context.Context, cancel context.CancelFunc) error {
	pending := false
	for chunk := range speechChunks(s.text) {
		if err := s.send(ctx, speakMessage{Type: "Speak", Text: chunk}); err != nil {
			return err
		}
		pending = true
		if !s.flushAvailable(flushLimit - 1) {
			continue
		}
		if err := s.flush(ctx); err != nil {
			return err
		}
		pending = false
	}
	if pending {
		if err := s.flush(ctx); err != nil {
			return err
		}
	}
	if s.awaitAcks(ctx); ctx.Err() != nil {
		return nil
	}
	if err := s.ws.Write(ctx, websocket.MessageText, []byte(closeMessage)); err != nil {
		return fmt.Errorf("deepgram: send Close: %w", err)
	}
	t := time.NewTimer(closeGrace)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
		cancel()
	}
	return nil
}

func (s *speakStream) flushAvailable(limit int) bool {
	cutoff := time.Now().Add(-flushWindow)
	for len(s.flushes) > 0 && !s.flushes[0].After(cutoff) {
		s.flushes = s.flushes[1:]
	}
	return len(s.flushes) < limit
}

func (s *speakStream) flush(ctx context.Context) error {
	if err := s.ws.Write(ctx, websocket.MessageText, []byte(flushMessage)); err != nil {
		return fmt.Errorf("deepgram: send Flush: %w", err)
	}
	s.flushes = append(s.flushes, time.Now())
	s.sent.Add(1)
	return nil
}

func (s *speakStream) awaitAcks(ctx context.Context) {
	t := time.NewTimer(ackTimeout)
	defer t.Stop()
	for s.acks.Load()+s.warns.Load() < s.sent.Load() {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			return
		case <-s.flushed:
		}
	}
}

func (s *speakStream) send(ctx context.Context, msg speakMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("deepgram: encode speak message: %w", err)
	}
	if err := s.ws.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("deepgram: send speak message: %w", err)
	}
	return nil
}

func restFormat(format string, opts SpeechOptions) (audioFormat, error) {
	var f audioFormat
	switch format {
	case "", "mp3":
		f.encoding = "mp3"
	case "wav":
		f.encoding, f.container = "linear16", "wav"
	case "pcm", "linear16":
		f.encoding, f.container = "linear16", "none"
	case "mulaw", "alaw":
		f.encoding, f.container = format, "none"
	case "flac", "opus", "aac":
		f.encoding = format
	default:
		return audioFormat{}, fmt.Errorf("deepgram: unsupported speech format %q", format)
	}
	f.encoding = cmp.Or(opts.Encoding, f.encoding)
	f.container = cmp.Or(opts.Container, f.container)
	f.mediaType = mediaTypeOf(f.encoding, f.container)
	return f, nil
}

func streamFormat(format string, opts SpeechOptions) (audioFormat, error) {
	var f audioFormat
	switch format {
	case "", "pcm", "linear16":
		f.encoding = "linear16"
	case "mulaw", "alaw":
		f.encoding = format
	default:
		return audioFormat{}, fmt.Errorf("deepgram: speech format %q is not available over the streaming WebSocket; use pcm, mulaw or alaw", format)
	}
	f.encoding = cmp.Or(opts.Encoding, f.encoding)
	f.container = "none"
	f.mediaType = mediaTypeOf(f.encoding, f.container)
	return f, nil
}

func mediaTypeOf(encoding, container string) string {
	switch encoding {
	case "mp3":
		return "audio/mpeg"
	case "flac":
		return "audio/flac"
	case "opus":
		return "audio/ogg"
	case "aac":
		return "audio/aac"
	}
	if container == "" || container == "wav" {
		return "audio/wav"
	}
	if encoding == "linear16" {
		return "audio/pcm"
	}
	return "audio/" + encoding
}

func speechChunks(text iter.Seq[string]) iter.Seq[string] {
	return func(yield func(string) bool) {
		var buf strings.Builder
		for piece := range text {
			buf.WriteString(piece)
			for {
				chunk, rest, ok := cutChunk(buf.String())
				if !ok {
					break
				}
				buf.Reset()
				buf.WriteString(rest)
				if !yield(chunk) {
					return
				}
			}
		}
		rest := strings.TrimSpace(buf.String())
		for rest != "" {
			chunk, tail, ok := forcedCut(rest)
			if utf8.RuneCountInString(rest) <= maxSpeechChars || !ok {
				chunk, tail = rest, ""
			}
			if !yield(chunk) {
				return
			}
			rest = strings.TrimSpace(tail)
		}
	}
}

func cutChunk(s string) (chunk, rest string, ok bool) {
	runes := 0
	end := -1
	for i, r := range s {
		runes++
		if runes > maxSpeechChars {
			if end >= 0 {
				break
			}
			return forcedCut(s)
		}
		if !isSentenceEnd(r) {
			continue
		}
		next := i + utf8.RuneLen(r)
		if r == '\n' || next < len(s) && startsWithSpace(s[next:]) {
			end = next
		}
	}
	if end < 0 {
		return "", "", false
	}
	chunk = strings.TrimSpace(s[:end])
	return chunk, s[end:], chunk != ""
}

func forcedCut(s string) (chunk, rest string, ok bool) {
	limit := len(s)
	count := 0
	for i := range s {
		if count++; count > maxSpeechChars {
			limit = i
			break
		}
	}
	head := s[:limit]
	if i := strings.LastIndexFunc(head, unicode.IsSpace); i > 0 && limit < len(s) {
		head = head[:i]
	}
	chunk = strings.TrimSpace(head)
	return chunk, s[len(head):], chunk != ""
}

func startsWithSpace(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsSpace(r)
}

func isSentenceEnd(r rune) bool {
	switch r {
	case '.', '!', '?', '\n', '。', '！', '？':
		return true
	default:
		return false
	}
}

func acceptAudio(headers http.Header) http.Header {
	h := headers.Clone()
	if h == nil {
		h = http.Header{}
	}
	h.Set("Accept", "audio/*")
	return h
}

var (
	_ ai.SpeechModel       = (*SpeechModel)(nil)
	_ ai.SpeechStreamModel = (*SpeechModel)(nil)
)
