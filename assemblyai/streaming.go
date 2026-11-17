package assemblyai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/emit"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/retry"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	streamReadLimit   = 16 << 20
	defaultSampleRate = 16000
	audioChunk        = 100 * time.Millisecond
	minAudioChunk     = 20 * time.Millisecond
	maxAudioLead      = 500 * time.Millisecond
)

type streamFormat struct {
	encoding   string
	sampleRate int
	chunkBytes int
	minBytes   int
	byteRate   int
}

func (f streamFormat) duration(n int) time.Duration {
	return time.Duration(n) * time.Second / time.Duration(f.byteRate)
}

type streamWord struct {
	Text    string `json:"text"`
	Start   int64  `json:"start"`
	End     int64  `json:"end"`
	Speaker string `json:"speaker"`
}

type serverMessage struct {
	Type            string          `json:"type"`
	Transcript      string          `json:"transcript"`
	Utterance       string          `json:"utterance"`
	EndOfTurn       bool            `json:"end_of_turn"`
	TurnIsFormatted bool            `json:"turn_is_formatted"`
	SpeakerLabel    string          `json:"speaker_label"`
	Words           []streamWord    `json:"words"`
	ErrorCode       json.RawMessage `json:"error_code"`
	Error           string          `json:"error"`
}

func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	format, err := formatOf(c.MediaType)
	if err != nil {
		return nil, err
	}
	target, err := m.streamTarget(c, opts, format)
	if err != nil {
		return nil, err
	}
	if err := m.client.Wait(ctx); err != nil {
		return nil, err
	}
	header := m.client.Headers()
	for k, v := range c.Headers {
		header[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	ws, resp, err := websocket.Dial(ctx, target, &websocket.DialOptions{HTTPClient: m.client.HTTPClient(), HTTPHeader: header})
	if err != nil {
		if resp != nil {
			return nil, m.client.HandshakeError(resp)
		}
		return nil, fmt.Errorf("assemblyai: connect streaming: %w", err)
	}
	ws.SetReadLimit(streamReadLimit)
	return &ai.TranscribeStreamResponse{Segments: (&liveConn{ws: ws, format: format}).segments(ctx, c.Audio)}, nil
}

func (m *TranscriptionModel) streamTarget(c *ai.TranscribeCall, o TranscriptionOptions, f streamFormat) (string, error) {
	base, err := url.Parse(m.provider.streamURL)
	if err != nil {
		return "", fmt.Errorf("assemblyai: build streaming url: %w", err)
	}
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	case "http":
		base.Scheme = "ws"
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/v3/ws"
	q := url.Values{}
	q.Set("encoding", f.encoding)
	q.Set("sample_rate", strconv.Itoa(f.sampleRate))
	set := func(key, value string) {
		if value != "" {
			q.Set(key, value)
		}
	}
	set("speech_model", m.id)
	set("mode", o.Mode)
	set("prompt", o.Prompt)
	if o.MinTurnSilence > 0 {
		q.Set("min_turn_silence", strconv.FormatInt(o.MinTurnSilence.Milliseconds(), 10))
	}
	if o.MaxTurnSilence > 0 {
		q.Set("max_turn_silence", strconv.FormatInt(o.MaxTurnSilence.Milliseconds(), 10))
	}
	if o.VADThreshold != nil {
		q.Set("vad_threshold", strconv.FormatFloat(*o.VADThreshold, 'f', -1, 64))
	}
	if o.InactivityTimeout > 0 {
		q.Set("inactivity_timeout", strconv.FormatInt(int64(o.InactivityTimeout/time.Second), 10))
	}
	if o.MaxSpeakers > 0 {
		q.Set("max_speakers", strconv.Itoa(o.MaxSpeakers))
	}
	if o.SpeakerLabels {
		q.Set("speaker_labels", "true")
	}
	if o.FilterProfanity {
		q.Set("filter_profanity", "true")
	}
	if o.IncludePartialTurns != nil {
		q.Set("include_partial_turns", strconv.FormatBool(*o.IncludePartialTurns))
	}
	if o.LanguageDetection != nil {
		q.Set("language_detection", strconv.FormatBool(*o.LanguageDetection))
	}
	languages := o.LanguageCodes
	if len(languages) == 0 && c.Language != "" {
		languages = []string{c.Language}
	}
	for key, list := range map[string][]string{"language_codes": languages, "keyterms_prompt": o.KeytermsPrompt} {
		if len(list) == 0 {
			continue
		}
		raw, err := json.Marshal(list)
		if err != nil {
			return "", fmt.Errorf("assemblyai: encode %s: %w", key, err)
		}
		q.Set(key, string(raw))
	}
	base.RawQuery = q.Encode()
	return base.String(), nil
}

func formatOf(mediaType string) (streamFormat, error) {
	f := streamFormat{encoding: "pcm_s16le", sampleRate: defaultSampleRate}
	bytesPerSample := 2
	if mediaType != "" {
		base, params, err := mime.ParseMediaType(mediaType)
		if err != nil {
			return streamFormat{}, fmt.Errorf("assemblyai: parse media type %q: %w", mediaType, err)
		}
		switch base {
		case "audio/pcm", "audio/l16":
		case "audio/basic", "audio/pcmu", "audio/x-mulaw":
			f.encoding, bytesPerSample = "pcm_mulaw", 1
		default:
			return streamFormat{}, fmt.Errorf("assemblyai: streaming takes raw audio/pcm or audio/basic (mu-law), not %q", base)
		}
		if rate := params["rate"]; rate != "" {
			n, err := strconv.Atoi(rate)
			if err != nil || n < 8000 || n > 96000 {
				return streamFormat{}, fmt.Errorf("assemblyai: sample rate %q is outside 8000 to 96000", rate)
			}
			f.sampleRate = n
		}
	}
	perSecond := f.sampleRate * bytesPerSample
	f.byteRate = perSecond
	f.chunkBytes = perSecond * int(audioChunk/time.Millisecond) / 1000
	f.minBytes = perSecond * int(minAudioChunk/time.Millisecond) / 1000
	return f, nil
}

type liveConn struct {
	ws     *websocket.Conn
	format streamFormat
}

func (c *liveConn) segments(ctx context.Context, audio io.Reader) iter.Seq2[ai.Segment, error] {
	return func(yield func(ai.Segment, error) bool) {
		yield = emit.New(yield)
		ctx, cancel := context.WithCancel(ctx)
		sent := make(chan error, 1)
		var wg sync.WaitGroup
		defer func() {
			cancel()
			if closer, ok := audio.(io.Closer); ok {
				if err := closer.Close(); err != nil {
					yield(ai.Segment{}, fmt.Errorf("assemblyai: close audio: %w", err))
				}
				wg.Wait()
			}
			if err := c.ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				yield(ai.Segment{}, fmt.Errorf("assemblyai: close streaming: %w", err))
			}
		}()
		wg.Go(func() {
			if err := c.pump(ctx, audio); err != nil {
				sent <- err
				cancel()
			}
		})
		for {
			_, data, err := c.ws.Read(ctx)
			if err != nil {
				select {
				case perr := <-sent:
					yield(ai.Segment{}, perr)
				default:
					if websocket.CloseStatus(err) != websocket.StatusNormalClosure && ctx.Err() == nil {
						yield(ai.Segment{}, fmt.Errorf("assemblyai: read streaming: %w", err))
					}
				}
				return
			}
			seg, done, err := decodeServerMessage(data)
			if err != nil {
				yield(ai.Segment{}, err)
				return
			}
			if seg != nil && !yield(*seg, nil) {
				return
			}
			if done {
				return
			}
		}
	}
}

func (c *liveConn) pump(ctx context.Context, audio io.Reader) error {
	return pumpAudio(ctx, audio, c.format, c.ws.Write)
}

func pumpAudio(ctx context.Context, audio io.Reader, format streamFormat, write func(context.Context, websocket.MessageType, []byte) error) error {
	buf := make([]byte, format.chunkBytes)
	start := time.Now()
	var sent time.Duration
	for {
		n, err := io.ReadFull(audio, buf)
		if n > 0 {
			chunk := buf[:n]
			if n < format.minBytes {
				chunk = append(chunk, make([]byte, format.minBytes-n)...)
			}
			if lead := sent - maxAudioLead - time.Since(start); lead > 0 {
				if serr := retry.Sleep(ctx, lead); serr != nil {
					return serr
				}
			}
			if werr := write(ctx, websocket.MessageBinary, chunk); werr != nil {
				return fmt.Errorf("assemblyai: send audio: %w", werr)
			}
			sent += format.duration(len(chunk))
		}
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			if werr := write(ctx, websocket.MessageText, []byte(`{"type":"Terminate"}`)); werr != nil {
				return fmt.Errorf("assemblyai: send terminate: %w", werr)
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("assemblyai: read audio: %w", err)
		}
	}
}

func decodeServerMessage(data []byte) (seg *ai.Segment, done bool, err error) {
	var m serverMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, false, fmt.Errorf("assemblyai: decode streaming message: %w", err)
	}
	switch {
	case m.Type == "Termination":
		return nil, true, nil
	case m.Type == "Error" || (m.Type == "" && m.Error != ""):
		return nil, false, streamError(m)
	case m.Type != "Turn":
		return nil, false, nil
	}
	text := m.Transcript
	if text == "" {
		text = m.Utterance
	}
	if strings.TrimSpace(text) == "" {
		return nil, false, nil
	}
	out := ai.Segment{Text: text, Speaker: m.SpeakerLabel, Partial: !m.EndOfTurn || !m.TurnIsFormatted}
	if len(m.Words) > 0 {
		out.Start, out.End = millis(m.Words[0].Start), millis(m.Words[len(m.Words)-1].End)
		if out.Speaker == "" {
			out.Speaker = m.Words[0].Speaker
		}
	}
	return &out, false, nil
}

func streamError(m serverMessage) error {
	code := strings.Trim(string(m.ErrorCode), `"`)
	if code == "null" {
		code = ""
	}
	return &ai.APIError{Provider: providerName, Code: code, Message: transport.Truncate(m.Error)}
}
