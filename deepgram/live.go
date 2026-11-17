package deepgram

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/emit"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	liveAudioChunk     = 8192
	defaultKeepAlive   = 4 * time.Second
	defaultRawSampleHz = 16000
	closeStreamMessage = `{"type":"CloseStream"}`
	keepAliveMessage   = `{"type":"KeepAlive"}`
)

type liveMessage struct {
	Type        string          `json:"type"`
	IsFinal     bool            `json:"is_final"`
	Start       float64         `json:"start"`
	Duration    float64         `json:"duration"`
	Channel     json.RawMessage `json:"channel"`
	Variant     string          `json:"variant"`
	Description string          `json:"description"`
	Code        string          `json:"code"`
}

func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	q := m.commonQuery(c, opts)
	encoding, rate := liveEncoding(c.MediaType, opts)
	q.set("encoding", encoding)
	q.setInt("sample_rate", rate)
	q.setInt("channels", opts.Channels)
	q.setFlag("multichannel", opts.Multichannel)
	q.setBool("interim_results", orTrue(opts.InterimResults))
	switch {
	case opts.DisableEndpointing:
		q.Set("endpointing", "false")
	case opts.Endpointing != nil:
		q.Set("endpointing", strconv.Itoa(*opts.Endpointing))
	}
	q.setInt("utterance_end_ms", opts.UtteranceEndMS)
	q.setFlag("vad_events", opts.VADEvents)
	ws, header, err := dial(ctx, m.client, "/listen", q.Values, c.Headers)
	if err != nil {
		return nil, err
	}
	keepAlive := opts.KeepAlive
	if keepAlive <= 0 {
		keepAlive = defaultKeepAlive
	}
	l := &liveStream{ws: ws, audio: c.Audio, keepAlive: keepAlive}
	return &ai.TranscribeStreamResponse{Segments: l.segments(ctx), Header: header}, nil
}

func liveEncoding(mediaType string, opts TranscriptionOptions) (encoding string, rate int) {
	base, params, err := mime.ParseMediaType(mediaType)
	if err != nil {
		base, _, _ = strings.Cut(mediaType, ";")
		base = strings.ToLower(strings.TrimSpace(base))
	}
	switch base {
	case "audio/pcm", "audio/l16", "audio/x-raw":
		encoding = "linear16"
	case "audio/basic", "audio/mulaw", "audio/x-mulaw", "audio/pcmu":
		encoding = "mulaw"
	case "audio/alaw", "audio/pcma":
		encoding = "alaw"
	}
	if opts.Encoding != "" {
		encoding = opts.Encoding
	}
	rate, _ = strconv.Atoi(params["rate"])
	if opts.SampleRate != 0 {
		rate = opts.SampleRate
	}
	if encoding != "" && rate == 0 {
		rate = defaultRawSampleHz
	}
	return encoding, rate
}

type liveStream struct {
	ws        *websocket.Conn
	audio     io.Reader
	keepAlive time.Duration
	sent      atomic.Bool
	closeSent atomic.Bool
}

func (l *liveStream) segments(parent context.Context) iter.Seq2[ai.Segment, error] {
	return func(yield func(ai.Segment, error) bool) {
		yield = emit.New(yield)
		ctx, cancel := context.WithCancel(parent)
		failed := make(chan error, 1)
		var wg sync.WaitGroup
		defer func() {
			cancel()
			if closer, ok := l.audio.(io.Closer); ok {
				if err := closer.Close(); err != nil {
					yield(ai.Segment{}, fmt.Errorf("deepgram: close audio: %w", err))
				}
				wg.Wait()
			}
			if err := l.ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				yield(ai.Segment{}, fmt.Errorf("deepgram: close live transcription: %w", err))
			}
		}()
		pumped := make(chan struct{})
		wg.Go(func() {
			defer close(pumped)
			if err := l.pump(ctx); err != nil {
				failed <- err
				cancel()
			}
		})
		wg.Go(func() { l.pingLoop(ctx, pumped) })
		for {
			kind, data, err := l.ws.Read(ctx)
			if err != nil {
				select {
				case werr := <-failed:
					yield(ai.Segment{}, werr)
				default:
					switch {
					case parent.Err() != nil:
						yield(ai.Segment{}, fmt.Errorf("deepgram: read live transcription: %w", parent.Err()))
					case !l.finished(err):
						yield(ai.Segment{}, fmt.Errorf("deepgram: read live transcription: %w", err))
					}
				}
				return
			}
			if kind != websocket.MessageText {
				continue
			}
			segs, err := decodeLiveMessage(data)
			if err != nil {
				yield(ai.Segment{}, err)
				return
			}
			for _, seg := range segs {
				if !yield(seg, nil) {
					return
				}
			}
		}
	}
}

func (l *liveStream) finished(err error) bool {
	status := websocket.CloseStatus(err)
	return l.closeSent.Load() && (status == websocket.StatusNormalClosure || status == websocket.StatusNoStatusRcvd)
}

func (l *liveStream) pump(ctx context.Context) error {
	buf := make([]byte, liveAudioChunk)
	for {
		n, err := l.audio.Read(buf)
		if n > 0 {
			if werr := l.ws.Write(ctx, websocket.MessageBinary, buf[:n]); werr != nil {
				return fmt.Errorf("deepgram: send audio: %w", werr)
			}
			l.sent.Store(true)
		}
		if errors.Is(err, io.EOF) {
			l.closeSent.Store(true)
			if werr := l.ws.Write(ctx, websocket.MessageText, []byte(closeStreamMessage)); werr != nil {
				return fmt.Errorf("deepgram: send CloseStream: %w", werr)
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("deepgram: read audio: %w", err)
		}
	}
}

func (l *liveStream) pingLoop(ctx context.Context, stop <-chan struct{}) {
	t := time.NewTicker(l.keepAlive)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			return
		case <-t.C:
			if l.sent.Swap(false) {
				continue
			}
			if err := l.ws.Write(ctx, websocket.MessageText, []byte(keepAliveMessage)); err != nil {
				return
			}
		}
	}
}

func decodeLiveMessage(data []byte) ([]ai.Segment, error) {
	var msg liveMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("deepgram: decode live message: %w", err)
	}
	switch msg.Type {
	case "Results":
		var ch struct {
			Alternatives []struct {
				Transcript string `json:"transcript"`
				Words      []word `json:"words"`
			} `json:"alternatives"`
		}
		if err := json.Unmarshal(msg.Channel, &ch); err != nil {
			return nil, fmt.Errorf("deepgram: decode live results: %w", err)
		}
		if len(ch.Alternatives) == 0 || ch.Alternatives[0].Transcript == "" {
			return nil, nil
		}
		alt := ch.Alternatives[0]
		segs := speakerRuns(alt.Words)
		if len(segs) == 0 {
			segs = []ai.Segment{{Text: alt.Transcript, Start: seconds(msg.Start), End: seconds(msg.Start + msg.Duration)}}
		}
		for i := range segs {
			segs[i].Partial = !msg.IsFinal
		}
		return segs, nil
	case "Error":
		code := msg.Variant
		if code == "" {
			code = msg.Code
		}
		return nil, transport.ErrorFromEvent(providerName, code, msg.Description)
	default:
		return nil, nil
	}
}

var _ ai.TranscriptionStreamModel = (*TranscriptionModel)(nil)
