package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/emit"
	"github.com/minodevss/ai-go/internal/mediaio"
)

const (
	defaultLiveFormat  = "pcm_16000"
	defaultCommit      = "vad"
	liveChunkMillis    = 100
	manualCommit       = "manual"
	liveSettle         = 750 * time.Millisecond
	typeCommitted      = "committed_transcript"
	typeCommittedStamp = "committed_transcript_with_timestamps"
)

type liveChunk struct {
	MessageType string `json:"message_type"`
	Audio       string `json:"audio_base_64"`
	Commit      bool   `json:"commit"`
	SampleRate  int    `json:"sample_rate"`
}

type liveMessage struct {
	MessageType string `json:"message_type"`
	Text        string `json:"text"`
	Error       string `json:"error"`
	Words       []struct {
		Start     *float64 `json:"start"`
		End       *float64 `json:"end"`
		Type      string   `json:"type"`
		SpeakerID string   `json:"speaker_id"`
	} `json:"words"`
}

func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	format, err := liveFormat(c.MediaType, opts.AudioFormat)
	if err != nil {
		return nil, err
	}
	rate, err := formatRate(format)
	if err != nil {
		return nil, err
	}
	timestamps := opts.IncludeTimestamps != nil && *opts.IncludeTimestamps
	conn, err := dialWS(ctx, m.client, "/v1/speech-to-text/realtime"+queryString(m.liveQuery(c, opts, format)))
	if err != nil {
		return nil, err
	}
	settle := liveSettle
	if opts.CommitStrategy == manualCommit {
		settle = 0
	}
	return &ai.TranscribeStreamResponse{Segments: liveSegments(ctx, conn, c.Audio, format, rate, timestamps, settle)}, nil
}

func (m *TranscriptionModel) liveQuery(c *ai.TranscribeCall, opts TranscriptionOptions, format string) url.Values {
	q := url.Values{"model_id": {liveModelID(m.id)}, "audio_format": {format}, "commit_strategy": {defaultCommit}}
	if c.Language != "" {
		q.Set("language_code", c.Language)
	}
	if opts.CommitStrategy != "" {
		q.Set("commit_strategy", opts.CommitStrategy)
	}
	for k, v := range map[string]*bool{"include_timestamps": opts.IncludeTimestamps, "diarize": opts.Diarize, "no_verbatim": opts.NoVerbatim, "enable_logging": opts.EnableLogging} {
		if v != nil {
			q.Set(k, strconv.FormatBool(*v))
		}
	}
	return q
}

func liveModelID(id string) string {
	switch id {
	case "scribe_v2":
		return "scribe_v2_realtime"
	case "scribe_v3":
		return "scribe_v3_live"
	default:
		return id
	}
}

func liveFormat(mediaType, explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	base, params, _ := strings.Cut(mediaType, ";")
	switch strings.TrimSpace(base) {
	case "", "audio/pcm", "audio/l16":
		rate := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(params), "rate="))
		if rate == "" || !strings.HasPrefix(strings.TrimSpace(params), "rate=") {
			return defaultLiveFormat, nil
		}
		return "pcm_" + rate, nil
	case "audio/basic", "audio/x-mulaw", "audio/ulaw":
		return "ulaw_8000", nil
	default:
		return "", fmt.Errorf("elevenlabs: live transcription needs raw PCM or mu-law audio, got media type %q; set MediaType to audio/pcm;rate=16000", mediaType)
	}
}

func formatRate(format string) (int, error) {
	_, rate, ok := strings.Cut(format, "_")
	n, err := strconv.Atoi(rate)
	if !ok || err != nil || n <= 0 {
		return 0, fmt.Errorf("elevenlabs: invalid audio format %q, use pcm_16000 or ulaw_8000", format)
	}
	return n, nil
}

func liveSegments(ctx context.Context, conn *websocket.Conn, audio io.Reader, format string, rate int, timestamps bool, settle time.Duration) iter.Seq2[ai.Segment, error] {
	return func(yield func(ai.Segment, error) bool) {
		yield = emit.New(yield)
		ctx, cancel := context.WithCancel(ctx)
		sent := make(chan error, 1)
		var (
			wg        sync.WaitGroup
			inputDone atomic.Bool
			closing   atomic.Bool
		)
		defer func() {
			cancel()
			closing.Store(true)
			if closer, ok := audio.(io.Closer); ok {
				if err := closer.Close(); err != nil {
					yield(ai.Segment{}, fmt.Errorf("elevenlabs: close audio: %w", err))
				}
				wg.Wait()
			}
			if err := conn.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				yield(ai.Segment{}, fmt.Errorf("elevenlabs: close live transcription: %w", err))
			}
		}()
		wg.Go(func() {
			if err := pumpAudio(ctx, conn, audio, format, rate, &inputDone); err != nil {
				sent <- err
				cancel()
			}
		})
		finalSeen := false
		for {
			readCtx, stopRead := ctx, func() {}
			if finalSeen {
				readCtx, stopRead = context.WithTimeout(ctx, settle)
			}
			_, data, err := conn.Read(readCtx)
			stopRead()
			if err != nil {
				if finalSeen && ctx.Err() == nil {
					return
				}
				select {
				case perr := <-sent:
					yield(ai.Segment{}, perr)
				default:
					if !closing.Load() && websocket.CloseStatus(err) != websocket.StatusNormalClosure {
						yield(ai.Segment{}, fmt.Errorf("elevenlabs: read live transcription: %w", err))
					}
				}
				return
			}
			seg, committed, err := decodeLiveMessage(data, timestamps)
			if err != nil {
				yield(ai.Segment{}, err)
				return
			}
			if seg != nil && !yield(*seg, nil) {
				return
			}
			if committed && inputDone.Load() {
				if settle == 0 {
					return
				}
				finalSeen = true
			}
		}
	}
}

func pumpAudio(ctx context.Context, conn *websocket.Conn, audio io.Reader, format string, rate int, inputDone *atomic.Bool) error {
	bytesPerSample := 2
	if strings.HasPrefix(format, "ulaw") {
		bytesPerSample = 1
	}
	buf := make([]byte, rate*bytesPerSample*liveChunkMillis/1000)
	var held []byte
	send := func(data []byte, commit bool) error {
		if commit {
			inputDone.Store(true)
		}
		return writeJSON(ctx, conn, liveChunk{MessageType: "input_audio_chunk", Audio: base64.StdEncoding.EncodeToString(data), Commit: commit, SampleRate: rate})
	}
	for {
		n, err := audio.Read(buf)
		if n > 0 {
			if held != nil {
				if serr := send(held, false); serr != nil {
					return serr
				}
			}
			held = append(held[:0], buf[:n]...)
		}
		if errors.Is(err, io.EOF) {
			return send(held, true)
		}
		if err != nil {
			return fmt.Errorf("elevenlabs: read audio: %w", err)
		}
	}
}

func decodeLiveMessage(data []byte, timestamps bool) (seg *ai.Segment, committed bool, err error) {
	var m liveMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, false, fmt.Errorf("elevenlabs: decode live transcription message: %w", err)
	}
	switch m.MessageType {
	case "partial_transcript":
		return &ai.Segment{Text: m.Text, Partial: true}, false, nil
	case typeCommitted:
		if timestamps {
			return nil, false, nil
		}
		return &ai.Segment{Text: m.Text}, true, nil
	case typeCommittedStamp:
		s := &ai.Segment{Text: m.Text}
		first := true
		for _, w := range m.Words {
			if w.Type == "spacing" {
				continue
			}
			if first && w.Start != nil {
				s.Start = seconds(*w.Start)
			}
			if w.End != nil {
				s.End = seconds(*w.End)
			}
			if s.Speaker == "" {
				s.Speaker = w.SpeakerID
			}
			first = false
		}
		return s, true, nil
	case "session_started", "warning", "":
		return nil, false, nil
	default:
		if m.Error != "" {
			return nil, false, liveError(m.MessageType, m.Error, data)
		}
		return nil, false, nil
	}
}

func liveError(kind, message string, body []byte) *ai.APIError {
	status := 0
	switch kind {
	case "auth_error":
		status = http.StatusUnauthorized
	case "quota_exceeded", "rate_limited", "commit_throttled", "queue_overflow", "resource_exhausted":
		status = http.StatusTooManyRequests
	case "input_error", "invalid_request", "chunk_size_exceeded", "insufficient_audio_activity":
		status = http.StatusBadRequest
	case "error", "transcriber_error":
		status = http.StatusInternalServerError
	}
	return &ai.APIError{Provider: providerName, StatusCode: status, Code: kind, Message: message, Body: string(body)}
}

var _ ai.TranscriptionStreamModel = (*TranscriptionModel)(nil)
