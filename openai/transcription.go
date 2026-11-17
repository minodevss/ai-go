package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxTranscribeBytes = 25 << 20
	pcmMediaType       = "audio/pcm"
)

// TranscriptionOptions are passed with ai.Providers. Diarization models need ChunkingStrategy "auto" for audio longer than 30 seconds.
type TranscriptionOptions struct {
	Prompt           string   `json:"prompt,omitzero"`
	Temperature      *float64 `json:"temperature,omitzero"`
	ChunkingStrategy string   `json:"chunkingStrategy,omitzero"`
	Languages        []string `json:"languages,omitzero"`
	Keywords         []string `json:"keywords,omitzero"`
}

func (TranscriptionOptions) ProviderName() string { return providerName }

// TranscriptionModel streams 24 kHz mono PCM16 over a Realtime session when the MediaType is "audio/pcm". That path closes Audio when done if it is an io.Closer.
type TranscriptionModel struct {
	provider *Provider
	client   *transport.Client
	id       string
}

type transcriptionBody struct {
	Text      string `json:"text"`
	Language  string `json:"language"`
	Languages []struct {
		Code string `json:"code"`
	} `json:"languages"`
	Segments []transcriptSegment `json:"segments"`
}

type transcriptSegment struct {
	Text    string  `json:"text"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}

type transcriptEvent struct {
	Type    string  `json:"type"`
	Delta   string  `json:"delta"`
	Text    string  `json:"text"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
	Error   *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (p *Provider) TranscriptionModel(id string) *TranscriptionModel {
	return &TranscriptionModel{provider: p, client: p.client, id: id}
}

func (m *TranscriptionModel) Provider() string { return providerName }

func (m *TranscriptionModel) ModelID() string { return m.id }

func (m *TranscriptionModel) Transcribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeResponse, error) {
	req, err := m.request(c, false)
	if err != nil {
		return nil, err
	}
	var body transcriptionBody
	header, err := m.client.JSON(ctx, req, &body)
	if err != nil {
		return nil, err
	}
	return &ai.TranscribeResponse{Text: body.Text, Segments: segmentsOf(body.Segments), Language: languageOf(body), Header: header}, nil
}

func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	if strings.HasPrefix(c.MediaType, pcmMediaType) {
		return m.streamLive(ctx, c)
	}
	if strings.Contains(m.id, "whisper") {
		return nil, fmt.Errorf("openai: model %q does not support streaming transcription", m.id)
	}
	req, err := m.request(c, true)
	if err != nil {
		return nil, err
	}
	resp, err := m.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	return &ai.TranscribeStreamResponse{Segments: transcriptSegments(resp.Body), Header: resp.Header}, nil
}

func (m *TranscriptionModel) request(c *ai.TranscribeCall, stream bool) (transport.Request, error) {
	opts, err := decodeOptions[TranscriptionOptions](c.Provider)
	if err != nil {
		return transport.Request{}, err
	}
	audio, err := readLimited(c.Audio, maxTranscribeBytes, "audio")
	if err != nil {
		return transport.Request{}, err
	}
	f := form.New()
	f.File("file", mediaio.AudioName(c.MediaType), c.MediaType, bytes.NewReader(audio))
	f.Field("model", m.id)
	f.Field("language", c.Language)
	f.Field("prompt", opts.Prompt)
	if opts.Temperature != nil {
		f.Field("temperature", strconv.FormatFloat(*opts.Temperature, 'f', -1, 64))
	}
	f.Field("chunking_strategy", opts.ChunkingStrategy)
	for _, l := range opts.Languages {
		f.Field("languages[]", l)
	}
	for _, k := range opts.Keywords {
		f.Field("keywords[]", k)
	}
	switch {
	case strings.Contains(m.id, "diarize"):
		f.Field("response_format", "diarized_json")
	case strings.Contains(m.id, "whisper") && !stream:
		f.Field("response_format", "verbose_json")
		f.Field("timestamp_granularities[]", "segment")
	default:
		f.Field("response_format", "json")
	}
	if stream {
		f.Field("stream", "true")
	}
	req, err := f.Request("/audio/transcriptions")
	req.Header = c.Headers.Clone()
	if stream {
		if req.Header == nil {
			req.Header = http.Header{}
		}
		req.Header.Set("Accept", "text/event-stream")
	}
	return req, err
}

func transcriptSegments(body io.ReadCloser) iter.Seq2[ai.Segment, error] {
	return func(yield func(ai.Segment, error) bool) {
		defer body.Close()
		r := sse.NewReader(body)
		sawSegment := false
		for {
			ev, err := r.Next()
			if err != nil {
				if !errors.Is(err, io.EOF) {
					yield(ai.Segment{}, fmt.Errorf("openai: read transcription stream: %w", err))
				}
				return
			}
			seg, done, err := decodeTranscriptEvent([]byte(ev.Data), &sawSegment)
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

func decodeTranscriptEvent(data []byte, sawSegment *bool) (seg *ai.Segment, done bool, err error) {
	var e transcriptEvent
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, false, fmt.Errorf("openai: decode transcription event: %w", err)
	}
	switch e.Type {
	case "transcript.text.delta":
		return &ai.Segment{Text: e.Delta, Partial: true}, false, nil
	case "transcript.text.segment":
		*sawSegment = true
		return &ai.Segment{Text: e.Text, Start: seconds(e.Start), End: seconds(e.End), Speaker: e.Speaker}, false, nil
	case "transcript.text.done":
		if *sawSegment {
			return nil, true, nil
		}
		return &ai.Segment{Text: e.Text}, true, nil
	case "error":
		if e.Error != nil {
			return nil, false, transport.ErrorFromEvent(providerName, e.Error.Code, e.Error.Message)
		}
		return nil, false, errors.New("openai: transcription stream error")
	default:
		return nil, false, nil
	}
}

func segmentsOf(in []transcriptSegment) []ai.Segment {
	out := make([]ai.Segment, len(in))
	for i, s := range in {
		out[i] = ai.Segment{Text: s.Text, Start: seconds(s.Start), End: seconds(s.End), Speaker: s.Speaker}
	}
	return out
}

func languageOf(b transcriptionBody) string {
	if b.Language != "" {
		return b.Language
	}
	if len(b.Languages) > 0 {
		return b.Languages[0].Code
	}
	return ""
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

var (
	_ ai.TranscriptionModel       = (*TranscriptionModel)(nil)
	_ ai.TranscriptionStreamModel = (*TranscriptionModel)(nil)
)
