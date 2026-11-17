package elevenlabs

import (
	"context"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxSpeechBytes  = 256 << 20
	speechReadChunk = 16 << 10
)

// SpeechModel synthesizes speech with ElevenLabs text-to-speech models such as eleven_multilingual_v2. SpeechRequest.Voice is a required voice id and Format an output_format such as mp3_44100_128.
// StreamSpeech sends text over a stream-input WebSocket while it arrives, so Text must end when ctx ends. Instructions are not supported and produce a warning.
type SpeechModel struct {
	client *transport.Client
	id     string
}

type speechBody struct {
	Text                   string                    `json:"text"`
	ModelID                string                    `json:"model_id,omitzero"`
	LanguageCode           string                    `json:"language_code,omitzero"`
	VoiceSettings          *VoiceSettings            `json:"voice_settings,omitzero"`
	Dictionaries           []PronunciationDictionary `json:"pronunciation_dictionary_locators,omitzero"`
	Seed                   *int64                    `json:"seed,omitzero"`
	PreviousText           string                    `json:"previous_text,omitzero"`
	NextText               string                    `json:"next_text,omitzero"`
	ApplyTextNormalization string                    `json:"apply_text_normalization,omitzero"`
}

func (p *Provider) SpeechModel(id string) *SpeechModel {
	return &SpeechModel{client: p.client, id: id}
}

func NewSpeechModel(id string) *SpeechModel { return New().SpeechModel(id) }

func (m *SpeechModel) Provider() string { return providerName }

func (m *SpeechModel) ModelID() string { return m.id }

func (m *SpeechModel) GenerateSpeech(ctx context.Context, c *ai.SpeechCall) (*ai.SpeechResponse, error) {
	opts, err := mediaio.DecodeOptions[SpeechOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	format, err := parseFormat(c.Format)
	if err != nil {
		return nil, err
	}
	resp, err := m.speak(ctx, "", c.Voice, c.Text, c.Speed, format, opts, c.Headers)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxSpeechBytes+1))
	if err != nil {
		return nil, fmt.Errorf("elevenlabs: read audio: %w", err)
	}
	if len(data) > maxSpeechBytes {
		return nil, fmt.Errorf("elevenlabs: audio exceeds the %d MB limit", maxSpeechBytes>>20)
	}
	return &ai.SpeechResponse{Audio: ai.File{MediaType: format.mediaType, Data: data}, Warnings: speechWarnings(c.Instructions), Header: resp.Header}, nil
}

func (m *SpeechModel) StreamSpeech(ctx context.Context, c *ai.SpeechStreamCall) (*ai.SpeechStreamResponse, error) {
	opts, err := mediaio.DecodeOptions[SpeechOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	format, err := parseFormat(c.Format)
	if err != nil {
		return nil, err
	}
	if c.Voice == "" {
		return nil, errNoVoice
	}
	var audio iter.Seq2[[]byte, error]
	switch opts.StreamTransport {
	case "", StreamWebSocket:
		audio = m.streamInput(ctx, c, format, opts)
	case StreamHTTP:
		audio = m.streamHTTP(ctx, c, format, opts)
	default:
		return nil, fmt.Errorf("elevenlabs: unknown stream transport %q", opts.StreamTransport)
	}
	return &ai.SpeechStreamResponse{Audio: audio, MediaType: format.mediaType}, nil
}

func (m *SpeechModel) streamHTTP(ctx context.Context, c *ai.SpeechStreamCall, format outputFormat, opts SpeechOptions) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		var text strings.Builder
		for piece := range c.Text {
			text.WriteString(piece)
		}
		resp, err := m.speak(ctx, "/stream", c.Voice, text.String(), c.Speed, format, opts, c.Headers)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()
		buf := make([]byte, speechReadChunk)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 && !yield(append([]byte(nil), buf[:n]...), nil) {
				return
			}
			if err == io.EOF {
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("elevenlabs: read audio stream: %w", err))
				return
			}
		}
	}
}

func (m *SpeechModel) speak(ctx context.Context, suffix, voice, text string, speed *float64, format outputFormat, opts SpeechOptions, headers http.Header) (*http.Response, error) {
	if voice == "" {
		return nil, errNoVoice
	}
	q := url.Values{}
	if format.query != "" {
		q.Set("output_format", format.query)
	}
	if opts.EnableLogging != nil {
		q.Set("enable_logging", strconv.FormatBool(*opts.EnableLogging))
	}
	body := speechBody{
		Text:                   text,
		ModelID:                m.id,
		LanguageCode:           opts.LanguageCode,
		VoiceSettings:          withSpeed(opts.VoiceSettings, speed),
		Dictionaries:           opts.PronunciationDictionaries,
		Seed:                   opts.Seed,
		PreviousText:           opts.PreviousText,
		NextText:               opts.NextText,
		ApplyTextNormalization: opts.ApplyTextNormalization,
	}
	h := headers.Clone()
	if h == nil {
		h = http.Header{}
	}
	h.Set("Accept", "audio/*")
	return m.client.Do(ctx, transport.Request{Path: "/v1/text-to-speech/" + url.PathEscape(voice) + suffix + queryString(q), Body: body, Header: h})
}

func withSpeed(vs *VoiceSettings, speed *float64) *VoiceSettings {
	if vs == nil && speed == nil {
		return nil
	}
	out := VoiceSettings{}
	if vs != nil {
		out = *vs
	}
	if speed != nil {
		out.Speed = speed
	}
	return &out
}

func speechWarnings(instructions string) []ai.Warning {
	if instructions == "" {
		return nil
	}
	return []ai.Warning{{Message: "elevenlabs: speech instructions are not supported and were ignored"}}
}

func queryString(q url.Values) string {
	if len(q) == 0 {
		return ""
	}
	return "?" + q.Encode()
}

var (
	_ ai.SpeechModel       = (*SpeechModel)(nil)
	_ ai.SpeechStreamModel = (*SpeechModel)(nil)
)
