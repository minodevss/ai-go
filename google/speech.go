package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	defaultSpeechVoice = "Kore"
	defaultSampleRate  = 24000
	wavHeaderBytes     = 44
)

// SpeechOptions are speech settings passed with ai.Providers under "google".
// Turns needs up to two Speakers and replaces the call's Text, which must still be set; SampleRate applies to Gemini 3.8 models.
type SpeechOptions struct {
	Speakers   []SpeechSpeaker `json:"speakers,omitzero"`
	Turns      []SpeechTurn    `json:"turns,omitzero"`
	SampleRate int             `json:"sampleRate,omitzero"`
}

func (SpeechOptions) ProviderName() string { return providerName }

type SpeechSpeaker struct {
	Name  string `json:"name"`
	Voice string `json:"voice"`
}

type SpeechTurn struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
	Style   string `json:"style,omitzero"`
}

// SpeechModel synthesizes speech with a Gemini TTS model. Voice defaults to "Kore" and Format to "wav";
// "pcm", "mulaw" and "alaw" need a Gemini 3.8 model.
type SpeechModel struct {
	backend *backend
	id      string
}

type speechRequest struct {
	Contents         []content        `json:"contents"`
	GenerationConfig speechGeneration `json:"generationConfig"`
}

type speechGeneration struct {
	ResponseModalities []string        `json:"responseModalities"`
	SpeechConfig       speechConfig    `json:"speechConfig"`
	ResponseFormat     *responseFormat `json:"responseFormat,omitzero"`
}

type speechConfig struct {
	VoiceConfig  *voiceConfig       `json:"voiceConfig,omitzero"`
	MultiSpeaker *multiSpeakerVoice `json:"multiSpeakerVoiceConfig,omitzero"`
}

type voiceConfig struct {
	Voice    string         `json:"voice,omitzero"`
	Prebuilt *prebuiltVoice `json:"prebuiltVoiceConfig,omitzero"`
}

type prebuiltVoice struct {
	VoiceName string `json:"voiceName"`
}

type multiSpeakerVoice struct {
	SpeakerVoiceConfigs []speakerVoice `json:"speakerVoiceConfigs"`
}

type speakerVoice struct {
	Speaker     string      `json:"speaker"`
	VoiceConfig voiceConfig `json:"voiceConfig"`
}

type responseFormat struct {
	Audio audioFormat `json:"audio"`
}

type audioFormat struct {
	MimeType   string `json:"mimeType"`
	SampleRate int    `json:"sampleRate,omitzero"`
}

type speechPart struct {
	Text     string          `json:"text"`
	Metadata *speechMetadata `json:"speech_metadata,omitzero"`
}

type speechMetadata struct {
	Style   string `json:"style,omitzero"`
	Speaker string `json:"speaker,omitzero"`
}

func (p *Provider) SpeechModel(id string) *SpeechModel {
	return &SpeechModel{backend: p.backend, id: id}
}

func (m *SpeechModel) Provider() string { return m.backend.name }

func (m *SpeechModel) ModelID() string { return m.id }

func (m *SpeechModel) GenerateSpeech(ctx context.Context, c *ai.SpeechCall) (*ai.SpeechResponse, error) {
	opts, err := decodeOptions[SpeechOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	req, format, warnings, err := m.request(c, opts)
	if err != nil {
		return nil, err
	}
	var body responseBody
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":generateContent", Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, fail(err)
	}
	if err := body.failure(m.backend); err != nil {
		return nil, err
	}
	audio, err := m.audio(body, format)
	if err != nil {
		return nil, err
	}
	return &ai.SpeechResponse{Audio: audio, Warnings: warnings, Header: header}, nil
}

func (m *SpeechModel) legacy() bool {
	id := strings.TrimPrefix(m.id, "models/")
	return strings.HasPrefix(id, "gemini-2.5") || strings.HasPrefix(id, "gemini-3.1")
}

func (m *SpeechModel) request(c *ai.SpeechCall, opts SpeechOptions) (*speechRequest, string, []ai.Warning, error) {
	var warnings []ai.Warning
	if c.Speed != nil {
		warnings = append(warnings, ai.Warning{Message: "google: speed is not supported by Gemini TTS and was ignored; describe the pace in Instructions"})
	}
	format := strings.ToLower(c.Format)
	gen := speechGeneration{ResponseModalities: []string{"AUDIO"}}
	if !m.legacy() {
		wire, err := wireFormat(format)
		if err != nil {
			return nil, "", nil, fmt.Errorf("google: %w", err)
		}
		if wire != "" || opts.SampleRate > 0 {
			gen.ResponseFormat = &responseFormat{Audio: audioFormat{MimeType: cmp.Or(wire, "AUDIO_WAV"), SampleRate: opts.SampleRate}}
		}
	} else if format != "" && format != "wav" && format != "pcm" {
		return nil, "", nil, fmt.Errorf("google: model %s returns PCM or WAV, not %q", m.id, c.Format)
	}
	voice := cmp.Or(c.Voice, defaultSpeechVoice)
	if len(opts.Speakers) > 0 {
		cfg, err := m.speakers(opts.Speakers)
		if err != nil {
			return nil, "", nil, err
		}
		gen.SpeechConfig.MultiSpeaker = cfg
	} else {
		gen.SpeechConfig.VoiceConfig = m.voice(voice)
	}
	parts, err := m.parts(c, opts)
	if err != nil {
		return nil, "", nil, err
	}
	return &speechRequest{Contents: []content{{Role: "user", Parts: parts}}, GenerationConfig: gen}, format, warnings, nil
}

func (m *SpeechModel) voice(name string) *voiceConfig {
	if m.legacy() {
		return &voiceConfig{Prebuilt: &prebuiltVoice{VoiceName: name}}
	}
	return &voiceConfig{Voice: name}
}

func (m *SpeechModel) speakers(speakers []SpeechSpeaker) (*multiSpeakerVoice, error) {
	if len(speakers) != 2 {
		return nil, fmt.Errorf("google: multi-speaker speech takes exactly 2 speakers, got %d", len(speakers))
	}
	cfg := &multiSpeakerVoice{}
	for _, s := range speakers {
		if s.Name == "" || s.Voice == "" {
			return nil, errors.New("google: a speaker needs a name and a voice")
		}
		cfg.SpeakerVoiceConfigs = append(cfg.SpeakerVoiceConfigs, speakerVoice{Speaker: s.Name, VoiceConfig: voiceConfig{Prebuilt: &prebuiltVoice{VoiceName: s.Voice}}})
	}
	return cfg, nil
}

func (m *SpeechModel) parts(c *ai.SpeechCall, opts SpeechOptions) ([]json.RawMessage, error) {
	if len(opts.Turns) > 0 {
		if len(opts.Speakers) == 0 {
			return nil, errors.New("google: speech turns need speakers")
		}
		return m.turnParts(opts.Turns), nil
	}
	if c.Text == "" {
		return nil, errors.New("google: speech needs text")
	}
	if m.legacy() {
		text := c.Text
		if c.Instructions != "" {
			text = c.Instructions + ": " + text
		}
		return []json.RawMessage{pdata.JSON(providerName, speechPart{Text: text})}, nil
	}
	part := speechPart{Text: c.Text}
	if c.Instructions != "" {
		part.Metadata = &speechMetadata{Style: c.Instructions}
	}
	return []json.RawMessage{pdata.JSON(providerName, part)}, nil
}

func (m *SpeechModel) turnParts(turns []SpeechTurn) []json.RawMessage {
	parts := make([]json.RawMessage, 0, len(turns))
	if m.legacy() {
		var lines []string
		for _, t := range turns {
			lines = append(lines, t.Speaker+": "+t.Text)
		}
		return append(parts, pdata.JSON(providerName, speechPart{Text: strings.Join(lines, "\n")}))
	}
	for _, t := range turns {
		parts = append(parts, pdata.JSON(providerName, speechPart{Text: t.Text, Metadata: &speechMetadata{Speaker: t.Speaker, Style: t.Style}}))
	}
	return parts
}

func wireFormat(format string) (string, error) {
	switch format {
	case "", "wav":
		return "", nil
	case "pcm", "l16":
		return "AUDIO_L16", nil
	case "mulaw":
		return "AUDIO_MULAW", nil
	case "alaw":
		return "AUDIO_ALAW", nil
	default:
		return "", fmt.Errorf("google: unknown speech format %q; use wav, pcm, mulaw or alaw", format)
	}
}

func (m *SpeechModel) audio(body responseBody, format string) (ai.File, error) {
	for _, raw := range body.first().Content.Parts {
		var p partProbe
		if err := json.Unmarshal(raw, &p); err != nil {
			return ai.File{}, fmt.Errorf("google: decode speech part: %w", err)
		}
		if p.InlineData == nil || !strings.HasPrefix(p.InlineData.MimeType, "audio/") {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return ai.File{}, fmt.Errorf("google: decode speech data: %w", err)
		}
		mediaType := p.InlineData.MimeType
		if m.legacy() && format != "pcm" && isPCM(mediaType) {
			return ai.File{MediaType: "audio/wav", Data: wrapWAV(data, rateOf(mediaType))}, nil
		}
		return ai.File{MediaType: mediaType, Data: data}, nil
	}
	reason := cmp.Or(body.first().FinishReason, blockReason(body.PromptFeedback))
	if reason == "" {
		return ai.File{}, errors.New("google: the response holds no audio")
	}
	return ai.File{}, fmt.Errorf("google: the response holds no audio (%s)", strings.ToLower(reason))
}

func isPCM(mediaType string) bool {
	base, _, err := mime.ParseMediaType(mediaType)
	return err == nil && (strings.EqualFold(base, "audio/l16") || strings.EqualFold(base, "audio/pcm"))
}

func rateOf(mediaType string) int {
	_, params, err := mime.ParseMediaType(mediaType)
	if err != nil {
		return defaultSampleRate
	}
	if rate, err := strconv.Atoi(params["rate"]); err == nil && rate > 0 {
		return rate
	}
	return defaultSampleRate
}

func wrapWAV(pcm []byte, rate int) []byte {
	out := make([]byte, wavHeaderBytes, wavHeaderBytes+len(pcm))
	copy(out[0:], "RIFF")
	binary.LittleEndian.PutUint32(out[4:], uint32(36+len(pcm)))
	copy(out[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:], 16)
	binary.LittleEndian.PutUint16(out[20:], 1)
	binary.LittleEndian.PutUint16(out[22:], 1)
	binary.LittleEndian.PutUint32(out[24:], uint32(rate))
	binary.LittleEndian.PutUint32(out[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(out[32:], 2)
	binary.LittleEndian.PutUint16(out[34:], 16)
	copy(out[36:], "data")
	binary.LittleEndian.PutUint32(out[40:], uint32(len(pcm)))
	return append(out, pcm...)
}

var _ ai.SpeechModel = (*SpeechModel)(nil)
