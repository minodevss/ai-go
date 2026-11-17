package google

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/realtime"
)

const (
	defaultInputRate  = 16000
	defaultOutputRate = 24000
	audioModality     = "AUDIO"
	startLow          = "START_SENSITIVITY_LOW"
	startHigh         = "START_SENSITIVITY_HIGH"
	endLow            = "END_SENSITIVITY_LOW"
	endHigh           = "END_SENSITIVITY_HIGH"
)

// RealtimeOptions are Live API session settings passed in realtime.Config.Provider with ai.Providers.
// ResponseModalities defaults to AUDIO and ContextWindowCompression lifts the 15 minute limit of audio sessions.
type RealtimeOptions struct {
	ResponseModalities       []string `json:"responseModalities,omitzero"`
	DisableTranscription     bool     `json:"disableTranscription,omitzero"`
	ContextWindowCompression bool     `json:"contextWindowCompression,omitzero"`
}

func (RealtimeOptions) ProviderName() string { return providerName }

type liveSetupMessage struct {
	Setup liveSetup `json:"setup"`
}

type liveSetup struct {
	Model                    string             `json:"model"`
	GenerationConfig         *liveGenerationCfg `json:"generationConfig,omitzero"`
	SystemInstruction        *liveContent       `json:"systemInstruction,omitzero"`
	Tools                    []any              `json:"tools,omitzero"`
	RealtimeInputConfig      *liveInputConfig   `json:"realtimeInputConfig,omitzero"`
	SessionResumption        *liveResumption    `json:"sessionResumption,omitzero"`
	ContextWindowCompression *liveCompression   `json:"contextWindowCompression,omitzero"`
	InputAudioTranscription  *struct{}          `json:"inputAudioTranscription,omitzero"`
	OutputAudioTranscription *struct{}          `json:"outputAudioTranscription,omitzero"`
}

type liveGenerationCfg struct {
	ResponseModalities []string          `json:"responseModalities,omitzero"`
	SpeechConfig       *liveSpeechConfig `json:"speechConfig,omitzero"`
}

type liveSpeechConfig struct {
	VoiceConfig struct {
		PrebuiltVoiceConfig struct {
			VoiceName string `json:"voiceName"`
		} `json:"prebuiltVoiceConfig"`
	} `json:"voiceConfig"`
}

type liveContent struct {
	Parts []liveTextPart `json:"parts"`
}

type liveTextPart struct {
	Text string `json:"text"`
}

type liveInputConfig struct {
	AutomaticActivityDetection liveActivityDetection `json:"automaticActivityDetection"`
}

type liveActivityDetection struct {
	StartOfSpeechSensitivity string `json:"startOfSpeechSensitivity,omitzero"`
	EndOfSpeechSensitivity   string `json:"endOfSpeechSensitivity,omitzero"`
	SilenceDurationMs        int    `json:"silenceDurationMs,omitzero"`
}

type liveResumption struct {
	Handle string `json:"handle,omitzero"`
}

type liveCompression struct {
	SlidingWindow struct{} `json:"slidingWindow"`
}

type liveConnectConfig struct {
	ResponseModalities []string          `json:"responseModalities,omitzero"`
	SpeechConfig       *liveSpeechConfig `json:"speechConfig,omitzero"`
	SystemInstruction  *liveContent      `json:"systemInstruction,omitzero"`
	Tools              []any             `json:"tools,omitzero"`
	SessionResumption  *liveResumption   `json:"sessionResumption,omitzero"`
	InputConfig        *liveInputConfig  `json:"realtimeInputConfig,omitzero"`
}

func liveSettings(c *realtime.Config, model string) (liveSetup, error) {
	opts, err := decodeRealtimeOptions(c.Provider)
	if err != nil {
		return liveSetup{}, err
	}
	if err := checkAudio(c); err != nil {
		return liveSetup{}, err
	}
	s := liveSetup{
		Model:             "models/" + strings.TrimPrefix(model, "models/"),
		GenerationConfig:  &liveGenerationCfg{ResponseModalities: liveModalities(opts)},
		SystemInstruction: liveInstruction(c.Instructions),
		SessionResumption: &liveResumption{Handle: c.Resume},
	}
	if c.Voice != "" {
		s.GenerationConfig.SpeechConfig = liveSpeech(c.Voice)
	}
	if s.Tools, err = liveTools(c.Tools); err != nil {
		return liveSetup{}, err
	}
	if s.RealtimeInputConfig, err = liveTurns(c.TurnDetection); err != nil {
		return liveSetup{}, err
	}
	if !opts.DisableTranscription {
		s.InputAudioTranscription = &struct{}{}
		s.OutputAudioTranscription = &struct{}{}
	}
	if opts.ContextWindowCompression {
		s.ContextWindowCompression = &liveCompression{}
	}
	return s, nil
}

func liveConstraints(c *realtime.Config) (liveConnectConfig, error) {
	opts, err := decodeRealtimeOptions(c.Provider)
	if err != nil {
		return liveConnectConfig{}, err
	}
	if err := checkAudio(c); err != nil {
		return liveConnectConfig{}, err
	}
	cfg := liveConnectConfig{
		ResponseModalities: liveModalities(opts),
		SystemInstruction:  liveInstruction(c.Instructions),
		SessionResumption:  &liveResumption{},
	}
	if c.Voice != "" {
		cfg.SpeechConfig = liveSpeech(c.Voice)
	}
	if cfg.Tools, err = liveTools(c.Tools); err != nil {
		return liveConnectConfig{}, err
	}
	if cfg.InputConfig, err = liveTurns(c.TurnDetection); err != nil {
		return liveConnectConfig{}, err
	}
	return cfg, nil
}

func decodeRealtimeOptions(d ai.ProviderData) (RealtimeOptions, error) {
	var o RealtimeOptions
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return RealtimeOptions{}, fmt.Errorf("google: decode realtime options: %w", err)
	}
	return o, nil
}

func liveModalities(o RealtimeOptions) []string {
	if len(o.ResponseModalities) > 0 {
		return o.ResponseModalities
	}
	return []string{audioModality}
}

func liveInstruction(text string) *liveContent {
	if text == "" {
		return nil
	}
	return &liveContent{Parts: []liveTextPart{{Text: text}}}
}

func liveSpeech(voice string) *liveSpeechConfig {
	var s liveSpeechConfig
	s.VoiceConfig.PrebuiltVoiceConfig.VoiceName = voice
	return &s
}

func checkAudio(c *realtime.Config) error {
	if enc := c.InputAudio.Encoding; enc != "" && enc != "pcm16" {
		return fmt.Errorf("google: unsupported realtime input audio encoding %q: the Live API takes 16-bit PCM", enc)
	}
	if enc := c.OutputAudio.Encoding; enc != "" && enc != "pcm16" {
		return fmt.Errorf("google: unsupported realtime output audio encoding %q: the Live API returns 16-bit PCM", enc)
	}
	if rate := c.OutputAudio.SampleRate; rate != 0 && rate != defaultOutputRate {
		return fmt.Errorf("google: unsupported realtime output sample rate %d: the Live API always returns %d Hz", rate, defaultOutputRate)
	}
	return nil
}

func liveTools(tools []ai.Tool) ([]any, error) {
	var entries []any
	var decls []functionDeclaration
	for _, t := range tools {
		if hosted, ok := hostedToolOf(t); ok {
			entries = append(entries, hosted)
			continue
		}
		decl, _, err := declaration(t, false)
		if err != nil {
			return nil, err
		}
		decls = append(decls, decl)
	}
	if len(decls) > 0 {
		entries = append(entries, map[string][]functionDeclaration{"functionDeclarations": decls})
	}
	return entries, nil
}

func liveTurns(t realtime.TurnDetection) (*liveInputConfig, error) {
	var d liveActivityDetection
	switch t.Mode {
	case "":
		if t.Threshold == nil && t.SilenceDuration == 0 && t.Eagerness == "" {
			return nil, nil
		}
	case realtime.TurnServerVAD, realtime.TurnSemanticVAD:
	case realtime.TurnManual:
		return nil, fmt.Errorf("%w: manual turn detection needs activity signals that realtime.Conn cannot send", realtime.ErrUnsupported)
	default:
		return nil, fmt.Errorf("google: unknown turn detection mode %q", t.Mode)
	}
	if t.Threshold != nil {
		d.StartOfSpeechSensitivity = startHigh
		if *t.Threshold >= 0.5 {
			d.StartOfSpeechSensitivity = startLow
		}
	}
	switch strings.ToLower(t.Eagerness) {
	case "":
	case "low":
		d.EndOfSpeechSensitivity = endLow
	case "high":
		d.EndOfSpeechSensitivity = endHigh
	default:
		return nil, fmt.Errorf("google: unknown turn detection eagerness %q", t.Eagerness)
	}
	d.SilenceDurationMs = int(t.SilenceDuration / time.Millisecond)
	return &liveInputConfig{AutomaticActivityDetection: d}, nil
}

type liveServerMessage struct {
	SetupComplete           *struct{}         `json:"setupComplete"`
	ServerContent           *liveServerBody   `json:"serverContent"`
	ToolCall                *liveToolCall     `json:"toolCall"`
	ToolCallCancellation    *liveCancellation `json:"toolCallCancellation"`
	GoAway                  *liveGoAway       `json:"goAway"`
	SessionResumptionUpdate *liveResumeUpdate `json:"sessionResumptionUpdate"`
	UsageMetadata           json.RawMessage   `json:"usageMetadata"`
}

type liveServerBody struct {
	ModelTurn           *liveModelTurn     `json:"modelTurn"`
	TurnComplete        bool               `json:"turnComplete"`
	Interrupted         bool               `json:"interrupted"`
	InputTranscription  *liveTranscription `json:"inputTranscription"`
	OutputTranscription *liveTranscription `json:"outputTranscription"`
}

type liveModelTurn struct {
	Parts []liveServerPart `json:"parts"`
}

type liveServerPart struct {
	Text       string `json:"text"`
	Thought    bool   `json:"thought"`
	InlineData *struct {
		Data string `json:"data"`
	} `json:"inlineData"`
}

type liveTranscription struct {
	Text string `json:"text"`
}

type liveToolCall struct {
	FunctionCalls []struct {
		ID   string          `json:"id"`
		Name string          `json:"name"`
		Args json.RawMessage `json:"args"`
	} `json:"functionCalls"`
}

type liveCancellation struct {
	IDs []string `json:"ids"`
}

type liveGoAway struct {
	TimeLeft string `json:"timeLeft"`
}

type liveResumeUpdate struct {
	NewHandle string `json:"newHandle"`
	Resumable bool   `json:"resumable"`
}

type liveUsageBody struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ResponseTokenCount      int `json:"responseTokenCount"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
}

type liveState struct {
	mu     sync.Mutex
	input  strings.Builder
	output strings.Builder
	usage  ai.Usage
	names  map[string]string
}

var errSetupFailed = errors.New("google: live session did not start with setupComplete")

func (s *liveState) nameOf(id string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.names[id]
}

func decodeLiveMessage(data []byte, s *liveState) ([]realtime.Event, error) {
	var m liveServerMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("google: decode live message: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var events []realtime.Event
	if len(m.UsageMetadata) > 0 && string(m.UsageMetadata) != "null" {
		s.usage = liveUsage(m.UsageMetadata)
	}
	if m.ServerContent != nil {
		evs, err := s.content(m.ServerContent)
		if err != nil {
			return nil, err
		}
		events = append(events, evs...)
	}
	if m.ToolCall != nil {
		if s.names == nil {
			s.names = map[string]string{}
		}
		for _, fc := range m.ToolCall.FunctionCalls {
			id := fc.ID
			if id == "" {
				id = fc.Name
			}
			s.names[id] = fc.Name
			events = append(events, realtime.ToolCall{ID: id, Name: fc.Name, Input: objectArgs(fc.Args)})
		}
	}
	if m.ToolCallCancellation != nil {
		for _, id := range m.ToolCallCancellation.IDs {
			delete(s.names, id)
			events = append(events, realtime.ToolCanceled{CallID: id})
		}
	}
	if m.GoAway != nil {
		after, err := time.ParseDuration(m.GoAway.TimeLeft)
		if err != nil {
			return nil, fmt.Errorf("google: decode goAway timeLeft: %w", err)
		}
		events = append(events, realtime.GoAway{After: after})
	}
	if u := m.SessionResumptionUpdate; u != nil && u.Resumable && u.NewHandle != "" {
		events = append(events, realtime.Resumable{Handle: u.NewHandle})
	}
	return events, nil
}

func (s *liveState) content(c *liveServerBody) ([]realtime.Event, error) {
	var events []realtime.Event
	if c.ModelTurn != nil {
		for _, p := range c.ModelTurn.Parts {
			ev, err := livePart(p)
			if err != nil {
				return nil, err
			}
			if ev != nil {
				events = append(events, s.flushInput(), ev)
			}
		}
	}
	if t := c.InputTranscription; t != nil && t.Text != "" {
		s.input.WriteString(t.Text)
		events = append(events, realtime.Transcript{Role: ai.RoleUser, Text: t.Text})
	}
	if t := c.OutputTranscription; t != nil && t.Text != "" {
		events = append(events, s.flushInput())
		s.output.WriteString(t.Text)
		events = append(events, realtime.Transcript{Role: ai.RoleAssistant, Text: t.Text})
	}
	if c.Interrupted {
		events = append(events, s.flushInput(), s.flushOutput(), realtime.Interrupted{})
	}
	if c.TurnComplete {
		events = append(events, s.flushInput(), s.flushOutput(), realtime.TurnEnd{Usage: s.usage, FinishReason: ai.FinishStop})
		s.usage = ai.Usage{}
	}
	return dropNil(events), nil
}

func (s *liveState) flushInput() realtime.Event {
	return flush(&s.input, ai.RoleUser)
}

func (s *liveState) flushOutput() realtime.Event {
	return flush(&s.output, ai.RoleAssistant)
}

func flush(b *strings.Builder, role ai.Role) realtime.Event {
	if b.Len() == 0 {
		return nil
	}
	text := b.String()
	b.Reset()
	return realtime.Transcript{Role: role, Text: text, Final: true}
}

func dropNil(events []realtime.Event) []realtime.Event {
	out := events[:0]
	for _, ev := range events {
		if ev != nil {
			out = append(out, ev)
		}
	}
	return out
}

func livePart(p liveServerPart) (realtime.Event, error) {
	switch {
	case p.InlineData != nil && p.InlineData.Data != "":
		audio, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return nil, fmt.Errorf("google: decode live audio: %w", err)
		}
		return realtime.AudioDelta{Data: audio}, nil
	case p.Text != "" && !p.Thought:
		return realtime.TextDelta{Text: p.Text}, nil
	default:
		return nil, nil
	}
}

func objectArgs(args json.RawMessage) json.RawMessage {
	if len(args) == 0 || string(args) == "null" {
		return json.RawMessage("{}")
	}
	return args
}

func liveUsage(raw json.RawMessage) ai.Usage {
	var u liveUsageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{
		Input:     u.PromptTokenCount + u.ToolUsePromptTokenCount,
		Output:    u.ResponseTokenCount + u.ThoughtsTokenCount,
		CacheRead: u.CachedContentTokenCount,
		Reasoning: u.ThoughtsTokenCount,
		Raw:       raw,
	}
}
