package openai

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/transport"
	"github.com/minodevss/ai-go/realtime"
)

const (
	defaultRealtimeRate = 24000
	errCancelNotActive  = "response_cancel_not_active"
)

type serverEvent struct {
	Type       string          `json:"type"`
	ItemID     string          `json:"item_id"`
	Delta      string          `json:"delta"`
	Transcript string          `json:"transcript"`
	Text       string          `json:"text"`
	Response   *serverResponse `json:"response"`
	Error      *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type serverResponse struct {
	Status        string `json:"status"`
	StatusDetails struct {
		Reason string `json:"reason"`
	} `json:"status_details"`
	Output []struct {
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"output"`
	Usage json.RawMessage `json:"usage"`
}

type wireState struct {
	mu          sync.Mutex
	active      bool
	interrupted bool
	calls       map[string]struct{}
}

func (w *wireState) expect(ids []string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.calls = make(map[string]struct{}, len(ids))
	for _, id := range ids {
		w.calls[id] = struct{}{}
	}
}

func (w *wireState) answered(id string) (last bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.calls, id)
	return len(w.calls) == 0
}

func (w *wireState) clearCalls() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.calls = nil
}

func decodeServerEvent(data []byte, w *wireState) ([]realtime.Event, error) {
	var e serverEvent
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, fmt.Errorf("openai: decode realtime event: %w", err)
	}
	switch e.Type {
	case "response.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(e.Delta)
		if err != nil {
			return nil, fmt.Errorf("openai: decode realtime audio: %w", err)
		}
		return []realtime.Event{realtime.AudioDelta{Data: audio}}, nil
	case "response.output_text.delta":
		return []realtime.Event{realtime.TextDelta{Text: e.Delta}}, nil
	case "response.output_audio_transcript.delta":
		return []realtime.Event{realtime.Transcript{Role: ai.RoleAssistant, Text: e.Delta}}, nil
	case "response.output_audio_transcript.done":
		return []realtime.Event{realtime.Transcript{Role: ai.RoleAssistant, Text: e.Transcript, Final: true}}, nil
	case "conversation.item.input_audio_transcription.delta":
		return []realtime.Event{realtime.Transcript{Role: ai.RoleUser, Text: e.Delta}}, nil
	case "conversation.item.input_audio_transcription.completed":
		return []realtime.Event{realtime.Transcript{Role: ai.RoleUser, Text: e.Transcript, Final: true}}, nil
	case "response.created":
		w.mu.Lock()
		w.active, w.interrupted = true, false
		w.mu.Unlock()
		return nil, nil
	case "input_audio_buffer.speech_started":
		w.mu.Lock()
		defer w.mu.Unlock()
		if !w.active || w.interrupted {
			return nil, nil
		}
		w.interrupted = true
		return []realtime.Event{realtime.Interrupted{}}, nil
	case "response.done":
		return decodeResponseDone(e, w), nil
	case "error":
		return nil, realtimeError(e)
	default:
		return nil, nil
	}
}

func decodeResponseDone(e serverEvent, w *wireState) []realtime.Event {
	if e.Response == nil {
		return nil
	}
	w.mu.Lock()
	wasInterrupted := w.interrupted
	w.active, w.interrupted = false, false
	w.mu.Unlock()
	var out []realtime.Event
	finish := ai.FinishStop
	switch e.Response.Status {
	case "canceled":
		w.clearCalls()
		if !wasInterrupted {
			out = append(out, realtime.Interrupted{})
		}
		finish = ai.FinishOther
	case "failed":
		finish = ai.FinishError
	case "incomplete":
		finish = ai.FinishLength
		if e.Response.StatusDetails.Reason == "content_filter" {
			finish = ai.FinishContentFilter
		}
	default:
		out = append(out, callEvents(e.Response, w)...)
	}
	return append(out, realtime.TurnEnd{Usage: openaiwire.Usage(e.Response.Usage), FinishReason: finish})
}

func callEvents(r *serverResponse, w *wireState) []realtime.Event {
	var out []realtime.Event
	var ids []string
	for _, item := range r.Output {
		if item.Type != "function_call" {
			continue
		}
		ids = append(ids, item.CallID)
		out = append(out, realtime.ToolCall{ID: item.CallID, Name: item.Name, Input: json.RawMessage(item.Arguments)})
	}
	w.expect(ids)
	return out
}

func realtimeError(e serverEvent) error {
	if e.Error == nil {
		return errors.New("openai: realtime error")
	}
	if e.Error.Code == errCancelNotActive {
		return nil
	}
	return transport.ErrorFromEvent(providerName, cmp.Or(e.Error.Code, e.Error.Type), e.Error.Message)
}

// RealtimeOptions are passed in realtime.Config.Provider. Without InputTranscription no user Transcript events arrive.
type RealtimeOptions struct {
	OutputModalities   []string `json:"outputModalities,omitzero"`
	InputTranscription string   `json:"inputTranscription,omitzero"`
	NoiseReduction     string   `json:"noiseReduction,omitzero"`
	MaxOutputTokens    int      `json:"maxOutputTokens,omitzero"`
	SafetyIdentifier   string   `json:"safetyIdentifier,omitzero"`
}

func (RealtimeOptions) ProviderName() string { return providerName }

type audioFormatWire struct {
	Type string `json:"type"`
	Rate int    `json:"rate,omitzero"`
}

type sessionWire struct {
	Type             string     `json:"type"`
	Model            string     `json:"model,omitzero"`
	Instructions     string     `json:"instructions,omitzero"`
	OutputModalities []string   `json:"output_modalities,omitzero"`
	Audio            *audioWire `json:"audio,omitzero"`
	Tools            []toolWire `json:"tools,omitzero"`
	ToolChoice       string     `json:"tool_choice,omitzero"`
	MaxOutputTokens  int        `json:"max_output_tokens,omitzero"`
}

type audioWire struct {
	Input  *audioInputWire  `json:"input,omitzero"`
	Output *audioOutputWire `json:"output,omitzero"`
}

type audioInputWire struct {
	Format         *audioFormatWire    `json:"format,omitzero"`
	Transcription  *transcriptionWire  `json:"transcription,omitzero"`
	NoiseReduction *noiseReductionWire `json:"noise_reduction,omitzero"`
	TurnDetection  json.RawMessage     `json:"turn_detection,omitzero"`
}

type audioOutputWire struct {
	Format *audioFormatWire `json:"format,omitzero"`
	Voice  any              `json:"voice,omitzero"`
}

type transcriptionWire struct {
	Model    string `json:"model"`
	Language string `json:"language,omitzero"`
	Prompt   string `json:"prompt,omitzero"`
}

type noiseReductionWire struct {
	Type string `json:"type"`
}

type toolWire struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters"`
}

func buildSession(c *realtime.Config, model string) (sessionWire, error) {
	opts, err := decodeOptions[RealtimeOptions](c.Provider)
	if err != nil {
		return sessionWire{}, err
	}
	in, err := audioFormat(c.InputAudio)
	if err != nil {
		return sessionWire{}, err
	}
	out, err := audioFormat(c.OutputAudio)
	if err != nil {
		return sessionWire{}, err
	}
	turn, err := turnDetection(c.TurnDetection)
	if err != nil {
		return sessionWire{}, err
	}
	s := sessionWire{
		Type:             "realtime",
		Model:            model,
		Instructions:     c.Instructions,
		OutputModalities: opts.OutputModalities,
		MaxOutputTokens:  opts.MaxOutputTokens,
		Audio: &audioWire{
			Input:  &audioInputWire{Format: in, TurnDetection: turn},
			Output: &audioOutputWire{Format: out},
		},
	}
	if c.Voice != "" {
		s.Audio.Output.Voice = voiceOf(c.Voice)
	}
	if opts.InputTranscription != "" {
		s.Audio.Input.Transcription = &transcriptionWire{Model: opts.InputTranscription}
	}
	if opts.NoiseReduction != "" {
		s.Audio.Input.NoiseReduction = &noiseReductionWire{Type: opts.NoiseReduction}
	}
	for _, t := range c.Tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		s.Tools = append(s.Tools, toolWire{Type: "function", Name: t.Name, Description: t.Description, Parameters: params})
	}
	if len(s.Tools) > 0 {
		s.ToolChoice = "auto"
	}
	return s, nil
}

func audioFormat(f realtime.AudioFormat) (*audioFormatWire, error) {
	switch f.Encoding {
	case "", "pcm16":
		if err := checkRealtimeRate(f.SampleRate); err != nil {
			return nil, err
		}
		return &audioFormatWire{Type: "audio/pcm", Rate: defaultRealtimeRate}, nil
	case "g711-ulaw":
		return &audioFormatWire{Type: "audio/pcmu"}, nil
	case "g711-alaw":
		return &audioFormatWire{Type: "audio/pcma"}, nil
	default:
		return nil, fmt.Errorf("openai: unsupported realtime audio encoding %q", f.Encoding)
	}
}

func checkRealtimeRate(rate int) error {
	if rate != 0 && rate != defaultRealtimeRate {
		return fmt.Errorf("openai: realtime pcm audio supports only %d Hz, got %d", defaultRealtimeRate, rate)
	}
	return nil
}

func turnDetection(t realtime.TurnDetection) (json.RawMessage, error) {
	switch t.Mode {
	case "":
		return nil, nil
	case realtime.TurnManual:
		return json.RawMessage("null"), nil
	case realtime.TurnServerVAD:
		cfg := map[string]any{"type": "server_vad"}
		if t.Threshold != nil {
			cfg["threshold"] = *t.Threshold
		}
		if t.SilenceDuration > 0 {
			cfg["silence_duration_ms"] = t.SilenceDuration.Milliseconds()
		}
		return transport.EncodeJSON(cfg)
	case realtime.TurnSemanticVAD:
		cfg := map[string]any{"type": "semantic_vad"}
		if t.Eagerness != "" {
			cfg["eagerness"] = t.Eagerness
		}
		return transport.EncodeJSON(cfg)
	default:
		return nil, fmt.Errorf("openai: unsupported turn detection mode %q", t.Mode)
	}
}

func clientEvent(typ string, fields map[string]any) ([]byte, error) {
	fields["type"] = typ
	return transport.EncodeJSON(fields)
}

func realtimeHeaders(opts RealtimeOptions) http.Header {
	h := http.Header{}
	if opts.SafetyIdentifier != "" {
		h.Set("OpenAI-Safety-Identifier", opts.SafetyIdentifier)
	}
	return h
}
