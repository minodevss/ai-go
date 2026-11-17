package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"time"
)

type ReasoningEffort string

const (
	ReasoningNone    ReasoningEffort = "none"
	ReasoningMinimal ReasoningEffort = "minimal"
	ReasoningLow     ReasoningEffort = "low"
	ReasoningMedium  ReasoningEffort = "medium"
	ReasoningHigh    ReasoningEffort = "high"
	ReasoningMax     ReasoningEffort = "max"
)

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice zero value means automatic.
type ToolChoice struct {
	mode ToolChoiceMode
	name string
}

func ToolRequired() ToolChoice { return ToolChoice{mode: ToolChoiceRequired} }

func ToolNone() ToolChoice { return ToolChoice{mode: ToolChoiceNone} }

func ToolNamed(name string) ToolChoice { return ToolChoice{mode: ToolChoiceNamed, name: name} }

func (c ToolChoice) Mode() ToolChoiceMode {
	if c.mode == "" {
		return ToolChoiceAuto
	}
	return c.mode
}

func (c ToolChoice) Name() string { return c.name }

type FinishReason string

const (
	FinishStop            FinishReason = "stop"
	FinishLength          FinishReason = "length"
	FinishToolCalls       FinishReason = "tool-calls"
	FinishContentFilter   FinishReason = "content-filter"
	FinishRefusal         FinishReason = "refusal"
	FinishStepLimit       FinishReason = "step-limit"
	FinishApprovalPending FinishReason = "approval-pending"
	FinishError           FinishReason = "error"
	FinishOther           FinishReason = "other"
	FinishPause           FinishReason = "pause"
)

type Warning struct {
	Message string `json:"message"`
}

type Usage struct {
	Input      int             `json:"input,omitzero"`
	Output     int             `json:"output,omitzero"`
	CacheRead  int             `json:"cacheRead,omitzero"`
	CacheWrite int             `json:"cacheWrite,omitzero"`
	Reasoning  int             `json:"reasoning,omitzero"`
	Raw        json.RawMessage `json:"raw,omitzero"`
}

func (u Usage) add(o Usage) Usage {
	return Usage{
		Input:      u.Input + o.Input,
		Output:     u.Output + o.Output,
		CacheRead:  u.CacheRead + o.CacheRead,
		CacheWrite: u.CacheWrite + o.CacheWrite,
		Reasoning:  u.Reasoning + o.Reasoning,
		Raw:        o.Raw,
	}
}

type Timeouts struct {
	Step       time.Duration
	FirstChunk time.Duration
	Chunk      time.Duration
	Tool       time.Duration
}

type StopCondition func(steps []Step) bool

func HasToolCall(name string) StopCondition {
	return func(steps []Step) bool {
		if len(steps) == 0 {
			return false
		}
		return slices.ContainsFunc(steps[len(steps)-1].ToolCalls, func(c ToolCall) bool { return c.Name == name })
	}
}

func MaxTotalTokens(n int) StopCondition {
	return func(steps []Step) bool {
		total := 0
		for _, s := range steps {
			total += s.Usage.Input + s.Usage.Output
		}
		return total >= n
	}
}

type Step struct {
	Number       int
	Content      Parts
	ToolCalls    []ToolCall
	ToolResults  []ToolResult
	Usage        Usage
	FinishReason FinishReason
	FinishRaw    string
	Warnings     []Warning
	Provider     ProviderData
	Header       http.Header
}

type StepState struct {
	Number int
	Steps  []Step
}

type StepRequest struct {
	Model           LanguageModel
	Instructions    []Text
	Messages        []Message
	Tools           []Tool
	ToolChoice      ToolChoice
	MaxOutputTokens int
	Temperature     *float64
	TopP            *float64
	Reasoning       ReasoningEffort
	Headers         http.Header
	Provider        ProviderData
}

type TextRequest struct {
	Instructions       []Text
	Messages           []Message
	Tools              []Tool
	ToolChoice         ToolChoice
	MaxToolConcurrency int
	Output             Output
	MaxSteps           int
	StopWhen           []StopCondition
	PrepareStep        func(ctx context.Context, s StepState, step *StepRequest) error
	OnStepEnd          func(ctx context.Context, s Step) error
	MaxOutputTokens    int
	Temperature        *float64
	TopP               *float64
	TopK               int
	PresencePenalty    *float64
	FrequencyPenalty   *float64
	Seed               *int
	StopSequences      []string
	Reasoning          ReasoningEffort
	MaxRetries         *int
	Timeouts           Timeouts
	Headers            http.Header
	Provider           ProviderData
	Telemetry          Telemetry
}

func (r TextRequest) Clone() TextRequest {
	r.Instructions = slices.Clone(r.Instructions)
	r.Messages = cloneMessages(r.Messages)
	r.Tools = slices.Clone(r.Tools)
	r.StopWhen = slices.Clone(r.StopWhen)
	r.StopSequences = slices.Clone(r.StopSequences)
	r.Temperature = clonePtr(r.Temperature)
	r.TopP = clonePtr(r.TopP)
	r.PresencePenalty = clonePtr(r.PresencePenalty)
	r.FrequencyPenalty = clonePtr(r.FrequencyPenalty)
	r.Seed = clonePtr(r.Seed)
	r.MaxRetries = clonePtr(r.MaxRetries)
	r.Headers = r.Headers.Clone()
	r.Provider = r.Provider.clone()
	return r
}

func (r TextRequest) WithMessages(msgs []Message) TextRequest {
	c := r.Clone()
	c.Messages = cloneMessages(msgs)
	return c
}

func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

type TextResult struct {
	Text         string
	Content      Parts
	Messages     []Message
	Steps        []Step
	Usage        Usage
	FinishReason FinishReason
	FinishRaw    string
	Warnings     []Warning
	Provider     ProviderData
	output       Output
}

func joinText(parts Parts) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(Text); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func mergeHeaders(dst, defaults http.Header) http.Header {
	if len(defaults) == 0 {
		return dst
	}
	out := dst.Clone()
	if out == nil {
		out = http.Header{}
	}
	for k, v := range defaults {
		if _, ok := out[k]; !ok {
			out[k] = slices.Clone(v)
		}
	}
	return out
}
