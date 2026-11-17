package ai

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"slices"
)

// LanguageModel implementations must not mutate the Call and must return provider-executed output in Content as ToolCall and ToolResult parts.
type LanguageModel interface {
	Provider() string
	ModelID() string
	Generate(ctx context.Context, c *Call) (*Response, error)
	Stream(ctx context.Context, c *Call) (*StreamResponse, error)
}

type Call struct {
	Instructions     []Text
	Messages         []Message
	Tools            []Tool
	ToolChoice       ToolChoice
	Output           *OutputSpec
	MaxOutputTokens  int
	Temperature      *float64
	TopP             *float64
	TopK             int
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Seed             *int
	StopSequences    []string
	Reasoning        ReasoningEffort
	Headers          http.Header
	Provider         ProviderData
}

// OutputSpec asks the model for JSON that matches Schema. The model answers with that JSON as Text content; adapters without a native schema mode force a tool call and return its input as Text.
type OutputSpec struct {
	Name        string
	Description string
	Schema      json.RawMessage
}

// Response.FinishReason may be FinishPause when the provider paused its turn; the core then continues with the response appended.
type Response struct {
	Content      Parts
	Usage        Usage
	FinishReason FinishReason
	FinishRaw    string
	Warnings     []Warning
	Provider     ProviderData
	Header       http.Header
}

// StreamResponse.Parts must end with a StepEnd part; an error in the sequence ends it.
type StreamResponse struct {
	Parts  iter.Seq2[StreamPart, error]
	Header http.Header
}

type Limiter interface {
	Wait(ctx context.Context) error
}

type Provider interface {
	LanguageModel(id string) (LanguageModel, error)
}

func (c *Call) Clone() *Call {
	out := *c
	out.Instructions = slices.Clone(c.Instructions)
	out.Messages = cloneMessages(c.Messages)
	out.Tools = slices.Clone(c.Tools)
	out.Temperature = clonePtr(c.Temperature)
	out.TopP = clonePtr(c.TopP)
	out.PresencePenalty = clonePtr(c.PresencePenalty)
	out.FrequencyPenalty = clonePtr(c.FrequencyPenalty)
	out.Seed = clonePtr(c.Seed)
	out.StopSequences = slices.Clone(c.StopSequences)
	out.Headers = c.Headers.Clone()
	out.Provider = c.Provider.clone()
	if c.Output != nil {
		spec := *c.Output
		spec.Schema = slices.Clone(c.Output.Schema)
		out.Output = &spec
	}
	return &out
}
