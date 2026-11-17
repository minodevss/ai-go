package chatcompletions

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/schemafit"
	"github.com/minodevss/ai-go/internal/transport"
)

// Request is the Chat Completions request body. Extra wins over the typed fields on a name clash.
type Request struct {
	Model               string         `json:"model"`
	Messages            []any          `json:"messages"`
	Tools               []any          `json:"tools,omitzero"`
	ToolChoice          any            `json:"tool_choice,omitzero"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitzero"`
	MaxTokens           int            `json:"max_tokens,omitzero"`
	Temperature         *float64       `json:"temperature,omitzero"`
	TopP                *float64       `json:"top_p,omitzero"`
	PresencePenalty     *float64       `json:"presence_penalty,omitzero"`
	FrequencyPenalty    *float64       `json:"frequency_penalty,omitzero"`
	Seed                *int           `json:"seed,omitzero"`
	Stop                []string       `json:"stop,omitzero"`
	ReasoningEffort     string         `json:"reasoning_effort,omitzero"`
	ResponseFormat      any            `json:"response_format,omitzero"`
	ParallelToolCalls   *bool          `json:"parallel_tool_calls,omitzero"`
	Stream              bool           `json:"stream,omitzero"`
	StreamOptions       *StreamOptions `json:"stream_options,omitzero"`
	Extra               map[string]any `json:"-"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type responseFormat struct {
	Type       string `json:"type"`
	JSONSchema struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitzero"`
		Schema      json.RawMessage `json:"schema"`
		Strict      bool            `json:"strict"`
	} `json:"json_schema"`
}

type tool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitzero"`
		Parameters  json.RawMessage `json:"parameters"`
		Strict      *bool           `json:"strict,omitzero"`
	} `json:"function"`
}

type namedChoice struct {
	Type     string `json:"type"`
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

func (r Request) MarshalJSON() ([]byte, error) {
	type plain Request
	typed, err := transport.EncodeJSON(plain(r))
	if err != nil {
		return nil, err
	}
	return mergeExtra(typed, r.Extra)
}

func mergeExtra[V any](typed []byte, extra map[string]V) ([]byte, error) {
	if len(extra) == 0 {
		return typed, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(typed, &fields); err != nil {
		return nil, err
	}
	for k, v := range extra {
		raw, err := transport.EncodeJSON(v)
		if err != nil {
			return nil, fmt.Errorf("chatcompletions: encode %s: %w", k, err)
		}
		fields[k] = raw
	}
	return transport.EncodeJSON(fields)
}

func (m *Model) request(c *ai.Call, stream bool) (*Request, []ai.Warning, error) {
	messages, err := m.messages(c)
	if err != nil {
		return nil, nil, err
	}
	req := &Request{
		Model:             m.id,
		Messages:          messages,
		Temperature:       c.Temperature,
		TopP:              c.TopP,
		PresencePenalty:   c.PresencePenalty,
		FrequencyPenalty:  c.FrequencyPenalty,
		Seed:              c.Seed,
		Stop:              c.StopSequences,
		ReasoningEffort:   string(c.Reasoning),
		Stream:            stream,
		MaxTokens:         0,
		ParallelToolCalls: nil,
	}
	switch m.dialect.MaxTokens {
	case MaxTokens:
		req.MaxTokens = c.MaxOutputTokens
	default:
		req.MaxCompletionTokens = c.MaxOutputTokens
	}
	if stream && !m.dialect.NoStreamOptions {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	var warnings []ai.Warning
	req.Tools, req.ToolChoice, warnings = m.tools(c)
	if c.Output != nil {
		format, w := m.format(c.Output)
		req.ResponseFormat = format
		warnings = append(warnings, w...)
	}
	if c.TopK != 0 && m.dialect.TopK {
		req.Extra = map[string]any{"top_k": c.TopK}
	}
	if c.TopK != 0 && !m.dialect.TopK {
		warnings = append(warnings, ai.Warning{Message: m.dialect.Provider + ": topK is not supported by the Chat Completions API and was ignored"})
	}
	if m.dialect.Request != nil {
		w, err := m.dialect.Request(c, req)
		if err != nil {
			return nil, nil, err
		}
		warnings = append(warnings, w...)
	}
	return req, warnings, nil
}

func (m *Model) format(spec *ai.OutputSpec) (responseFormat, []ai.Warning) {
	format, warnings := openaiwire.JSONSchema(m.dialect.Provider, spec)
	out := responseFormat{Type: "json_schema"}
	out.JSONSchema.Name = format.Name
	out.JSONSchema.Description = format.Description
	out.JSONSchema.Schema = format.Schema
	out.JSONSchema.Strict = format.Strict != nil && *format.Strict
	return out, warnings
}

func (m *Model) tools(c *ai.Call) ([]any, any, []ai.Warning) {
	var tools []any
	var warnings []ai.Warning
	for _, t := range c.Tools {
		if hosted(m.dialect.Provider, t) {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("%s: provider tool %q is not supported by the Chat Completions API and was ignored", m.dialect.Provider, t.Name)})
			continue
		}
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		strict := false
		if t.Strict {
			fit, err := schemafit.Fit(params, openaiwire.StrictRules(m.dialect.Provider))
			if err != nil {
				warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("%s: tool %q cannot use strict mode and was sent without it: %v", m.dialect.Provider, t.Name, err)})
			} else {
				params, strict = fit.Schema, fit.Strict
				warnings = append(warnings, fit.Warnings...)
			}
		}
		var out tool
		out.Type = "function"
		out.Function.Name = t.Name
		out.Function.Description = t.Description
		out.Function.Parameters = params
		if m.dialect.ToolStrict {
			out.Function.Strict = &strict
		}
		tools = append(tools, out)
	}
	switch c.ToolChoice.Mode() {
	case ai.ToolChoiceRequired:
		return tools, "required", warnings
	case ai.ToolChoiceNone:
		return tools, "none", warnings
	case ai.ToolChoiceNamed:
		choice := namedChoice{Type: "function"}
		choice.Function.Name = c.ToolChoice.Name()
		return tools, choice, warnings
	default:
		return tools, nil, warnings
	}
}

func hosted(provider string, t ai.Tool) bool {
	raw, ok := t.Provider[provider]
	if !ok {
		return false
	}
	var probe struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &probe) == nil && probe.Type != ""
}
