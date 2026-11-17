package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
	"github.com/minodevss/ai-go/internal/transport"
)

const chatPath = "/v2/chat"

// LanguageModel talks to the Chat v2 API. Citation offsets are positions in the generated text, and DocIndex points into the request's documents.
type LanguageModel struct {
	client *transport.Client
	id     string
}

type chatRequest struct {
	Model           string           `json:"model"`
	Messages        []wireMessage    `json:"messages"`
	Tools           []wireTool       `json:"tools,omitzero"`
	StrictTools     bool             `json:"strict_tools,omitzero"`
	ToolChoice      string           `json:"tool_choice,omitzero"`
	Documents       []wireDocument   `json:"documents,omitzero"`
	CitationOptions *citationOptions `json:"citation_options,omitzero"`
	ResponseFormat  *responseFormat  `json:"response_format,omitzero"`
	SafetyMode      string           `json:"safety_mode,omitzero"`
	MaxTokens       int              `json:"max_tokens,omitzero"`
	StopSequences   []string         `json:"stop_sequences,omitzero"`
	Temperature     *float64         `json:"temperature,omitzero"`
	Seed            *int             `json:"seed,omitzero"`
	FrequencyPen    *float64         `json:"frequency_penalty,omitzero"`
	PresencePen     *float64         `json:"presence_penalty,omitzero"`
	K               int              `json:"k,omitzero"`
	P               *float64         `json:"p,omitzero"`
	Thinking        *wireThinking    `json:"thinking,omitzero"`
	Priority        int              `json:"priority,omitzero"`
	Stream          bool             `json:"stream,omitzero"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters"`
}

type citationOptions struct {
	Mode string `json:"mode"`
}

type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitzero"`
}

type wireThinking struct {
	Type        string `json:"type"`
	TokenBudget int    `json:"token_budget,omitzero"`
}

type prepared struct {
	req      *chatRequest
	warnings []ai.Warning
	header   http.Header
	docIDs   []string
}

func schemaRules() schemafit.Rules {
	return schemafit.Rules{
		Provider: providerName,
		Unsupported: []string{
			"$schema", "allOf", "not", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
			"minItems", "maxItems", "minLength", "maxLength", "uniqueItems", "multipleOf",
		},
		Renamed: map[string]string{"oneOf": "anyOf"},
		Formats: []string{"date-time", "uuid", "date", "time"},
	}
}

func (m *LanguageModel) Provider() string { return providerName }

func (m *LanguageModel) ModelID() string { return m.id }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	p, err := m.prepare(c, false)
	if err != nil {
		return nil, err
	}
	var body chatResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: chatPath, Body: p.req, Header: p.header}, &body)
	if err != nil {
		return nil, err
	}
	parts, data, err := m.convert(body, p.docIDs)
	if err != nil {
		return nil, err
	}
	return &ai.Response{
		Content:      parts,
		Usage:        usageOf(body.Usage),
		FinishReason: finishOf(body.FinishReason),
		FinishRaw:    body.FinishReason,
		Warnings:     p.warnings,
		Provider:     responseProvider(data),
		Header:       header,
	}, nil
}

func (m *LanguageModel) prepare(c *ai.Call, stream bool) (*prepared, error) {
	opts, err := optionsOf[Options](c.Provider)
	if err != nil {
		return nil, err
	}
	in, err := buildInput(c.Instructions, c.Messages)
	if err != nil {
		return nil, err
	}
	docs := in.documents
	for _, d := range opts.Documents {
		if d.ID == "" {
			d.ID = strconv.Itoa(len(docs))
		}
		docs = append(docs, wireDocument{ID: d.ID, Data: d.Data})
	}
	req := &chatRequest{
		Model:         m.id,
		Messages:      in.messages,
		Documents:     docs,
		SafetyMode:    opts.SafetyMode,
		MaxTokens:     c.MaxOutputTokens,
		StopSequences: slices.Clone(c.StopSequences),
		Temperature:   c.Temperature,
		Seed:          c.Seed,
		FrequencyPen:  c.FrequencyPenalty,
		PresencePen:   c.PresencePenalty,
		K:             c.TopK,
		P:             c.TopP,
		Priority:      opts.Priority,
		StrictTools:   opts.StrictTools,
		Stream:        stream,
		Thinking:      thinkingOf(c.Reasoning, opts),
	}
	if opts.CitationMode != "" {
		req.CitationOptions = &citationOptions{Mode: opts.CitationMode}
	}
	warnings := in.warnings
	tools, choice, toolWarnings := toolsOf(c)
	req.Tools, req.ToolChoice = tools, choice
	warnings = slices.Concat(warnings, toolWarnings)
	if !req.StrictTools {
		req.StrictTools = slices.ContainsFunc(c.Tools, func(t ai.Tool) bool { return t.Strict })
	}
	if c.Output != nil {
		format, outWarnings, err := outputFormat(c.Output, len(req.Tools) > 0 || len(docs) > 0)
		if err != nil {
			return nil, err
		}
		req.ResponseFormat = format
		warnings = slices.Concat(warnings, outWarnings)
	}
	ids := make([]string, len(docs))
	for i, d := range docs {
		ids[i] = d.ID
	}
	return &prepared{req: req, warnings: warnings, header: c.Headers.Clone(), docIDs: ids}, nil
}

func thinkingOf(r ai.ReasoningEffort, opts Options) *wireThinking {
	switch {
	case opts.Thinking != nil:
		return &wireThinking{Type: opts.Thinking.Type, TokenBudget: opts.Thinking.TokenBudget}
	case r == ai.ReasoningNone:
		return &wireThinking{Type: ThinkingDisabled}
	case r != "":
		return &wireThinking{Type: ThinkingEnabled}
	default:
		return nil
	}
}

func toolsOf(c *ai.Call) ([]wireTool, string, []ai.Warning) {
	if len(c.Tools) == 0 {
		return nil, "", nil
	}
	out := make([]wireTool, len(c.Tools))
	for i, t := range c.Tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out[i] = wireTool{Type: "function", Function: wireFunction{Name: t.Name, Description: t.Description, Parameters: params}}
	}
	switch c.ToolChoice.Mode() {
	case ai.ToolChoiceRequired:
		return out, "REQUIRED", nil
	case ai.ToolChoiceNone:
		return out, "NONE", nil
	case ai.ToolChoiceNamed:
		warning := ai.Warning{Message: fmt.Sprintf("cohere: tool choice %q cannot name a tool, so any tool is required", c.ToolChoice.Name())}
		return out, "REQUIRED", []ai.Warning{warning}
	case ai.ToolChoiceAuto:
		return out, "", nil
	default:
		return out, "", nil
	}
}

func outputFormat(spec *ai.OutputSpec, conflicts bool) (*responseFormat, []ai.Warning, error) {
	if conflicts {
		return nil, nil, errors.New("cohere: structured output cannot be combined with tools or documents")
	}
	if len(spec.Schema) == 0 {
		return &responseFormat{Type: "json_object"}, nil, nil
	}
	fit, err := schemafit.Fit(spec.Schema, schemaRules())
	if err != nil {
		return nil, nil, err
	}
	return &responseFormat{Type: "json_object", JSONSchema: fit.Schema}, fit.Warnings, nil
}

var _ ai.LanguageModel = (*LanguageModel)(nil)
