package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/schemafit"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	encryptedReasoning = "reasoning.encrypted_content"
)

// LanguageModel keeps raw output items in Provider["openai"]; the next request replays them.
type LanguageModel struct {
	client *transport.Client
	id     string
}

type responsesRequest struct {
	Model                string            `json:"model"`
	Input                []any             `json:"input"`
	Instructions         string            `json:"instructions,omitzero"`
	Tools                []any             `json:"tools,omitzero"`
	ToolChoice           any               `json:"tool_choice,omitzero"`
	MaxOutputTokens      int               `json:"max_output_tokens,omitzero"`
	Temperature          *float64          `json:"temperature,omitzero"`
	TopP                 *float64          `json:"top_p,omitzero"`
	Reasoning            *reasoningConfig  `json:"reasoning,omitzero"`
	Text                 *textConfig       `json:"text,omitzero"`
	Include              []string          `json:"include,omitzero"`
	Store                *bool             `json:"store,omitzero"`
	PreviousResponseID   string            `json:"previous_response_id,omitzero"`
	Conversation         string            `json:"conversation,omitzero"`
	ServiceTier          string            `json:"service_tier,omitzero"`
	PromptCacheKey       string            `json:"prompt_cache_key,omitzero"`
	PromptCacheRetention string            `json:"prompt_cache_retention,omitzero"`
	ParallelToolCalls    *bool             `json:"parallel_tool_calls,omitzero"`
	SafetyIdentifier     string            `json:"safety_identifier,omitzero"`
	User                 string            `json:"user,omitzero"`
	Metadata             map[string]string `json:"metadata,omitzero"`
	Truncation           string            `json:"truncation,omitzero"`
	MaxToolCalls         int               `json:"max_tool_calls,omitzero"`
	TopLogprobs          int               `json:"top_logprobs,omitzero"`
	Stream               bool              `json:"stream,omitzero"`
}

type reasoningConfig struct {
	Effort  string `json:"effort,omitzero"`
	Summary string `json:"summary,omitzero"`
}

type textConfig struct {
	Format    *textFormat `json:"format,omitzero"`
	Verbosity string      `json:"verbosity,omitzero"`
}

type textFormat = openaiwire.SchemaFormat

type functionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict"`
}

type namedToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitzero"`
}

func (m *LanguageModel) Provider() string { return providerName }

func (m *LanguageModel) ModelID() string { return m.id }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	req, warnings, err := m.request(c, false)
	if err != nil {
		return nil, err
	}
	var body responseBody
	header, err := m.client.JSON(ctx, transport.Request{Path: "/responses", Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	return body.response(warnings, header)
}

func (b responseBody) response(warnings []ai.Warning, header http.Header) (*ai.Response, error) {
	if err := b.failure(); err != nil {
		return nil, err
	}
	conv, err := convertOutput(b.Output)
	if err != nil {
		return nil, err
	}
	incomplete := ""
	if b.IncompleteDetails != nil {
		incomplete = b.IncompleteDetails.Reason
	}
	finish, raw := finishOf(b.Status, incomplete, conv)
	return &ai.Response{
		Content:      conv.parts,
		Usage:        openaiwire.Usage(b.Usage),
		FinishReason: finish,
		FinishRaw:    raw,
		Warnings:     warnings,
		Provider:     b.metadata(),
		Header:       header,
	}, nil
}

func (m *LanguageModel) request(c *ai.Call, stream bool) (*responsesRequest, []ai.Warning, error) {
	opts, err := optionsOf(c.Provider)
	if err != nil {
		return nil, nil, err
	}
	input, warnings, err := buildInput(c.Messages)
	if err != nil {
		return nil, nil, err
	}
	req := &responsesRequest{
		Model:                m.id,
		Input:                input,
		Instructions:         openaiwire.Instructions(c.Instructions),
		MaxOutputTokens:      c.MaxOutputTokens,
		Temperature:          c.Temperature,
		TopP:                 c.TopP,
		Store:                opts.Store,
		PreviousResponseID:   opts.PreviousResponseID,
		Conversation:         opts.Conversation,
		ServiceTier:          opts.ServiceTier,
		PromptCacheKey:       opts.PromptCacheKey,
		PromptCacheRetention: opts.PromptCacheRetention,
		ParallelToolCalls:    opts.ParallelToolCalls,
		SafetyIdentifier:     opts.SafetyIdentifier,
		User:                 opts.User,
		Metadata:             opts.Metadata,
		Truncation:           opts.Truncation,
		MaxToolCalls:         opts.MaxToolCalls,
		TopLogprobs:          opts.TopLogprobs,
		Stream:               stream,
		Include:              include(opts),
	}
	if c.Reasoning != "" || opts.ReasoningSummary != "" {
		req.Reasoning = &reasoningConfig{Effort: effort(c.Reasoning), Summary: opts.ReasoningSummary}
	}
	tools, toolWarnings, err := responsesTools(c.Tools)
	if err != nil {
		return nil, nil, err
	}
	req.Tools = tools
	req.ToolChoice = responsesToolChoice(c.ToolChoice, c.Tools)
	text, textWarnings := textSettings(c.Output, opts.TextVerbosity)
	req.Text = text
	warnings = slices.Concat(warnings, toolWarnings, textWarnings, unsupported(c))
	return req, warnings, nil
}

func include(opts Options) []string {
	out := slices.Clone(opts.Include)
	if opts.Store != nil && !*opts.Store && !slices.Contains(out, encryptedReasoning) {
		out = append(out, encryptedReasoning)
	}
	return out
}

func unsupported(c *ai.Call) []ai.Warning {
	var out []ai.Warning
	add := func(name string) {
		out = append(out, ai.Warning{Message: fmt.Sprintf("openai: %s is not supported by the Responses API and was ignored", name)})
	}
	if c.TopK != 0 {
		add("topK")
	}
	if c.PresencePenalty != nil {
		add("presencePenalty")
	}
	if c.FrequencyPenalty != nil {
		add("frequencyPenalty")
	}
	if c.Seed != nil {
		add("seed")
	}
	if len(c.StopSequences) > 0 {
		add("stopSequences")
	}
	return out
}

func textSettings(spec *ai.OutputSpec, verbosity string) (*textConfig, []ai.Warning) {
	if spec == nil && verbosity == "" {
		return nil, nil
	}
	cfg := &textConfig{Verbosity: verbosity}
	if spec == nil {
		return cfg, nil
	}
	format, warnings := openaiwire.JSONSchema(providerName, spec)
	cfg.Format = &format
	return cfg, warnings
}

func responsesTools(tools []ai.Tool) ([]any, []ai.Warning, error) {
	var out []any
	var warnings []ai.Warning
	for _, t := range tools {
		if raw := providerToolConfig(t); raw != nil {
			out = append(out, raw)
			continue
		}
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		strict := false
		if t.Strict {
			fit, err := schemafit.Fit(params, openaiwire.StrictRules(providerName))
			if err != nil {
				return nil, nil, err
			}
			params, strict = fit.Schema, fit.Strict
			warnings = append(warnings, fit.Warnings...)
			if !strict {
				warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("openai: tool %q is not strict because its schema is not strict-compatible", t.Name)})
			}
		}
		out = append(out, functionTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: params, Strict: strict})
	}
	return out, warnings, nil
}

func providerToolConfig(t ai.Tool) json.RawMessage {
	raw, ok := t.Provider[providerName]
	if !ok {
		return nil
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil || probe.Type == "" {
		return nil
	}
	return raw
}

func responsesToolChoice(choice ai.ToolChoice, tools []ai.Tool) any {
	switch choice.Mode() {
	case ai.ToolChoiceRequired:
		return "required"
	case ai.ToolChoiceNone:
		return "none"
	case ai.ToolChoiceNamed:
		for _, t := range tools {
			if t.Name != choice.Name() {
				continue
			}
			if raw := providerToolConfig(t); raw != nil {
				var probe struct {
					Type string `json:"type"`
				}
				if err := json.Unmarshal(raw, &probe); err == nil {
					return namedToolChoice{Type: probe.Type}
				}
			}
		}
		return namedToolChoice{Type: "function", Name: choice.Name()}
	default:
		return nil
	}
}

var _ ai.LanguageModel = (*LanguageModel)(nil)
