package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
)

var jsonSchemaRules = schemafit.Rules{
	Provider: providerName,
	Unsupported: []string{
		"$schema", "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
		"minLength", "maxLength", "uniqueItems", "contains", "minContains", "maxContains",
		"minProperties", "maxProperties", "propertyNames", "patternProperties",
		"unevaluatedProperties", "unevaluatedItems", "default", "examples",
	},
	Renamed: map[string]string{"oneOf": "anyOf"},
}

var openAPIRules = schemafit.Rules{
	Provider: providerName,
	Unsupported: []string{
		"$schema", "$defs", "definitions", "$ref", "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
		"additionalProperties", "const", "pattern", "uniqueItems", "contains", "minContains", "maxContains",
		"propertyNames", "patternProperties", "unevaluatedProperties", "unevaluatedItems", "prefixItems", "examples",
	},
	Renamed: map[string]string{"oneOf": "anyOf"},
}

type generateRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitzero"`
	Tools             []any             `json:"tools,omitzero"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitzero"`
	SafetySettings    []SafetySetting   `json:"safetySettings,omitzero"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitzero"`
	CachedContent     string            `json:"cachedContent,omitzero"`
	Labels            map[string]string `json:"labels,omitzero"`
	ServiceTier       string            `json:"serviceTier,omitzero"`
}

type generationConfig struct {
	Temperature        *float64        `json:"temperature,omitzero"`
	TopP               *float64        `json:"topP,omitzero"`
	TopK               int             `json:"topK,omitzero"`
	MaxOutputTokens    int             `json:"maxOutputTokens,omitzero"`
	StopSequences      []string        `json:"stopSequences,omitzero"`
	Seed               *int            `json:"seed,omitzero"`
	PresencePenalty    *float64        `json:"presencePenalty,omitzero"`
	FrequencyPenalty   *float64        `json:"frequencyPenalty,omitzero"`
	ResponseMimeType   string          `json:"responseMimeType,omitzero"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitzero"`
	ResponseSchema     json.RawMessage `json:"responseSchema,omitzero"`
	ResponseModalities []string        `json:"responseModalities,omitzero"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitzero"`
}

type thinkingConfig struct {
	IncludeThoughts *bool  `json:"includeThoughts,omitzero"`
	ThinkingBudget  *int   `json:"thinkingBudget,omitzero"`
	ThinkingLevel   string `json:"thinkingLevel,omitzero"`
}

type toolConfig struct {
	FunctionCallingConfig            *functionCallingConfig `json:"functionCallingConfig,omitzero"`
	IncludeServerSideToolInvocations bool                   `json:"includeServerSideToolInvocations,omitzero"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitzero"`
}

type functionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description,omitzero"`
	Parameters           json.RawMessage `json:"parameters,omitzero"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitzero"`
}

func (m *LanguageModel) request(c *ai.Call) (*generateRequest, []ai.Warning, error) {
	opts, err := optionsOf(c.Provider)
	if err != nil {
		return nil, nil, err
	}
	contents, warnings, err := m.backend.buildContents(m.id, c.Messages)
	if err != nil {
		return nil, nil, err
	}
	req := &generateRequest{
		Contents:       contents,
		SafetySettings: opts.SafetySettings,
		CachedContent:  opts.CachedContent,
		Labels:         opts.Labels,
		ServiceTier:    opts.ServiceTier,
	}
	if len(c.Instructions) > 0 {
		sys := &content{}
		for _, t := range c.Instructions {
			sys.Parts = append(sys.Parts, pdata.JSON(providerName, partOut{Text: &t.Text}))
		}
		req.SystemInstruction = sys
	}
	cfg := &generationConfig{
		Temperature:        c.Temperature,
		TopP:               c.TopP,
		TopK:               c.TopK,
		MaxOutputTokens:    c.MaxOutputTokens,
		StopSequences:      c.StopSequences,
		Seed:               c.Seed,
		PresencePenalty:    c.PresencePenalty,
		FrequencyPenalty:   c.FrequencyPenalty,
		ResponseModalities: opts.ResponseModalities,
	}
	thinking, thinkWarnings, err := thinkingFor(m.id, c.Reasoning, opts)
	if err != nil {
		return nil, nil, err
	}
	cfg.ThinkingConfig = thinking
	toolEntries, toolCfg, toolWarnings, err := toolsFor(c, opts)
	if err != nil {
		return nil, nil, err
	}
	req.Tools, req.ToolConfig = toolEntries, toolCfg
	outWarnings, err := applyOutput(cfg, c.Output, opts)
	if err != nil {
		return nil, nil, err
	}
	if !reflect.ValueOf(cfg).Elem().IsZero() {
		req.GenerationConfig = cfg
	}
	return req, slices.Concat(warnings, thinkWarnings, toolWarnings, outWarnings), nil
}

func thinkingFor(model string, effort ai.ReasoningEffort, opts Options) (*thinkingConfig, []ai.Warning, error) {
	if opts.ThinkingBudget != nil && opts.ThinkingLevel != "" {
		return nil, nil, errors.New("google: thinkingBudget and thinkingLevel are mutually exclusive")
	}
	cfg := &thinkingConfig{ThinkingBudget: opts.ThinkingBudget, ThinkingLevel: opts.ThinkingLevel}
	var warnings []ai.Warning
	if cfg.ThinkingBudget == nil && cfg.ThinkingLevel == "" && effort != "" {
		if strings.HasPrefix(model, "gemini-2") {
			budget := budgetFor(effort)
			if budget < proMinBudget && strings.Contains(model, "2.5-pro") {
				budget = proMinBudget
				warnings = append(warnings, ai.Warning{Message: "google: thinking cannot be turned off on " + model + "; used the minimum budget of 128"})
			}
			cfg.ThinkingBudget = &budget
		} else {
			level, warning := levelFor(model, effort)
			cfg.ThinkingLevel = level
			if warning != "" {
				warnings = append(warnings, ai.Warning{Message: "google: " + warning})
			}
		}
	}
	if cfg.ThinkingBudget == nil && cfg.ThinkingLevel == "" && opts.IncludeThoughts == nil {
		return nil, nil, nil
	}
	cfg.IncludeThoughts = opts.IncludeThoughts
	if cfg.IncludeThoughts == nil && (cfg.ThinkingBudget == nil || *cfg.ThinkingBudget != 0) {
		cfg.IncludeThoughts = new(true)
	}
	return cfg, warnings, nil
}

const proMinBudget = 128

func budgetFor(effort ai.ReasoningEffort) int {
	switch effort {
	case ai.ReasoningNone:
		return 0
	case ai.ReasoningMinimal:
		return 512
	case ai.ReasoningLow:
		return 1024
	case ai.ReasoningMedium:
		return 8192
	default:
		return 24576
	}
}

func levelFor(model string, effort ai.ReasoningEffort) (string, string) {
	noMinimal := strings.HasPrefix(model, "gemini-4") || strings.HasPrefix(model, "gemini-3.8")
	switch effort {
	case ai.ReasoningNone:
		if noMinimal {
			return "low", "thinking cannot be turned off on " + model + "; used the lowest level"
		}
		return "minimal", "thinking cannot be fully turned off on " + model + "; used the minimal level"
	case ai.ReasoningMinimal:
		if noMinimal {
			return "low", model + " has no minimal thinking level; used low"
		}
		return "minimal", ""
	case ai.ReasoningLow:
		return "low", ""
	case ai.ReasoningMedium:
		return "medium", ""
	case ai.ReasoningMax:
		if strings.Contains(model, "hydrogen") {
			return "xhigh", ""
		}
		return "high", "xhigh is only offered by gemini-4-hydrogen; used high"
	default:
		return "high", ""
	}
}

func toolsFor(c *ai.Call, opts Options) ([]any, *toolConfig, []ai.Warning, error) {
	var entries []any
	var decls []functionDeclaration
	var warnings []ai.Warning
	strict := false
	for _, t := range c.Tools {
		if hosted, ok := hostedToolOf(t); ok {
			entries = append(entries, hosted)
			continue
		}
		decl, w, err := declaration(t, opts.OpenAPISchemas)
		if err != nil {
			return nil, nil, nil, err
		}
		decls = append(decls, decl)
		warnings = append(warnings, w...)
		strict = strict || t.Strict
	}
	if len(decls) > 0 {
		entries = append(entries, map[string][]functionDeclaration{"functionDeclarations": decls})
	}
	cfg := &toolConfig{IncludeServerSideToolInvocations: opts.ServerSideToolInvocations}
	if mode := callingMode(c, len(decls) > 0, strict, opts.ServerSideToolInvocations); mode != nil {
		cfg.FunctionCallingConfig = mode
	}
	if cfg.FunctionCallingConfig == nil && !cfg.IncludeServerSideToolInvocations {
		cfg = nil
	}
	return entries, cfg, warnings, nil
}

func callingMode(c *ai.Call, hasFunctions, strict, serverSide bool) *functionCallingConfig {
	if !hasFunctions {
		return nil
	}
	switch c.ToolChoice.Mode() {
	case ai.ToolChoiceRequired:
		return &functionCallingConfig{Mode: "ANY"}
	case ai.ToolChoiceNone:
		return &functionCallingConfig{Mode: "NONE"}
	case ai.ToolChoiceNamed:
		return &functionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{c.ToolChoice.Name()}}
	default:
		if strict || serverSide {
			return &functionCallingConfig{Mode: "VALIDATED"}
		}
		return nil
	}
}

func declaration(t ai.Tool, openAPI bool) (functionDeclaration, []ai.Warning, error) {
	decl := functionDeclaration{Name: t.Name, Description: t.Description}
	params := t.InputSchema
	if len(params) == 0 || isEmptyObjectSchema(params) {
		return decl, nil, nil
	}
	if openAPI {
		fit, err := openAPISchema(params)
		if err != nil {
			return decl, nil, err
		}
		decl.Parameters = fit.Schema
		return decl, prefixTool(t.Name, fit.Warnings), nil
	}
	fit, err := schemafit.Fit(params, jsonSchemaRules)
	if err != nil {
		return decl, nil, err
	}
	decl.ParametersJSONSchema = fit.Schema
	return decl, prefixTool(t.Name, fit.Warnings), nil
}

func prefixTool(name string, warnings []ai.Warning) []ai.Warning {
	out := make([]ai.Warning, len(warnings))
	for i, w := range warnings {
		out[i] = ai.Warning{Message: fmt.Sprintf("%s (tool %q)", w.Message, name)}
	}
	return out
}

func isEmptyObjectSchema(raw json.RawMessage) bool {
	var s struct {
		Type       string         `json:"type"`
		Properties map[string]any `json:"properties"`
	}
	return json.Unmarshal(raw, &s) == nil && s.Type == "object" && len(s.Properties) == 0
}

func applyOutput(cfg *generationConfig, spec *ai.OutputSpec, opts Options) ([]ai.Warning, error) {
	if spec == nil {
		return nil, nil
	}
	cfg.ResponseMimeType = "application/json"
	if opts.OpenAPISchemas {
		fit, err := openAPISchema(spec.Schema)
		if err != nil {
			return nil, err
		}
		cfg.ResponseSchema = fit.Schema
		return fit.Warnings, nil
	}
	fit, err := schemafit.Fit(spec.Schema, jsonSchemaRules)
	if err != nil {
		return nil, err
	}
	cfg.ResponseJSONSchema = fit.Schema
	return fit.Warnings, nil
}

func hasGroundingTool(tools []ai.Tool) bool {
	return slices.ContainsFunc(tools, func(t ai.Tool) bool {
		hosted, ok := hostedToolOf(t)
		if !ok {
			return false
		}
		_, search := hosted["googleSearch"]
		_, maps := hosted["googleMaps"]
		_, files := hosted["fileSearch"]
		return search || maps || files
	})
}
