package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
	"github.com/minodevss/ai-go/internal/transport"
)

const defaultToolName = "response"

// LanguageModel talks to the Converse and ConverseStream APIs. Raw content blocks are kept in Provider["bedrock"] and are what the next request replays.
type LanguageModel struct {
	client *transport.Client
	id     string
}

type converseRequest struct {
	Messages                          []claudewire.Message `json:"messages"`
	System                            []any                `json:"system,omitzero"`
	InferenceConfig                   *inferenceConfig     `json:"inferenceConfig,omitzero"`
	ToolConfig                        *toolConfig          `json:"toolConfig,omitzero"`
	GuardrailConfig                   *guardrailConfig     `json:"guardrailConfig,omitzero"`
	AdditionalModelRequestFields      json.RawMessage      `json:"additionalModelRequestFields,omitzero"`
	AdditionalModelResponseFieldPaths []string             `json:"additionalModelResponseFieldPaths,omitzero"`
	PerformanceConfig                 *performanceConfig   `json:"performanceConfig,omitzero"`
	ServiceTier                       *serviceTier         `json:"serviceTier,omitzero"`
	RequestMetadata                   map[string]string    `json:"requestMetadata,omitzero"`
	OutputConfig                      *outputConfig        `json:"outputConfig,omitzero"`
}

type inferenceConfig struct {
	MaxTokens     int      `json:"maxTokens,omitzero"`
	StopSequences []string `json:"stopSequences,omitzero"`
	Temperature   *float64 `json:"temperature,omitzero"`
	TopP          *float64 `json:"topP,omitzero"`
}

type guardrailConfig struct {
	Identifier           string `json:"guardrailIdentifier"`
	Version              string `json:"guardrailVersion"`
	Trace                string `json:"trace,omitzero"`
	StreamProcessingMode string `json:"streamProcessingMode,omitzero"`
}

type performanceConfig struct {
	Latency string `json:"latency"`
}

type serviceTier struct {
	Type string `json:"type"`
}

type outputConfig struct {
	TextFormat struct {
		Type      string `json:"type"`
		Structure struct {
			JSONSchema struct {
				Schema      string `json:"schema"`
				Name        string `json:"name,omitzero"`
				Description string `json:"description,omitzero"`
			} `json:"jsonSchema"`
		} `json:"structure"`
	} `json:"textFormat"`
}

type toolConfig struct {
	Tools      []any       `json:"tools"`
	ToolChoice *toolChoice `json:"toolChoice,omitzero"`
}

type toolSpecEntry struct {
	ToolSpec toolSpec `json:"toolSpec"`
}

type toolSpec struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitzero"`
	InputSchema inputSchema `json:"inputSchema"`
	Strict      bool        `json:"strict,omitzero"`
}

type inputSchema struct {
	JSON json.RawMessage `json:"json"`
}

type systemToolEntry struct {
	SystemTool struct {
		Name string `json:"name"`
	} `json:"systemTool"`
}

type toolChoice struct {
	Auto *struct{} `json:"auto,omitzero"`
	Any  *struct{} `json:"any,omitzero"`
	Tool *struct {
		Name string `json:"name"`
	} `json:"tool,omitzero"`
}

type prepared struct {
	req      *converseRequest
	warnings []ai.Warning
	header   http.Header
	forced   string
}

func (m *LanguageModel) Provider() string { return providerName }

func (m *LanguageModel) ModelID() string { return m.id }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	p, err := m.prepare(c, false)
	if err != nil {
		return nil, err
	}
	var body converseResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: modelPath(m.id, "converse"), Body: p.req, Header: p.header}, &body)
	if err != nil {
		return nil, err
	}
	conv, err := convertContent(body.Output.Message.Content, m.id, p.forced)
	if err != nil {
		return nil, err
	}
	finish, raw := finishOf(body.StopReason, slices.ContainsFunc(conv, isClientCall))
	return &ai.Response{
		Content:      conv,
		Usage:        usageOf(body.Usage),
		FinishReason: finish,
		FinishRaw:    raw,
		Warnings:     p.warnings,
		Provider:     body.metadata(),
		Header:       header,
	}, nil
}

func (m *LanguageModel) prepare(c *ai.Call, stream bool) (*prepared, error) {
	opts, err := optionsOf(c.Provider)
	if err != nil {
		return nil, err
	}
	in, err := buildInput(m.id, c.Messages)
	if err != nil {
		return nil, err
	}
	req := &converseRequest{
		Messages:                          in.messages,
		System:                            systemBlocks(c.Instructions),
		AdditionalModelRequestFields:      opts.AdditionalModelRequestFields,
		AdditionalModelResponseFieldPaths: slices.Clone(opts.AdditionalModelResponseFieldPaths),
		RequestMetadata:                   opts.RequestMetadata,
	}
	if inf := inferenceOf(c); inf != nil {
		req.InferenceConfig = inf
	}
	if opts.Guardrail != nil {
		g := opts.Guardrail
		req.GuardrailConfig = &guardrailConfig{Identifier: g.Identifier, Version: g.Version, Trace: g.Trace}
		if stream {
			req.GuardrailConfig.StreamProcessingMode = g.StreamProcessingMode
		}
	}
	if opts.PerformanceLatency != "" {
		req.PerformanceConfig = &performanceConfig{Latency: opts.PerformanceLatency}
	}
	if opts.ServiceTier != "" {
		req.ServiceTier = &serviceTier{Type: opts.ServiceTier}
	}
	warnings := slices.Concat(in.warnings, unsupported(c))

	tools, toolWarnings, err := toolsOf(c.Tools)
	if err != nil {
		return nil, err
	}
	warnings = slices.Concat(warnings, toolWarnings)
	choice, choiceWarning := toolChoiceOf(c.ToolChoice, len(tools) > 0, in.hasTools)
	warnings = append(warnings, choiceWarning...)
	if choice.drop {
		tools = nil
	}

	forced := ""
	if c.Output != nil {
		var out outputResult
		out, err = applyOutput(c.Output, opts)
		if err != nil {
			return nil, err
		}
		warnings = slices.Concat(warnings, out.warnings)
		req.OutputConfig = out.config
		if out.tool != nil {
			tools = append(tools, out.tool)
			forced = out.name
			choice.choice = &toolChoice{Tool: &struct {
				Name string `json:"name"`
			}{Name: forced}}
		}
	}
	if len(tools) > 0 {
		req.ToolConfig = &toolConfig{Tools: tools, ToolChoice: choice.choice}
	}
	header := http.Header{}
	for k, v := range c.Headers {
		header[http.CanonicalHeaderKey(k)] = slices.Clone(v)
	}
	return &prepared{req: req, warnings: warnings, header: header, forced: forced}, nil
}

func isClientCall(p ai.Part) bool {
	c, ok := p.(ai.ToolCall)
	return ok && !c.ProviderExecuted
}

func inferenceOf(c *ai.Call) *inferenceConfig {
	if c.MaxOutputTokens == 0 && len(c.StopSequences) == 0 && c.Temperature == nil && c.TopP == nil {
		return nil
	}
	return &inferenceConfig{MaxTokens: c.MaxOutputTokens, StopSequences: slices.Clone(c.StopSequences), Temperature: c.Temperature, TopP: c.TopP}
}

func systemBlocks(texts []ai.Text) []any {
	var out []any
	for _, t := range texts {
		if t.Text == "" {
			continue
		}
		out = append(out, textBlock{Text: t.Text})
		out = appendCachePoint(out, dataOf(t.Provider).CachePoint)
	}
	return out
}

func unsupported(c *ai.Call) []ai.Warning {
	var out []ai.Warning
	add := func(name, why string) {
		out = append(out, ai.Warning{Message: fmt.Sprintf("bedrock: %s %s and was ignored", name, why)})
	}
	const modelSpecific = "is model specific in Converse: use Options.AdditionalModelRequestFields"
	if c.TopK != 0 {
		add("topK", modelSpecific)
	}
	if c.Reasoning != "" {
		add("reasoning", modelSpecific)
	}
	if c.PresencePenalty != nil {
		add("presencePenalty", "is not supported by Converse")
	}
	if c.FrequencyPenalty != nil {
		add("frequencyPenalty", "is not supported by Converse")
	}
	if c.Seed != nil {
		add("seed", "is not supported by Converse")
	}
	return out
}

func toolsOf(tools []ai.Tool) ([]any, []ai.Warning, error) {
	var out []any
	var warnings []ai.Warning
	for _, t := range tools {
		d := dataOf(t.Provider)
		if d.SystemTool != "" {
			var entry systemToolEntry
			entry.SystemTool.Name = d.SystemTool
			out = append(out, entry)
			out = appendCachePoint(out, d.CachePoint)
			continue
		}
		if !validToolName(t.Name) {
			return nil, nil, fmt.Errorf("bedrock: tool name %q is invalid: names are 1 to 64 letters, digits, underscores or hyphens", t.Name)
		}
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		if t.Strict {
			fit, w, err := fitSchema(params)
			if err != nil {
				return nil, nil, err
			}
			params = json.RawMessage(fit)
			warnings = append(warnings, w...)
		}
		out = append(out, toolSpecEntry{ToolSpec: toolSpec{Name: t.Name, Description: t.Description, InputSchema: inputSchema{JSON: params}, Strict: t.Strict}})
		out = appendCachePoint(out, d.CachePoint)
	}
	return out, warnings, nil
}

func validToolName(name string) bool {
	if name == "" || len(name) > 64 {
		return false
	}
	return !strings.ContainsFunc(name, func(r rune) bool { return !isNameRune(r) })
}

func isNameRune(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-'
}

type choiceResult struct {
	choice *toolChoice
	drop   bool
}

func toolChoiceOf(choice ai.ToolChoice, hasTools, historyHasTools bool) (choiceResult, []ai.Warning) {
	if !hasTools {
		return choiceResult{}, nil
	}
	switch choice.Mode() {
	case ai.ToolChoiceRequired:
		return choiceResult{choice: &toolChoice{Any: &struct{}{}}}, nil
	case ai.ToolChoiceNamed:
		return choiceResult{choice: &toolChoice{Tool: &struct {
			Name string `json:"name"`
		}{Name: choice.Name()}}}, nil
	case ai.ToolChoiceNone:
		if historyHasTools {
			return choiceResult{}, []ai.Warning{{Message: "bedrock: tool choice none is not supported by Converse and tools are required while the history has tool calls, so the model may call a tool"}}
		}
		return choiceResult{drop: true}, []ai.Warning{{Message: "bedrock: tool choice none is not supported by Converse, so tools were left out of the request"}}
	case ai.ToolChoiceAuto:
		return choiceResult{}, nil
	default:
		return choiceResult{}, nil
	}
}

type outputResult struct {
	config   *outputConfig
	tool     any
	name     string
	warnings []ai.Warning
}

func applyOutput(spec *ai.OutputSpec, opts Options) (outputResult, error) {
	if len(spec.Schema) == 0 {
		name := toolName(spec.Name)
		return outputResult{
			tool:     toolSpecEntry{ToolSpec: toolSpec{Name: name, Description: spec.Description, InputSchema: inputSchema{JSON: json.RawMessage(`{"type":"object"}`)}}},
			name:     name,
			warnings: []ai.Warning{{Message: "bedrock: output without a schema uses a forced tool call, so the JSON must be an object"}},
		}, nil
	}
	fit, warnings, err := fitSchema(spec.Schema)
	if err != nil {
		return outputResult{}, err
	}
	if opts.OutputViaTool {
		name := toolName(spec.Name)
		return outputResult{
			tool:     toolSpecEntry{ToolSpec: toolSpec{Name: name, Description: spec.Description, InputSchema: inputSchema{JSON: json.RawMessage(fit)}}},
			name:     name,
			warnings: warnings,
		}, nil
	}
	var cfg outputConfig
	cfg.TextFormat.Type = "json_schema"
	cfg.TextFormat.Structure.JSONSchema.Schema = fit
	cfg.TextFormat.Structure.JSONSchema.Name = toolName(spec.Name)
	cfg.TextFormat.Structure.JSONSchema.Description = spec.Description
	return outputResult{config: &cfg, warnings: warnings}, nil
}

func toolName(name string) string {
	cleaned := strings.Map(func(r rune) rune {
		if isNameRune(r) {
			return r
		}
		return -1
	}, name)
	if cleaned == "" {
		return defaultToolName
	}
	return cleaned[:min(len(cleaned), 64)]
}

var _ ai.LanguageModel = (*LanguageModel)(nil)
