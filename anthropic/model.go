package anthropic

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	defaultMaxTokens = 16000
	mcpBeta          = "mcp-client-2025-11-20"
	defaultToolName  = "response"
)

// LanguageModel talks to the Messages API. Raw content blocks are kept in Provider["anthropic"] and are what the next request replays.
type LanguageModel struct {
	client *transport.Client
	id     string
	betas  []string
}

type wireCache struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitzero"`
}

type systemBlock struct {
	Type         string     `json:"type"`
	Text         string     `json:"text"`
	CacheControl *wireCache `json:"cache_control,omitzero"`
}

type messagesRequest struct {
	Model         string               `json:"model"`
	MaxTokens     int                  `json:"max_tokens"`
	Messages      []claudewire.Message `json:"messages"`
	System        []systemBlock        `json:"system,omitzero"`
	Tools         []any                `json:"tools,omitzero"`
	ToolChoice    any                  `json:"tool_choice,omitzero"`
	Thinking      *thinkingConfig      `json:"thinking,omitzero"`
	OutputConfig  *outputConfig        `json:"output_config,omitzero"`
	CacheControl  *wireCache           `json:"cache_control,omitzero"`
	Container     string               `json:"container,omitzero"`
	MCPServers    []mcpServerWire      `json:"mcp_servers,omitzero"`
	Metadata      *metadata            `json:"metadata,omitzero"`
	ServiceTier   string               `json:"service_tier,omitzero"`
	StopSequences []string             `json:"stop_sequences,omitzero"`
	Stream        bool                 `json:"stream,omitzero"`
}

type thinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitzero"`
	Display      string `json:"display,omitzero"`
}

type outputConfig struct {
	Effort string        `json:"effort,omitzero"`
	Format *outputFormat `json:"format,omitzero"`
}

type outputFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

type metadata struct {
	UserID string `json:"user_id"`
}

type mcpServerWire struct {
	Type               string `json:"type"`
	URL                string `json:"url"`
	Name               string `json:"name"`
	AuthorizationToken string `json:"authorization_token,omitzero"`
}

type toolConfig struct {
	Enabled bool `json:"enabled"`
}

type mcpToolset struct {
	Type          string                `json:"type"`
	ServerName    string                `json:"mcp_server_name"`
	DefaultConfig *toolConfig           `json:"default_config,omitzero"`
	Configs       map[string]toolConfig `json:"configs,omitzero"`
}

type customTool struct {
	Type        string          `json:"type,omitzero"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      bool            `json:"strict,omitzero"`

	CacheControl *wireCache `json:"cache_control,omitzero"`
}

type toolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitzero"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitzero"`
}

type prepared struct {
	req      *messagesRequest
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
	var body messageBody
	header, err := m.client.JSON(ctx, transport.Request{Path: "/v1/messages", Body: p.req, Header: p.header}, &body)
	if err != nil {
		return nil, err
	}
	conv, err := convertContent(body.Content, p.forced)
	if err != nil {
		return nil, err
	}
	finish, raw := finishOf(body.StopReason)
	if finish == ai.FinishToolCalls && !slices.ContainsFunc(conv, isClientCall) {
		finish = ai.FinishStop
	}
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
	in, err := buildInput(c.Messages)
	if err != nil {
		return nil, err
	}
	req := &messagesRequest{
		Model:         m.id,
		MaxTokens:     cmp.Or(c.MaxOutputTokens, defaultMaxTokens),
		Messages:      in.messages,
		System:        systemBlocks(c.Instructions),
		StopSequences: slices.Clone(c.StopSequences),
		Stream:        stream,
		ServiceTier:   opts.ServiceTier,
		CacheControl:  wireCacheOf(opts.CacheControl),
	}
	if opts.UserID != "" {
		req.Metadata = &metadata{UserID: opts.UserID}
	}
	req.Thinking, req.OutputConfig = thinkingOf(c.Reasoning, opts)
	warnings := slices.Concat(in.warnings, unsupported(c))

	tools, toolWarnings, err := toolsOf(c.Tools)
	if err != nil {
		return nil, err
	}
	req.Tools = tools
	req.Container = opts.Container
	if req.Container == "" && hasHostedTool(c.Tools) {
		req.Container = containerOf(c.Messages, time.Now())
	}
	servers, toolsets := mcpOf(opts.MCPServers)
	req.MCPServers = servers
	for _, ts := range toolsets {
		req.Tools = append(req.Tools, ts)
	}
	warnings = slices.Concat(warnings, toolWarnings)

	choice := toolChoiceOf(c.ToolChoice, opts.DisableParallelToolUse, len(req.Tools) > 0)
	forced := ""
	if c.Output != nil {
		var outWarnings []ai.Warning
		forced, outWarnings, err = m.applyOutput(req, c.Output, opts, in.citations)
		if err != nil {
			return nil, err
		}
		warnings = slices.Concat(warnings, outWarnings)
		if forced != "" {
			choice = toolChoice{Type: "tool", Name: forced, DisableParallelToolUse: opts.DisableParallelToolUse}
		}
	}
	if choice.Type != "" {
		req.ToolChoice = choice
	}
	cacheWarnings, err := cacheValidation(req)
	if err != nil {
		return nil, err
	}
	warnings = slices.Concat(warnings, cacheWarnings)
	header := http.Header{}
	if betas := m.betaNames(opts, c.Headers); len(betas) > 0 {
		header.Set(betaHeader, strings.Join(betas, ","))
	}
	for k, v := range c.Headers {
		if http.CanonicalHeaderKey(k) != http.CanonicalHeaderKey(betaHeader) {
			header[http.CanonicalHeaderKey(k)] = slices.Clone(v)
		}
	}
	return &prepared{req: req, warnings: warnings, header: header, forced: forced}, nil
}

func isClientCall(p ai.Part) bool {
	c, ok := p.(ai.ToolCall)
	return ok && !c.ProviderExecuted
}

func (m *LanguageModel) betaNames(opts Options, h http.Header) []string {
	names := slices.Concat(m.betas, opts.Betas, h.Values(betaHeader))
	if len(opts.MCPServers) > 0 {
		names = append(names, mcpBeta)
	}
	return splitBetas(names)
}

func (m *LanguageModel) applyOutput(req *messagesRequest, spec *ai.OutputSpec, opts Options, citations bool) (string, []ai.Warning, error) {
	if len(spec.Schema) == 0 {
		name := toolName(spec.Name)
		req.Tools = append(req.Tools, customTool{Name: name, Description: spec.Description, InputSchema: json.RawMessage(`{"type":"object"}`)})
		return name, []ai.Warning{{Message: "anthropic: output without a schema uses a forced tool call, so the JSON must be an object"}}, nil
	}
	fit, warnings, err := fitSchema(spec.Schema)
	if err != nil {
		return "", nil, err
	}
	if opts.OutputViaTool || citations {
		if citations && !opts.OutputViaTool {
			warnings = append(warnings, ai.Warning{Message: "anthropic: citations cannot be combined with output_config.format, so structured output uses a forced tool call"})
		}
		name := toolName(spec.Name)
		req.Tools = append(req.Tools, customTool{Name: name, Description: spec.Description, InputSchema: fit})
		return name, warnings, nil
	}
	if req.OutputConfig == nil {
		req.OutputConfig = &outputConfig{}
	}
	req.OutputConfig.Format = &outputFormat{Type: "json_schema", Schema: fit}
	return "", warnings, nil
}

func systemBlocks(texts []ai.Text) []systemBlock {
	var out []systemBlock
	for _, t := range texts {
		if t.Text == "" {
			continue
		}
		out = append(out, systemBlock{Type: "text", Text: t.Text, CacheControl: wireCacheOf(dataOf(t.Provider).CacheControl)})
	}
	return out
}

func wireCacheOf(c *CacheControl) *wireCache {
	if c == nil {
		return nil
	}
	return &wireCache{Type: "ephemeral", TTL: string(c.TTL)}
}

func thinkingOf(r ai.ReasoningEffort, opts Options) (*thinkingConfig, *outputConfig) {
	var thinking *thinkingConfig
	var cfg *outputConfig
	switch {
	case opts.Thinking != nil:
		thinking = &thinkingConfig{Type: opts.Thinking.Type, BudgetTokens: opts.Thinking.BudgetTokens, Display: opts.Thinking.Display}
	case r == ai.ReasoningNone:
		thinking = &thinkingConfig{Type: ThinkingDisabled}
	case r != "":
		thinking = &thinkingConfig{Type: ThinkingAdaptive, Display: DisplaySummarized}
	}
	if effort := cmp.Or(opts.Effort, effortOf(r)); effort != "" {
		cfg = &outputConfig{Effort: effort}
	}
	return thinking, cfg
}

func effortOf(r ai.ReasoningEffort) string {
	switch r {
	case ai.ReasoningMinimal:
		return "low"
	case ai.ReasoningLow, ai.ReasoningMedium, ai.ReasoningHigh, ai.ReasoningMax:
		return string(r)
	case ai.ReasoningNone:
		return ""
	default:
		return ""
	}
}

func unsupported(c *ai.Call) []ai.Warning {
	var out []ai.Warning
	add := func(name, why string) {
		out = append(out, ai.Warning{Message: fmt.Sprintf("anthropic: %s %s and was ignored", name, why)})
	}
	const removed = "is removed in API version " + apiVersion
	if c.Temperature != nil {
		add("temperature", removed)
	}
	if c.TopP != nil {
		add("topP", removed)
	}
	if c.TopK != 0 {
		add("topK", removed)
	}
	if c.PresencePenalty != nil {
		add("presencePenalty", "is not supported by the Messages API")
	}
	if c.FrequencyPenalty != nil {
		add("frequencyPenalty", "is not supported by the Messages API")
	}
	if c.Seed != nil {
		add("seed", "is not supported by the Messages API")
	}
	return out
}

func toolsOf(tools []ai.Tool) ([]any, []ai.Warning, error) {
	var out []any
	var warnings []ai.Warning
	for _, t := range tools {
		if raw := hostedConfig(t); raw != nil {
			out = append(out, cachedBlock(raw, dataOf(t.Provider).CacheControl))
			continue
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
			params = fit
			warnings = append(warnings, w...)
		}
		out = append(out, customTool{Name: t.Name, Description: t.Description, InputSchema: params, Strict: t.Strict, CacheControl: wireCacheOf(dataOf(t.Provider).CacheControl)})
	}
	return out, warnings, nil
}

func hostedConfig(t ai.Tool) json.RawMessage {
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

func hasHostedTool(tools []ai.Tool) bool {
	return slices.ContainsFunc(tools, func(t ai.Tool) bool { return hostedConfig(t) != nil })
}

func toolChoiceOf(choice ai.ToolChoice, disableParallel, hasTools bool) toolChoice {
	if !hasTools {
		return toolChoice{}
	}
	switch choice.Mode() {
	case ai.ToolChoiceRequired:
		return toolChoice{Type: "any", DisableParallelToolUse: disableParallel}
	case ai.ToolChoiceNone:
		return toolChoice{Type: "none"}
	case ai.ToolChoiceNamed:
		return toolChoice{Type: "tool", Name: choice.Name(), DisableParallelToolUse: disableParallel}
	case ai.ToolChoiceAuto:
		if disableParallel {
			return toolChoice{Type: "auto", DisableParallelToolUse: true}
		}
		return toolChoice{}
	default:
		return toolChoice{}
	}
}

func mcpOf(servers []MCPServer) ([]mcpServerWire, []mcpToolset) {
	var wires []mcpServerWire
	var sets []mcpToolset
	for _, s := range servers {
		wires = append(wires, mcpServerWire{Type: "url", URL: s.URL, Name: s.Name, AuthorizationToken: s.AuthorizationToken})
		set := mcpToolset{Type: "mcp_toolset", ServerName: s.Name}
		if len(s.AllowedTools) > 0 {
			set.DefaultConfig = &toolConfig{Enabled: false}
			set.Configs = map[string]toolConfig{}
			for _, name := range s.AllowedTools {
				set.Configs[name] = toolConfig{Enabled: true}
			}
		} else if len(s.DisabledTools) > 0 {
			set.Configs = map[string]toolConfig{}
			for _, name := range s.DisabledTools {
				set.Configs[name] = toolConfig{Enabled: false}
			}
		}
		sets = append(sets, set)
	}
	return wires, sets
}

func toolName(name string) string {
	cleaned := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' {
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
