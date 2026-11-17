package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	chatPath     = "/api/chat"
	acceptNDJSON = "application/x-ndjson"
)

// LanguageModel talks to /api/chat. Ollama has no tool call ids, so ids are derived from the call and its position.
type LanguageModel struct {
	client *transport.Client
	id     string
}

type chatRequest struct {
	Model     string          `json:"model"`
	Messages  []wireMessage   `json:"messages"`
	Tools     []wireTool      `json:"tools,omitzero"`
	Format    json.RawMessage `json:"format,omitzero"`
	Options   map[string]any  `json:"options,omitzero"`
	Stream    bool            `json:"stream"`
	Think     any             `json:"think,omitzero"`
	KeepAlive string          `json:"keep_alive,omitzero"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireToolFunc `json:"function"`
}

type wireToolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters"`
}

type prepared struct {
	req      *chatRequest
	warnings []ai.Warning
	header   http.Header
}

func (m *LanguageModel) Provider() string { return providerName }

func (m *LanguageModel) ModelID() string { return m.id }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	p, err := m.prepare(c, false)
	if err != nil {
		return nil, err
	}
	var body chatChunk
	header, err := m.client.JSON(ctx, transport.Request{Path: chatPath, Body: p.req, Header: p.header}, &body)
	if err != nil {
		return nil, err
	}
	if body.Error != "" {
		return nil, streamError(body.Error)
	}
	parts := convertMessage(body.Message)
	finish, raw := finishOf(body.DoneReason, hasToolCall(parts))
	return &ai.Response{
		Content:      parts,
		Usage:        usageOf(body),
		FinishReason: finish,
		FinishRaw:    raw,
		Warnings:     p.warnings,
		Provider:     body.metadata(),
		Header:       header,
	}, nil
}

func (m *LanguageModel) prepare(c *ai.Call, stream bool) (*prepared, error) {
	opts, err := optionsOf[Options](c.Provider)
	if err != nil {
		return nil, err
	}
	msgs, warnings, err := buildInput(c.Instructions, c.Messages)
	if err != nil {
		return nil, err
	}
	req := &chatRequest{
		Model:     m.id,
		Messages:  msgs,
		Stream:    stream,
		KeepAlive: opts.KeepAlive,
		Options:   modelOptions(c, opts),
		Think:     thinkOf(c.Reasoning, opts),
	}
	req.Tools = toolsOf(c.Tools)
	if c.ToolChoice.Mode() != ai.ToolChoiceAuto {
		warnings = append(warnings, ai.Warning{Message: "ollama: tool choice is not supported and was ignored"})
	}
	if c.PresencePenalty != nil {
		warnings = append(warnings, ai.Warning{Message: "ollama: presencePenalty is not a documented model option and was ignored"})
	}
	if c.FrequencyPenalty != nil {
		warnings = append(warnings, ai.Warning{Message: "ollama: frequencyPenalty is not a documented model option and was ignored"})
	}
	if c.Output != nil {
		req.Format = formatOf(c.Output)
	}
	header := c.Headers.Clone()
	if stream {
		if header == nil {
			header = http.Header{}
		}
		if header.Get("Accept") == "" {
			header.Set("Accept", acceptNDJSON)
		}
	}
	return &prepared{req: req, warnings: warnings, header: header}, nil
}

func modelOptions(c *ai.Call, o Options) map[string]any {
	out := map[string]any{}
	maps.Copy(out, o.Parameters)
	set := func(key string, v any) { out[key] = v }
	if c.Temperature != nil {
		set("temperature", *c.Temperature)
	}
	if c.TopP != nil {
		set("top_p", *c.TopP)
	}
	if c.TopK != 0 {
		set("top_k", c.TopK)
	}
	if c.Seed != nil {
		set("seed", *c.Seed)
	}
	if len(c.StopSequences) > 0 {
		set("stop", slices.Clone(c.StopSequences))
	}
	if c.MaxOutputTokens > 0 {
		set("num_predict", c.MaxOutputTokens)
	}
	if o.NumCtx != 0 {
		set("num_ctx", o.NumCtx)
	}
	if o.MinP != nil {
		set("min_p", *o.MinP)
	}
	if o.RepeatPenalty != nil {
		set("repeat_penalty", *o.RepeatPenalty)
	}
	if o.RepeatLastN != nil {
		set("repeat_last_n", *o.RepeatLastN)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func thinkOf(r ai.ReasoningEffort, o Options) any {
	switch {
	case o.Think != "":
		return o.Think
	case r == ai.ReasoningNone:
		return false
	case r != "":
		return true
	default:
		return nil
	}
}

func formatOf(spec *ai.OutputSpec) json.RawMessage {
	if len(spec.Schema) == 0 {
		return json.RawMessage(`"json"`)
	}
	return spec.Schema
}

func toolsOf(tools []ai.Tool) []wireTool {
	var out []wireTool
	for _, t := range tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, wireTool{Type: "function", Function: wireToolFunc{Name: t.Name, Description: t.Description, Parameters: params}})
	}
	return out
}

func callID(name string, args json.RawMessage, index int) string {
	sum := sha256.Sum256(fmt.Appendf(nil, "%s\x00%s\x00%d", name, args, index))
	return "call_" + hex.EncodeToString(sum[:])[:16]
}

func durationOf(ns int64) time.Duration { return time.Duration(ns) }

var _ ai.LanguageModel = (*LanguageModel)(nil)
