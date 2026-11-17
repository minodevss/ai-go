package anthropic

import (
	"cmp"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
	"github.com/minodevss/ai-go/internal/pdata"
)

type messageBody struct {
	ID           string            `json:"id"`
	Content      []json.RawMessage `json:"content"`
	StopReason   string            `json:"stop_reason"`
	StopSequence string            `json:"stop_sequence"`
	StopDetails  json.RawMessage   `json:"stop_details"`
	Usage        json.RawMessage   `json:"usage"`
	Container    *containerBody    `json:"container"`
}

type containerBody struct {
	ID        string `json:"id"`
	ExpiresAt string `json:"expires_at"`
}

type contentBlock struct {
	Type      string            `json:"type"`
	Text      string            `json:"text"`
	Citations []json.RawMessage `json:"citations"`
	Thinking  string            `json:"thinking"`
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	ServerNm  string            `json:"server_name"`
	Input     json.RawMessage   `json:"input"`
	ToolUseID string            `json:"tool_use_id"`
	Content   json.RawMessage   `json:"content"`
	IsError   bool              `json:"is_error"`
}

type citationBody struct {
	Type              string `json:"type"`
	CitedText         string `json:"cited_text"`
	DocumentIndex     *int   `json:"document_index"`
	DocumentTitle     string `json:"document_title"`
	Title             string `json:"title"`
	URL               string `json:"url"`
	Source            string `json:"source"`
	StartChar         int    `json:"start_char_index"`
	EndChar           int    `json:"end_char_index"`
	StartPage         int    `json:"start_page_number"`
	EndPage           int    `json:"end_page_number"`
	StartBlock        int    `json:"start_block_index"`
	EndBlock          int    `json:"end_block_index"`
	SearchResultIndex *int   `json:"search_result_index"`
}

type usageBody struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	CacheCreationTokens int `json:"cache_creation_input_tokens"`
	CacheReadTokens     int `json:"cache_read_input_tokens"`
	OutputDetails       struct {
		ThinkingTokens int `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}

type converter struct {
	forced string
	names  map[string]string
	parts  ai.Parts
}

func (b messageBody) metadata() ai.ProviderData {
	data := ResponseData{ID: b.ID, StopSequence: b.StopSequence, StopDetails: claudewire.Nullless(b.StopDetails)}
	if b.Container != nil && b.Container.ID != "" {
		data.Container = &Container{ID: b.Container.ID, ExpiresAt: b.Container.ExpiresAt}
	}
	var usage struct {
		ServiceTier   string          `json:"service_tier"`
		InferenceGeo  string          `json:"inference_geo"`
		ServerToolUse json.RawMessage `json:"server_tool_use"`
	}
	if err := json.Unmarshal(b.Usage, &usage); err == nil {
		data.ServiceTier = usage.ServiceTier
		data.InferenceGeo = usage.InferenceGeo
		data.ServerToolUse = claudewire.Nullless(usage.ServerToolUse)
	}
	return responseProvider(data)
}

func convertContent(blocks []json.RawMessage, forced string) (ai.Parts, error) {
	c := &converter{forced: forced, names: map[string]string{}}
	for _, raw := range blocks {
		if err := c.add(raw); err != nil {
			return nil, err
		}
	}
	return c.parts, nil
}

func (c *converter) add(raw json.RawMessage) error {
	var b contentBlock
	if err := json.Unmarshal(raw, &b); err != nil {
		return fmt.Errorf("anthropic: decode content block: %w", err)
	}
	if p, ok := c.convert(raw, b); ok {
		c.parts = append(c.parts, p)
	}
	return nil
}

func (c *converter) convert(raw json.RawMessage, b contentBlock) (ai.Part, bool) {
	switch {
	case b.Type == "text":
		cites := citations(b.Citations)
		if b.Text == "" && len(cites) == 0 {
			return nil, false
		}
		return ai.Text{Text: b.Text, Citations: cites}, true
	case b.Type == "thinking" || b.Type == "redacted_thinking":
		return ai.Reasoning{Text: b.Thinking, Provider: blockData(raw)}, true
	case b.Type == "tool_use" && b.Name == c.forced && c.forced != "":
		return forcedText(b.Input), true
	case b.Type == "tool_use":
		return ai.ToolCall{ID: b.ID, Name: b.Name, Input: claudewire.ToolInput(b.Input), Provider: blockData(raw)}, true
	case b.Type == "server_tool_use" || b.Type == "mcp_tool_use":
		name := callName(b)
		c.names[b.ID] = name
		return ai.ToolCall{ID: b.ID, Name: name, Input: claudewire.ToolInput(b.Input), ProviderExecuted: true, Provider: blockData(raw)}, true
	case strings.HasSuffix(b.Type, "_tool_result") && b.ToolUseID != "":
		return c.result(raw, b), true
	default:
		return ai.Custom{Kind: providerName + "." + b.Type, Data: raw, Provider: blockData(raw)}, true
	}
}

func forcedText(input json.RawMessage) ai.Text {
	return ai.Text{Text: string(claudewire.ToolInput(input)), Provider: pdata.Encode(providerName, partData{ForcedTool: true})}
}

func callName(b contentBlock) string {
	if b.Type == "mcp_tool_use" {
		return "mcp." + b.ServerNm + "." + b.Name
	}
	return b.Name
}

func (c *converter) result(raw json.RawMessage, b contentBlock) ai.ToolResult {
	name := c.names[b.ToolUseID]
	if name == "" {
		name = strings.TrimSuffix(b.Type, "_tool_result")
	}
	return resultPart(raw, b, name)
}

func resultPart(raw json.RawMessage, b contentBlock, name string) ai.ToolResult {
	output := raw
	if len(b.Content) > 0 {
		output = b.Content
	}
	return ai.ToolResult{
		CallID:           b.ToolUseID,
		Name:             name,
		Output:           ai.ToolOutput{JSON: output},
		IsError:          b.IsError || contentIsError(b.Content),
		ProviderExecuted: true,
		Provider:         blockData(raw),
	}
}

func contentIsError(content json.RawMessage) bool {
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(content, &probe); err != nil {
		return false
	}
	return strings.HasSuffix(probe.Type, "_error")
}

func citations(raws []json.RawMessage) []ai.Citation {
	var out []ai.Citation
	for _, raw := range raws {
		if c, ok := citationOf(raw); ok {
			out = append(out, c)
		}
	}
	return out
}

func citationOf(raw json.RawMessage) (ai.Citation, bool) {
	var b citationBody
	if err := json.Unmarshal(raw, &b); err != nil {
		return ai.Citation{}, false
	}
	c := ai.Citation{CitedText: b.CitedText, Provider: blockData(raw)}
	switch b.Type {
	case "char_location":
		c.Kind, c.DocIndex, c.Title, c.Start, c.End = "char", b.DocumentIndex, b.DocumentTitle, b.StartChar, b.EndChar
	case "page_location":
		c.Kind, c.DocIndex, c.Title, c.Start, c.End = "page", b.DocumentIndex, b.DocumentTitle, b.StartPage, b.EndPage
	case "content_block_location":
		c.Kind, c.DocIndex, c.Title, c.Start, c.End = "block", b.DocumentIndex, b.DocumentTitle, b.StartBlock, b.EndBlock
	case "web_search_result_location":
		c.Kind, c.URL, c.Title = "web", b.URL, b.Title
	case "search_result_location":
		c.Kind, c.DocIndex, c.URL, c.Title, c.Start, c.End = "search-result", b.SearchResultIndex, b.Source, b.Title, b.StartBlock, b.EndBlock
	default:
		c.Kind, c.URL, c.Title = b.Type, b.URL, cmp.Or(b.Title, b.DocumentTitle)
	}
	return c, true
}

func usageOf(raw json.RawMessage) ai.Usage {
	if len(raw) == 0 || string(raw) == "null" {
		return ai.Usage{}
	}
	var u usageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{
		Input:      u.InputTokens + u.CacheCreationTokens + u.CacheReadTokens,
		Output:     u.OutputTokens,
		CacheRead:  u.CacheReadTokens,
		CacheWrite: u.CacheCreationTokens,
		Reasoning:  min(u.OutputDetails.ThinkingTokens, u.OutputTokens),
		Raw:        raw,
	}
}

func finishOf(stop string) (ai.FinishReason, string) {
	switch stop {
	case "end_turn", "stop_sequence":
		return ai.FinishStop, stop
	case "max_tokens", "model_context_window_exceeded":
		return ai.FinishLength, stop
	case "tool_use":
		return ai.FinishToolCalls, stop
	case "pause_turn":
		return ai.FinishPause, stop
	case "refusal":
		return ai.FinishRefusal, stop
	default:
		return ai.FinishOther, stop
	}
}
