package bedrock

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
	"github.com/minodevss/ai-go/internal/pdata"
)

type converseResponse struct {
	Output struct {
		Message struct {
			Content []json.RawMessage `json:"content"`
		} `json:"message"`
	} `json:"output"`
	StopReason                    string          `json:"stopReason"`
	Usage                         json.RawMessage `json:"usage"`
	Metrics                       json.RawMessage `json:"metrics"`
	Trace                         json.RawMessage `json:"trace"`
	PerformanceConfig             json.RawMessage `json:"performanceConfig"`
	ServiceTier                   json.RawMessage `json:"serviceTier"`
	AdditionalModelResponseFields json.RawMessage `json:"additionalModelResponseFields"`
}

type contentBlock struct {
	Text             *string `json:"text"`
	ReasoningContent *struct {
		Text *struct {
			Text string `json:"text"`
		} `json:"reasoningText"`
	} `json:"reasoningContent"`
	ToolUse *struct {
		ID    string          `json:"toolUseId"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
		Type  string          `json:"type"`
	} `json:"toolUse"`
	ToolResult *struct {
		ID      string          `json:"toolUseId"`
		Content json.RawMessage `json:"content"`
		Status  string          `json:"status"`
		Type    string          `json:"type"`
	} `json:"toolResult"`
	Citations *struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Citations []json.RawMessage `json:"citations"`
	} `json:"citationsContent"`
}

type citationBody struct {
	Title    string `json:"title"`
	Source   string `json:"source"`
	Location struct {
		Char *struct {
			Index int `json:"documentIndex"`
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"documentChar"`
		Page *struct {
			Index int `json:"documentIndex"`
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"documentPage"`
		Chunk *struct {
			Index int `json:"documentIndex"`
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"documentChunk"`
		SearchResult *struct {
			Index int `json:"searchResultIndex"`
			Start int `json:"start"`
			End   int `json:"end"`
		} `json:"searchResultLocation"`
		Web *struct {
			URL string `json:"url"`
		} `json:"web"`
	} `json:"location"`
	SourceContent []struct {
		Text string `json:"text"`
	} `json:"sourceContent"`
}

type usageBody struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
}

func (r converseResponse) metadata() ai.ProviderData {
	return responseData(r.AdditionalModelResponseFields, r.Metrics, r.Trace, r.PerformanceConfig, r.ServiceTier)
}

func responseData(additional, metrics, trace, performance, tier json.RawMessage) ai.ProviderData {
	return pdata.Encode(providerName, ResponseData{
		AdditionalModelResponseFields: claudewire.Nullless(additional),
		Metrics:                       claudewire.Nullless(metrics),
		Trace:                         claudewire.Nullless(trace),
		PerformanceConfig:             claudewire.Nullless(performance),
		ServiceTier:                   claudewire.Nullless(tier),
	})
}

type converter struct {
	model  string
	forced string
	names  map[string]string
	parts  ai.Parts
}

func convertContent(blocks []json.RawMessage, model, forced string) (ai.Parts, error) {
	c := &converter{model: model, forced: forced, names: map[string]string{}}
	for _, raw := range blocks {
		var b contentBlock
		if err := json.Unmarshal(raw, &b); err != nil {
			return nil, fmt.Errorf("bedrock: decode content block: %w", err)
		}
		if p, ok := c.convert(raw, b); ok {
			c.parts = append(c.parts, p)
		}
	}
	return c.parts, nil
}

func (c *converter) convert(raw json.RawMessage, b contentBlock) (ai.Part, bool) {
	switch {
	case b.Text != nil:
		if *b.Text == "" {
			return nil, false
		}
		return ai.Text{Text: *b.Text}, true
	case b.ReasoningContent != nil:
		text := ""
		if b.ReasoningContent.Text != nil {
			text = b.ReasoningContent.Text.Text
		}
		return ai.Reasoning{Text: text, Provider: blockData(raw, c.model)}, true
	case b.ToolUse != nil && b.ToolUse.Type == "" && b.ToolUse.Name == c.forced && c.forced != "":
		return forcedText(b.ToolUse.Input), true
	case b.ToolUse != nil:
		u := b.ToolUse
		executed := u.Type != ""
		if executed {
			c.names[u.ID] = u.Name
		}
		return ai.ToolCall{ID: u.ID, Name: u.Name, Input: claudewire.ToolInput(u.Input), ProviderExecuted: executed, Provider: blockData(raw, c.model)}, true
	case b.ToolResult != nil:
		return c.result(raw, b), true
	case b.Citations != nil:
		return citedText(raw, b), true
	default:
		return ai.Custom{Kind: customKind(raw), Data: raw, Provider: blockData(raw, c.model)}, true
	}
}

func customKind(raw json.RawMessage) string {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || len(fields) != 1 {
		return providerName
	}
	for key := range fields {
		return providerName + "." + key
	}
	return providerName
}

func forcedText(input json.RawMessage) ai.Text {
	return ai.Text{Text: string(claudewire.ToolInput(input)), Provider: pdata.Encode(providerName, partData{ForcedTool: true})}
}

func (c *converter) result(raw json.RawMessage, b contentBlock) ai.ToolResult {
	r := b.ToolResult
	name := c.names[r.ID]
	if name == "" {
		name = r.Type
	}
	output := raw
	if len(r.Content) > 0 {
		output = r.Content
	}
	return ai.ToolResult{
		CallID:           r.ID,
		Name:             name,
		Output:           ai.ToolOutput{JSON: output},
		IsError:          r.Status == "error",
		ProviderExecuted: true,
		Provider:         blockData(raw, c.model),
	}
}

func citedText(raw json.RawMessage, b contentBlock) ai.Text {
	var text strings.Builder
	for _, part := range b.Citations.Content {
		text.WriteString(part.Text)
	}
	return ai.Text{Text: text.String(), Citations: citations(b.Citations.Citations), Provider: blockData(raw, "")}
}

func citations(raws []json.RawMessage) []ai.Citation {
	var out []ai.Citation
	for _, raw := range raws {
		var b citationBody
		if err := json.Unmarshal(raw, &b); err != nil {
			continue
		}
		out = append(out, citationOf(raw, b))
	}
	return out
}

func citationOf(raw json.RawMessage, b citationBody) ai.Citation {
	c := ai.Citation{Title: b.Title, Provider: blockData(raw, "")}
	var cited []string
	for _, s := range b.SourceContent {
		cited = append(cited, s.Text)
	}
	c.CitedText = strings.Join(cited, "")
	loc := b.Location
	switch {
	case loc.Char != nil:
		c.Kind, c.DocIndex, c.Start, c.End = "char", &loc.Char.Index, loc.Char.Start, loc.Char.End
	case loc.Page != nil:
		c.Kind, c.DocIndex, c.Start, c.End = "page", &loc.Page.Index, loc.Page.Start, loc.Page.End
	case loc.Chunk != nil:
		c.Kind, c.DocIndex, c.Start, c.End = "chunk", &loc.Chunk.Index, loc.Chunk.Start, loc.Chunk.End
	case loc.SearchResult != nil:
		c.Kind, c.DocIndex, c.Start, c.End, c.URL = "search-result", &loc.SearchResult.Index, loc.SearchResult.Start, loc.SearchResult.End, b.Source
	case loc.Web != nil:
		c.Kind, c.URL = "web", loc.Web.URL
	default:
		c.Kind, c.URL = "unknown", b.Source
	}
	return c
}

func usageOf(raw json.RawMessage) ai.Usage {
	// Input tokens exclude cached tokens when prompt caching is on, so the total is the sum: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
	if len(raw) == 0 || string(raw) == "null" {
		return ai.Usage{}
	}
	var u usageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{
		Input:      u.InputTokens + u.CacheReadInputTokens + u.CacheWriteInputTokens,
		Output:     u.OutputTokens,
		CacheRead:  u.CacheReadInputTokens,
		CacheWrite: u.CacheWriteInputTokens,
		Raw:        raw,
	}
}

func finishOf(stop string, clientCall bool) (ai.FinishReason, string) {
	switch stop {
	case "end_turn", "stop_sequence":
		return ai.FinishStop, stop
	case "max_tokens", "model_context_window_exceeded":
		return ai.FinishLength, stop
	case "tool_use":
		if !clientCall {
			return ai.FinishStop, stop
		}
		return ai.FinishToolCalls, stop
	case "guardrail_intervened", "content_filtered":
		return ai.FinishContentFilter, stop
	case "malformed_model_output", "malformed_tool_use":
		return ai.FinishError, stop
	default:
		return ai.FinishOther, stop
	}
}
