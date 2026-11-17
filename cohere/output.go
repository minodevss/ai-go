package cohere

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
)

type chatResponse struct {
	ID           string          `json:"id"`
	FinishReason string          `json:"finish_reason"`
	Message      responseMessage `json:"message"`
	Usage        json.RawMessage `json:"usage"`
}

type responseMessage struct {
	Content   []contentBlock    `json:"content"`
	ToolPlan  string            `json:"tool_plan"`
	ToolCalls []wireToolCall    `json:"tool_calls"`
	Citations []json.RawMessage `json:"citations"`
}

type contentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

type citationBody struct {
	Start        int             `json:"start"`
	End          int             `json:"end"`
	Text         string          `json:"text"`
	ContentIndex int             `json:"content_index"`
	Type         string          `json:"type"`
	Sources      []citationSrc   `json:"sources"`
	Raw          json.RawMessage `json:"-"`
}

type citationSrc struct {
	Type       string         `json:"type"`
	ID         string         `json:"id"`
	Document   map[string]any `json:"document"`
	ToolOutput map[string]any `json:"tool_output"`
}

type usageBody struct {
	BilledUnits struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"billed_units"`
	Tokens struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"tokens"`
	CachedTokens float64 `json:"cached_tokens"`
}

func (m *LanguageModel) convert(body chatResponse, docIDs []string) (ai.Parts, ResponseData, error) {
	data := ResponseData{ID: body.ID}
	cites := map[int][]ai.Citation{}
	for _, raw := range body.Message.Citations {
		c, err := parseCitation(raw)
		if err != nil {
			return nil, data, err
		}
		if c.Type != "" && c.Type != "TEXT_CONTENT" {
			data.Citations = append(data.Citations, raw)
			continue
		}
		cites[c.ContentIndex] = append(cites[c.ContentIndex], citationsOf(c, docIDs)...)
	}
	var parts ai.Parts
	for i, b := range body.Message.Content {
		switch b.Type {
		case "thinking":
			parts = append(parts, ai.Reasoning{Text: b.Thinking, Provider: pdata.Encode(providerName, partData{Thinking: true})})
		default:
			parts = append(parts, ai.Text{Text: b.Text, Citations: cites[i]})
		}
	}
	if body.Message.ToolPlan != "" {
		parts = append(parts, ai.Reasoning{Text: body.Message.ToolPlan, Provider: pdata.Encode(providerName, partData{ToolPlan: true})})
	}
	for _, c := range body.Message.ToolCalls {
		parts = append(parts, ai.ToolCall{ID: c.ID, Name: c.Function.Name, Input: openaiwire.ToolInput(c.Function.Arguments)})
	}
	return parts, data, nil
}

func parseCitation(raw json.RawMessage) (citationBody, error) {
	var c citationBody
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, fmt.Errorf("cohere: decode citation: %w", err)
	}
	c.Raw = raw
	return c, nil
}

func citationsOf(c citationBody, docIDs []string) []ai.Citation {
	out := make([]ai.Citation, 0, len(c.Sources))
	for _, s := range c.Sources {
		cite := ai.Citation{Kind: s.Type, Start: c.Start, End: c.End, Provider: pdata.Encode(providerName, partData{Citation: c.Raw})}
		src := s.Document
		if s.Type == "tool" {
			src = s.ToolOutput
		}
		cite.Title = firstString(src, "title", "name")
		cite.URL = firstString(src, "url")
		cite.CitedText = firstString(src, "snippet", "text", "content")
		if s.Type == "document" {
			cite.DocIndex = docIndex(docIDs, s.ID)
		}
		out = append(out, cite)
	}
	return out
}

func docIndex(ids []string, id string) *int {
	for i, v := range ids {
		if v == id {
			return &i
		}
	}
	return nil
}

func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := m[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

func finishOf(raw string) ai.FinishReason {
	switch raw {
	case "COMPLETE", "STOP_SEQUENCE":
		return ai.FinishStop
	case "MAX_TOKENS":
		return ai.FinishLength
	case "TOOL_CALL":
		return ai.FinishToolCalls
	case "ERROR", "TIMEOUT":
		return ai.FinishError
	default:
		return ai.FinishOther
	}
}

func usageOf(raw json.RawMessage) ai.Usage {
	if len(raw) == 0 || string(raw) == "null" {
		return ai.Usage{}
	}
	var u usageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	in, out := u.Tokens.InputTokens, u.Tokens.OutputTokens
	if in == 0 && out == 0 {
		in, out = u.BilledUnits.InputTokens, u.BilledUnits.OutputTokens
	}
	cached := min(int(u.CachedTokens), in)
	return ai.Usage{Input: in, Output: out, CacheRead: cached, Raw: raw}
}

func blockID(i int) string { return strconv.Itoa(i) }
