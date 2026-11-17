package chatcompletions

import (
	"bytes"
	"cmp"
	"encoding/json"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
)

type response struct {
	ID          string `json:"id"`
	ServiceTier string `json:"service_tier"`
	Choices     []struct {
		Message      json.RawMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

type message struct {
	Content     json.RawMessage   `json:"content"`
	Refusal     *string           `json:"refusal"`
	ToolCalls   []toolCall        `json:"tool_calls"`
	Annotations []json.RawMessage `json:"annotations"`
}

type contentChunk struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	Signature string `json:"signature"`
	Thinking  []struct {
		Text string `json:"text"`
	} `json:"thinking"`
}

type content struct {
	text      string
	thinking  string
	signature string
	chunked   bool
}

func parseContent(raw json.RawMessage) content {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return content{}
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return content{text: text}
	}
	var chunks []contentChunk
	if json.Unmarshal(raw, &chunks) != nil {
		return content{}
	}
	out := content{chunked: true}
	var texts, thoughts []string
	for _, c := range chunks {
		switch c.Type {
		case "text":
			texts = append(texts, c.Text)
		case "thinking":
			for _, t := range c.Thinking {
				thoughts = append(thoughts, t.Text)
			}
			out.signature = cmp.Or(c.Signature, out.signature)
		default:
		}
	}
	out.text = strings.Join(texts, "")
	out.thinking = strings.Join(thoughts, "")
	return out
}

func thinkingChunk(text, signature string) json.RawMessage {
	chunk := map[string]any{"type": "thinking", "thinking": []map[string]string{{"type": "text", "text": text}}, "closed": true}
	if signature != "" {
		chunk["signature"] = signature
	}
	return pdata.JSON(packageName, chunk)
}

func (m *Model) reasoningFields() []string {
	if m.dialect.ReasoningFields != nil {
		return m.dialect.ReasoningFields
	}
	return []string{"reasoning_content", "reasoning"}
}

func (m *Model) reasoningText(fields map[string]json.RawMessage) (text, field string) {
	for _, name := range m.reasoningFields() {
		var s string
		if json.Unmarshal(fields[name], &s) == nil && s != "" {
			return s, name
		}
	}
	return "", ""
}

func verbatimOf(keys []string, raw json.RawMessage) map[string]json.RawMessage {
	if len(keys) == 0 {
		return nil
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return nil
	}
	var out map[string]json.RawMessage
	for _, k := range keys {
		if v, ok := fields[k]; ok && string(bytes.TrimSpace(v)) != "null" {
			if out == nil {
				out = map[string]json.RawMessage{}
			}
			out[k] = v
		}
	}
	return out
}

func joinVerbatim(items []json.RawMessage) json.RawMessage {
	var arrays []json.RawMessage
	var texts strings.Builder
	allArrays, allStrings := true, true
	for _, raw := range items {
		var list []json.RawMessage
		if json.Unmarshal(raw, &list) == nil {
			arrays = append(arrays, list...)
			allStrings = false
			continue
		}
		var s string
		if json.Unmarshal(raw, &s) == nil {
			texts.WriteString(s)
			allArrays = false
			continue
		}
		allArrays, allStrings = false, false
	}
	switch {
	case allArrays:
		return pdata.JSON(packageName, arrays)
	case allStrings:
		return pdata.JSON(packageName, texts.String())
	default:
		return items[len(items)-1]
	}
}

func (m *Model) reasoningPart(text, field string, signature string, verbatim map[string]json.RawMessage) (ai.Reasoning, bool) {
	if text == "" && len(verbatim) == 0 {
		return ai.Reasoning{}, false
	}
	d := partData{Field: field, Fields: verbatim}
	if m.dialect.Replay.Chunks && text != "" {
		d.Chunk = thinkingChunk(text, signature)
		d.Field = "content"
	}
	return ai.Reasoning{Text: text, Provider: pdata.Encode(m.dialect.Provider, d)}, true
}

func (m *Model) parts(msg message, verbatim map[string]json.RawMessage, extra []ai.Citation, rawMsg json.RawMessage) (ai.Parts, bool) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(rawMsg, &fields); err != nil {
		fields = nil
	}
	body := parseContent(msg.Content)
	text, field := m.reasoningText(fields)
	signature := ""
	if body.thinking != "" {
		text, field, signature = body.thinking, "content", body.signature
	}
	var parts ai.Parts
	if p, ok := m.reasoningPart(text, field, signature, verbatim); ok {
		parts = append(parts, p)
	}
	refusal := false
	if body.text != "" {
		cites := append(openaiwire.Citations(m.dialect.Provider, msg.Annotations), extra...)
		parts = append(parts, ai.Text{Text: body.text, Citations: cites})
	}
	if msg.Refusal != nil && *msg.Refusal != "" {
		refusal = true
		parts = append(parts, ai.Text{Text: *msg.Refusal, Provider: pdata.Encode(m.dialect.Provider, partData{Refusal: true})})
	}
	for _, c := range msg.ToolCalls {
		parts = append(parts, ai.ToolCall{ID: c.ID, Name: c.Function.Name, Input: openaiwire.ToolInput(c.Function.Arguments)})
	}
	return parts, refusal
}

func finishReason(reason string, refusal bool) (ai.FinishReason, string) {
	switch reason {
	case "stop", "eos":
		if refusal {
			return ai.FinishRefusal, reason
		}
		return ai.FinishStop, reason
	case "length", "model_length":
		return ai.FinishLength, reason
	case "tool_calls", "function_call":
		return ai.FinishToolCalls, reason
	case "content_filter":
		return ai.FinishContentFilter, reason
	default:
		return ai.FinishOther, reason
	}
}
