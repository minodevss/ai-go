package bedrock_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type wireBlock struct {
	kind      string
	text      strings.Builder
	signature string
	redacted  string
	input     strings.Builder
	id        string
	name      string
	toolType  string
	status    string
	citations []json.RawMessage
	results   []any
}

type wireText struct {
	Text string `json:"text"`
}

type wireReasoning struct {
	ReasoningContent struct {
		Text *struct {
			Text      string `json:"text"`
			Signature string `json:"signature,omitzero"`
		} `json:"reasoningText,omitzero"`
		Redacted string `json:"redactedContent,omitzero"`
	} `json:"reasoningContent"`
}

type wireToolUse struct {
	ToolUse struct {
		ID    string          `json:"toolUseId"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
		Type  string          `json:"type,omitzero"`
	} `json:"toolUse"`
}

type wireToolResult struct {
	ToolResult struct {
		ID      string `json:"toolUseId"`
		Content []any  `json:"content"`
		Status  string `json:"status,omitzero"`
		Type    string `json:"type,omitzero"`
	} `json:"toolResult"`
}

type wireCited struct {
	CitationsContent struct {
		Content   []wireText        `json:"content"`
		Citations []json.RawMessage `json:"citations"`
	} `json:"citationsContent"`
}

type startPayload struct {
	Index int `json:"contentBlockIndex"`
	Start struct {
		ToolUse *struct {
			ID   string `json:"toolUseId"`
			Name string `json:"name"`
			Type string `json:"type"`
		} `json:"toolUse"`
		ToolResult *struct {
			ID     string `json:"toolUseId"`
			Status string `json:"status"`
			Type   string `json:"type"`
		} `json:"toolResult"`
	} `json:"start"`
}

type deltaPayload struct {
	Index int `json:"contentBlockIndex"`
	Delta struct {
		Text      *string `json:"text"`
		Reasoning *struct {
			Text      *string `json:"text"`
			Signature *string `json:"signature"`
			Redacted  *string `json:"redactedContent"`
		} `json:"reasoningContent"`
		ToolUse *struct {
			Input string `json:"input"`
		} `json:"toolUse"`
		Citation   json.RawMessage   `json:"citation"`
		ToolResult []json.RawMessage `json:"toolResult"`
	} `json:"delta"`
}

func marshal(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSpace(buf.Bytes()), nil
}

func streamBlocks(body []byte) ([]json.RawMessage, error) {
	var recorded struct {
		Events []struct {
			Type    string          `json:"type"`
			Payload json.RawMessage `json:"payload"`
		} `json:"events"`
	}
	if err := json.Unmarshal(body, &recorded); err != nil {
		return nil, err
	}
	blocks := map[int]*wireBlock{}
	var order []int
	get := func(i int) *wireBlock {
		if b, ok := blocks[i]; ok {
			return b
		}
		b := &wireBlock{}
		blocks[i] = b
		order = append(order, i)
		return b
	}
	for _, e := range recorded.Events {
		switch e.Type {
		case "contentBlockStart":
			var p startPayload
			if err := json.Unmarshal(e.Payload, &p); err != nil {
				return nil, err
			}
			b := get(p.Index)
			if u := p.Start.ToolUse; u != nil {
				b.kind, b.id, b.name, b.toolType = "tool", u.ID, u.Name, u.Type
			}
			if r := p.Start.ToolResult; r != nil {
				b.kind, b.id, b.status, b.toolType = "tresult", r.ID, r.Status, r.Type
			}
		case "contentBlockDelta":
			var p deltaPayload
			if err := json.Unmarshal(e.Payload, &p); err != nil {
				return nil, err
			}
			p.apply(get(p.Index))
		}
	}
	var out []json.RawMessage
	for _, i := range order {
		raw, err := blocks[i].encode()
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return out, nil
}

func (p deltaPayload) apply(b *wireBlock) {
	d := p.Delta
	switch {
	case d.Text != nil:
		b.kind = "text"
		b.text.WriteString(*d.Text)
	case d.Citation != nil:
		b.kind = "text"
		b.citations = append(b.citations, d.Citation)
	case d.Reasoning != nil:
		b.kind = "reasoning"
		if d.Reasoning.Text != nil {
			b.text.WriteString(*d.Reasoning.Text)
		}
		if d.Reasoning.Signature != nil {
			b.signature += *d.Reasoning.Signature
		}
		if d.Reasoning.Redacted != nil {
			b.redacted += *d.Reasoning.Redacted
		}
	case d.ToolUse != nil:
		b.input.WriteString(d.ToolUse.Input)
	case d.ToolResult != nil:
		for _, raw := range d.ToolResult {
			b.results = append(b.results, raw)
		}
	}
}

func (b *wireBlock) encode() (json.RawMessage, error) {
	switch b.kind {
	case "text":
		if len(b.citations) == 0 {
			return marshal(wireText{Text: b.text.String()})
		}
		var blk wireCited
		blk.CitationsContent.Content = []wireText{{Text: b.text.String()}}
		blk.CitationsContent.Citations = b.citations
		return marshal(blk)
	case "reasoning":
		var blk wireReasoning
		if b.redacted != "" {
			blk.ReasoningContent.Redacted = b.redacted
		} else {
			blk.ReasoningContent.Text = &struct {
				Text      string `json:"text"`
				Signature string `json:"signature,omitzero"`
			}{Text: b.text.String(), Signature: b.signature}
		}
		return marshal(blk)
	case "tool":
		var blk wireToolUse
		blk.ToolUse.ID, blk.ToolUse.Name, blk.ToolUse.Input, blk.ToolUse.Type = b.id, b.name, json.RawMessage(b.input.String()), b.toolType
		return marshal(blk)
	case "tresult":
		var blk wireToolResult
		blk.ToolResult.ID, blk.ToolResult.Content, blk.ToolResult.Status, blk.ToolResult.Type = b.id, b.results, b.status, b.toolType
		return marshal(blk)
	default:
		return nil, fmt.Errorf("bedrock: unknown wire block kind %q", b.kind)
	}
}
