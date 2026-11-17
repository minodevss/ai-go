package cohere

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const missingResultText = "error: this tool call has no result"

type wireMessage struct {
	Role       string            `json:"role"`
	Content    any               `json:"content,omitzero"`
	ToolCalls  []wireToolCall    `json:"tool_calls,omitzero"`
	ToolPlan   string            `json:"tool_plan,omitzero"`
	Citations  []json.RawMessage `json:"citations,omitzero"`
	ToolCallID string            `json:"tool_call_id,omitzero"`
}

type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type thinkingBlock struct {
	Type     string `json:"type"`
	Thinking string `json:"thinking"`
}

type imageURL struct {
	URL string `json:"url"`
}

type imageBlock struct {
	Type     string   `json:"type"`
	ImageURL imageURL `json:"image_url"`
}

type documentBlock struct {
	Type     string       `json:"type"`
	Document wireDocument `json:"document"`
}

type wireDocument struct {
	ID   string `json:"id,omitzero"`
	Data any    `json:"data"`
}

type wireToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function wireToolFunction `json:"function"`
}

type wireToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type input struct {
	messages  []wireMessage
	documents []wireDocument
	warnings  []ai.Warning
}

type inputBuilder struct {
	input
	pending []pendingCall
}

type pendingCall struct {
	id   string
	name string
}

func buildInput(instructions []ai.Text, msgs []ai.Message) (input, error) {
	b := &inputBuilder{}
	for _, t := range instructions {
		if t.Text != "" {
			b.messages = append(b.messages, wireMessage{Role: "system", Content: t.Text})
		}
	}
	for i, m := range msgs {
		var err error
		switch m.Role {
		case ai.RoleAssistant:
			b.assistant(m)
		case ai.RoleUser, ai.RoleTool:
			err = b.user(m)
		default:
			err = fmt.Errorf("cohere: unknown role %q", m.Role)
		}
		if err != nil {
			return input{}, fmt.Errorf("%w (message %d)", err, i)
		}
	}
	b.flush()
	return b.input, nil
}

func (b *inputBuilder) user(m ai.Message) error {
	var blocks []any
	var results []ai.ToolResult
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			blocks = append(blocks, textBlock{Type: "text", Text: v.Text})
		case ai.File:
			block, err := b.file(v)
			if err != nil {
				return err
			}
			if block != nil {
				blocks = append(blocks, block)
			}
		case ai.ToolResult:
			results = append(results, v)
		default:
		}
	}
	for _, r := range results {
		b.result(r)
	}
	if len(blocks) == 0 {
		return nil
	}
	b.flush()
	b.messages = append(b.messages, wireMessage{Role: "user", Content: blocks})
	return nil
}

func (b *inputBuilder) file(f ai.File) (any, error) {
	switch {
	case strings.HasPrefix(f.MediaType, "image/"):
		return imageBlock{Type: "image_url", ImageURL: imageURL{URL: fileURL(f)}}, nil
	case strings.HasPrefix(f.MediaType, "text/") || f.MediaType == "application/json":
		if len(f.Data) == 0 {
			return nil, errors.New("cohere: a text document needs Data")
		}
		data := map[string]any{"text": string(f.Data)}
		if f.Filename != "" {
			data["title"] = f.Filename
		}
		b.documents = append(b.documents, wireDocument{ID: strconv.Itoa(len(b.documents)), Data: data})
		return nil, nil
	default:
		return nil, fmt.Errorf("cohere: file media type %q is not supported; use an image or a text document", f.MediaType)
	}
}

func fileURL(f ai.File) string {
	if len(f.Data) == 0 {
		return f.URL
	}
	return "data:" + f.MediaType + ";base64," + base64.StdEncoding.EncodeToString(f.Data)
}

func (b *inputBuilder) assistant(m ai.Message) {
	b.flush()
	msg := wireMessage{Role: "assistant"}
	var content []any
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			content = append(content, textBlock{Type: "text", Text: v.Text})
			msg.Citations = appendCitations(msg.Citations, v.Citations)
		case ai.Reasoning:
			d := pdata.Decode[partData](providerName, v.Provider)
			switch {
			case d.ToolPlan:
				msg.ToolPlan = v.Text
			case d.Thinking:
				content = append(content, thinkingBlock{Type: "thinking", Thinking: v.Text})
			default:
			}
		case ai.ToolCall:
			if v.ProviderExecuted {
				continue
			}
			msg.ToolCalls = append(msg.ToolCalls, wireToolCall{ID: v.ID, Type: "function", Function: wireToolFunction{Name: v.Name, Arguments: argumentsOf(v.Input)}})
			b.pending = append(b.pending, pendingCall{id: v.ID, name: v.Name})
		case ai.ToolResult:
			if v.ProviderExecuted {
				content = append(content, textBlock{Type: "text", Text: outputText(v)})
			}
		default:
		}
	}
	if len(content) == 0 && len(msg.ToolCalls) == 0 && msg.ToolPlan == "" {
		return
	}
	if len(content) > 0 {
		msg.Content = content
	}
	b.messages = append(b.messages, msg)
}

func appendCitations(dst []json.RawMessage, cites []ai.Citation) []json.RawMessage {
	for _, c := range cites {
		raw := pdata.Decode[partData](providerName, c.Provider).Citation
		if len(raw) == 0 || slices.ContainsFunc(dst, func(r json.RawMessage) bool { return bytes.Equal(r, raw) }) {
			continue
		}
		dst = append(dst, raw)
	}
	return dst
}

func (b *inputBuilder) result(r ai.ToolResult) {
	i := slices.IndexFunc(b.pending, func(c pendingCall) bool { return c.id == r.CallID })
	if i < 0 {
		return
	}
	b.pending = slices.Delete(b.pending, i, i+1)
	b.messages = append(b.messages, wireMessage{Role: "tool", ToolCallID: r.CallID, Content: b.resultContent(r)})
}

func (b *inputBuilder) resultContent(r ai.ToolResult) []any {
	prefix := ""
	if r.IsError {
		prefix = "error: "
	}
	switch {
	case len(r.Output.Parts) > 0:
		var out []any
		for _, p := range r.Output.Parts {
			t, ok := p.(ai.Text)
			if !ok {
				b.warnings = append(b.warnings, ai.Warning{Message: "cohere: tool result part of a type other than text was dropped"})
				continue
			}
			out = append(out, documentBlock{Type: "document", Document: wireDocument{Data: prefix + t.Text}})
		}
		return out
	case len(r.Output.JSON) > 0:
		return []any{documentBlock{Type: "document", Document: wireDocument{Data: prefix + compact(r.Output.JSON)}}}
	default:
		return []any{documentBlock{Type: "document", Document: wireDocument{Data: prefix + r.Output.Text}}}
	}
}

func (b *inputBuilder) flush() {
	for _, c := range b.pending {
		b.messages = append(b.messages, wireMessage{Role: "tool", ToolCallID: c.id, Content: []any{documentBlock{Type: "document", Document: wireDocument{Data: missingResultText}}}})
	}
	b.pending = nil
}

func outputText(r ai.ToolResult) string {
	switch {
	case len(r.Output.JSON) > 0:
		return compact(r.Output.JSON)
	case len(r.Output.Parts) > 0:
		var s strings.Builder
		for _, p := range r.Output.Parts {
			if t, ok := p.(ai.Text); ok {
				s.WriteString(t.Text)
			}
		}
		return s.String()
	default:
		return r.Output.Text
	}
}

func argumentsOf(raw json.RawMessage) string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "{}"
	}
	return compact(raw)
}

func compact(raw json.RawMessage) string {
	var b bytes.Buffer
	if err := json.Compact(&b, raw); err != nil {
		return string(raw)
	}
	return b.String()
}
