package chatcompletions

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

type simpleMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type assistantMessage struct {
	Role      string
	Content   any
	Refusal   string
	Reasoning string
	ToolCalls []toolCall
	Extra     map[string]json.RawMessage
	field     string
}

type toolMessage struct {
	Role       string `json:"role"`
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name,omitzero"`
	Content    string `json:"content"`
}

type toolCall struct {
	Index    int    `json:"index,omitzero"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (a assistantMessage) MarshalJSON() ([]byte, error) {
	fields := map[string]any{"role": a.Role}
	if a.Content != nil {
		fields["content"] = a.Content
	}
	if a.Refusal != "" {
		fields["refusal"] = a.Refusal
	}
	if a.Reasoning != "" && a.field != "" {
		fields[a.field] = a.Reasoning
	}
	if len(a.ToolCalls) > 0 {
		fields["tool_calls"] = a.ToolCalls
	}
	typed, err := transport.EncodeJSON(fields)
	if err != nil {
		return nil, err
	}
	return mergeExtra(typed, a.Extra)
}

func (m *Model) messages(c *ai.Call) ([]any, error) {
	var out []any
	if text := openaiwire.Instructions(c.Instructions); text != "" {
		out = append(out, simpleMessage{Role: "system", Content: text})
	}
	called := map[string]bool{}
	answered := map[string]bool{}
	for _, msg := range c.Messages {
		for _, p := range msg.Content {
			switch v := p.(type) {
			case ai.ToolCall:
				called[m.callID(v.ID)] = true
			case ai.ToolResult:
				answered[m.callID(v.CallID)] = true
			default:
			}
		}
	}
	for _, msg := range c.Messages {
		msgs, err := m.historyMessage(msg, called, answered)
		if err != nil {
			return nil, err
		}
		out = append(out, msgs...)
	}
	return out, nil
}

func (m *Model) historyMessage(msg ai.Message, called, answered map[string]bool) ([]any, error) {
	switch msg.Role {
	case ai.RoleUser:
		content, err := m.userContent(msg.Content)
		if err != nil || content == nil {
			return nil, err
		}
		return []any{simpleMessage{Role: "user", Content: content}}, nil
	case ai.RoleAssistant:
		return m.assistant(msg, answered), nil
	case ai.RoleTool:
		return m.toolResults(msg, called), nil
	default:
		return nil, fmt.Errorf("%s: unknown message role %q", m.dialect.Provider, msg.Role)
	}
}

func (m *Model) userContent(parts ai.Parts) (any, error) {
	var out []any
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, map[string]string{"type": "text", "text": v.Text})
		case ai.File:
			c, err := m.file(v)
			if err != nil {
				return nil, err
			}
			out = append(out, c)
		default:
		}
	}
	if len(out) == 0 {
		return nil, nil
	}
	if len(out) == 1 {
		if t, ok := parts[0].(ai.Text); ok {
			return t.Text, nil
		}
	}
	return out, nil
}

func (m *Model) file(f ai.File) (any, error) {
	ref := openaiwire.FileRefOf(m.dialect.Provider, f.Provider)
	switch {
	case strings.HasPrefix(f.MediaType, "image/"):
		url := f.URL
		if len(f.Data) > 0 {
			url = openaiwire.DataURL(f.MediaType, f.Data)
		}
		if url == "" {
			return nil, fmt.Errorf("%s: image part has no data or url", m.dialect.Provider)
		}
		image := map[string]string{"url": url}
		if ref.Detail != "" {
			image["detail"] = ref.Detail
		}
		return map[string]any{"type": "image_url", "image_url": image}, nil
	case strings.HasPrefix(f.MediaType, "audio/"):
		if len(f.Data) == 0 {
			return nil, fmt.Errorf("%s: audio part has no data", m.dialect.Provider)
		}
		return map[string]any{"type": "input_audio", "input_audio": map[string]string{"data": base64.StdEncoding.EncodeToString(f.Data), "format": audioFormat(f.MediaType)}}, nil
	case ref.FileID != "":
		return map[string]any{"type": "file", "file": map[string]string{"file_id": ref.FileID}}, nil
	case len(f.Data) > 0:
		return map[string]any{"type": "file", "file": map[string]string{"file_data": openaiwire.DataURL(f.MediaType, f.Data), "filename": cmp.Or(f.Filename, "file")}}, nil
	default:
		return nil, errors.New(m.dialect.Provider + ": file part has no data or file id")
	}
}

func audioFormat(mediaType string) string {
	switch strings.TrimPrefix(mediaType, "audio/") {
	case "mpeg", "mp3":
		return "mp3"
	case "wav", "x-wav", "wave":
		return "wav"
	default:
		return strings.TrimPrefix(mediaType, "audio/")
	}
}

func (m *Model) assistant(msg ai.Message, answered map[string]bool) []any {
	var texts []string
	out := assistantMessage{Role: "assistant", field: m.dialect.Replay.Field}
	var reasoning []string
	var chunks []json.RawMessage
	verbatim := map[string][]json.RawMessage{}
	for _, p := range msg.Content {
		switch v := p.(type) {
		case ai.Text:
			if pdata.Decode[partData](m.dialect.Provider, v.Provider).Refusal {
				out.Refusal += v.Text
				continue
			}
			texts = append(texts, v.Text)
		case ai.Reasoning:
			d := pdata.Decode[partData](m.dialect.Provider, v.Provider)
			if d.Field != "" {
				reasoning = append(reasoning, v.Text)
			}
			if len(d.Chunk) > 0 {
				chunks = append(chunks, d.Chunk)
			}
			for k, raw := range d.Fields {
				verbatim[k] = append(verbatim[k], raw)
			}
		case ai.ToolCall:
			if v.ProviderExecuted {
				continue
			}
			var call toolCall
			call.ID, call.Type = m.callID(v.ID), "function"
			call.Function.Name, call.Function.Arguments = v.Name, openaiwire.Arguments(v.Input)
			out.ToolCalls = append(out.ToolCalls, call)
		case ai.ToolResult:
			if v.ProviderExecuted && !pdata.Decode[partData](m.dialect.Provider, v.Provider).Derived {
				texts = append(texts, fmt.Sprintf("[%s result] %s", v.Name, cmp.Or(v.Output.Text, string(v.Output.JSON))))
			}
		default:
		}
	}
	if len(texts) > 0 {
		out.Content = strings.Join(texts, "")
	}
	if m.dialect.Replay.Chunks && len(chunks) > 0 {
		content := make([]any, 0, len(chunks)+1)
		for _, c := range chunks {
			content = append(content, c)
		}
		if len(texts) > 0 {
			content = append(content, map[string]string{"type": "text", "text": strings.Join(texts, "")})
		}
		out.Content = content
	}
	if out.Content == nil && out.Refusal == "" && len(out.ToolCalls) == 0 {
		return nil
	}
	m.replayReasoning(&out, reasoning, verbatim)
	result := []any{out}
	for _, call := range out.ToolCalls {
		if !answered[call.ID] {
			answered[call.ID] = true
			missing := toolMessage{Role: "tool", ToolCallID: call.ID, Content: openaiwire.MissingResult}
			if m.dialect.ToolResultName {
				missing.Name = call.Function.Name
			}
			result = append(result, missing)
		}
	}
	return result
}

func (m *Model) replayReasoning(out *assistantMessage, reasoning []string, verbatim map[string][]json.RawMessage) {
	if r := m.dialect.Replay; r.Field != "" && len(verbatim) == 0 && (!r.ToolCallsOnly || len(out.ToolCalls) > 0) {
		out.Reasoning = strings.Join(reasoning, "\n\n")
	}
	for k, items := range verbatim {
		if out.Extra == nil {
			out.Extra = map[string]json.RawMessage{}
		}
		out.Extra[k] = joinVerbatim(items)
	}
}

func (m *Model) toolResults(msg ai.Message, called map[string]bool) []any {
	var out []any
	for _, p := range msg.Content {
		res, ok := p.(ai.ToolResult)
		if !ok || res.ProviderExecuted {
			continue
		}
		id := m.callID(res.CallID)
		if !called[id] {
			continue
		}
		tm := toolMessage{Role: "tool", ToolCallID: id, Content: toolText(res.Output)}
		if m.dialect.ToolResultName {
			tm.Name = res.Name
		}
		out = append(out, tm)
	}
	return out
}

func toolText(o ai.ToolOutput) string {
	switch {
	case len(o.Parts) > 0:
		var texts []string
		for _, p := range o.Parts {
			if t, ok := p.(ai.Text); ok {
				texts = append(texts, t.Text)
			}
		}
		return strings.Join(texts, "\n")
	case len(o.JSON) > 0:
		return string(o.JSON)
	default:
		return o.Text
	}
}
