package ollama

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const missingResultText = "error: this tool call has no result"

type wireMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Thinking  string            `json:"thinking,omitzero"`
	Images    []string          `json:"images,omitzero"`
	ToolCalls []json.RawMessage `json:"tool_calls,omitzero"`
	ToolName  string            `json:"tool_name,omitzero"`
}

type wireCall struct {
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type pendingCall struct {
	id   string
	name string
}

type inputBuilder struct {
	out      []wireMessage
	pending  []pendingCall
	warnings []ai.Warning
}

func buildInput(instructions []ai.Text, msgs []ai.Message) ([]wireMessage, []ai.Warning, error) {
	b := &inputBuilder{}
	var system []string
	for _, t := range instructions {
		if t.Text != "" {
			system = append(system, t.Text)
		}
	}
	if len(system) > 0 {
		b.out = append(b.out, wireMessage{Role: "system", Content: strings.Join(system, "\n\n")})
	}
	for _, m := range msgs {
		if err := b.add(m); err != nil {
			return nil, nil, err
		}
	}
	b.flush()
	return b.out, b.warnings, nil
}

func (b *inputBuilder) warn(format string, args ...any) {
	b.warnings = append(b.warnings, ai.Warning{Message: "ollama: " + fmt.Sprintf(format, args...)})
}

func (b *inputBuilder) add(m ai.Message) error {
	switch m.Role {
	case ai.RoleAssistant:
		b.flush()
		b.assistant(m)
		return nil
	case ai.RoleUser, ai.RoleTool:
		return b.user(m)
	default:
		return fmt.Errorf("ollama: unsupported message role %q", m.Role)
	}
}

func (b *inputBuilder) flush() {
	for _, p := range b.pending {
		b.out = append(b.out, wireMessage{Role: "tool", Content: missingResultText, ToolName: p.name})
	}
	b.pending = nil
}

func (b *inputBuilder) assistant(m ai.Message) {
	msg := wireMessage{Role: "assistant"}
	var text strings.Builder
	appendText := func(s string) {
		if text.Len() > 0 {
			text.WriteString("\n")
		}
		text.WriteString(s)
	}
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			text.WriteString(v.Text)
		case ai.Reasoning:
			if _, ok := v.Provider[providerName]; ok {
				msg.Thinking += v.Text
			}
		case ai.ToolCall:
			if v.ProviderExecuted {
				appendText(fmt.Sprintf("[%s call: %s]", v.Name, v.Input))
				continue
			}
			msg.ToolCalls = append(msg.ToolCalls, callOf(v))
			b.pending = append(b.pending, pendingCall{id: v.ID, name: v.Name})
		case ai.ToolResult:
			if v.ProviderExecuted {
				appendText(fmt.Sprintf("[%s result: %s]", v.Name, outputText(v)))
			}
		default:
		}
	}
	msg.Content = text.String()
	if msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0 {
		return
	}
	b.out = append(b.out, msg)
}

func callOf(c ai.ToolCall) json.RawMessage {
	if raw := pdata.Decode[partData](providerName, c.Provider).Call; len(raw) > 0 {
		return raw
	}
	args := c.Input
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	return pdata.JSON(providerName, wireCall{Function: wireFunction{Name: c.Name, Arguments: args}})
}

func (b *inputBuilder) user(m ai.Message) error {
	msg := wireMessage{Role: "user"}
	var text strings.Builder
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			text.WriteString(v.Text)
		case ai.File:
			image, err := imageOf(v)
			if err != nil {
				return err
			}
			msg.Images = append(msg.Images, image)
		case ai.ToolResult:
			b.result(v)
		default:
		}
	}
	msg.Content = text.String()
	if msg.Content == "" && len(msg.Images) == 0 {
		return nil
	}
	b.flush()
	b.out = append(b.out, msg)
	return nil
}

func (b *inputBuilder) result(r ai.ToolResult) {
	if r.ProviderExecuted {
		return
	}
	idx := -1
	for i, p := range b.pending {
		if p.id == r.CallID {
			idx = i
			break
		}
	}
	name := r.Name
	if idx < 0 {
		b.warn("tool result for unknown call %q was dropped", r.CallID)
		return
	}
	if name == "" {
		name = b.pending[idx].name
	}
	b.pending = append(b.pending[:idx], b.pending[idx+1:]...)
	b.out = append(b.out, wireMessage{Role: "tool", Content: outputText(r), ToolName: name})
}

func outputText(r ai.ToolResult) string {
	var s string
	switch {
	case r.Output.Text != "":
		s = r.Output.Text
	case len(r.Output.JSON) > 0:
		s = string(r.Output.JSON)
	default:
		var b strings.Builder
		for _, p := range r.Output.Parts {
			if t, ok := p.(ai.Text); ok {
				b.WriteString(t.Text)
			}
		}
		s = b.String()
	}
	if r.IsError {
		return "error: " + s
	}
	return s
}

func imageOf(f ai.File) (string, error) {
	if !strings.HasPrefix(f.MediaType, "image/") {
		return "", fmt.Errorf("ollama: unsupported file type %q: only images are supported", f.MediaType)
	}
	if len(f.Data) == 0 {
		return "", errors.New("ollama: image file has no data: URLs are not supported")
	}
	return base64.StdEncoding.EncodeToString(f.Data), nil
}
