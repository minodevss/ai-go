package openai

import (
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
)

const (
	maxResponsesCallID = 64
)

type inputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type inputImage struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url,omitzero"`
	FileID   string `json:"file_id,omitzero"`
	Detail   string `json:"detail,omitzero"`
}

type inputFile struct {
	Type     string `json:"type"`
	FileData string `json:"file_data,omitzero"`
	FileURL  string `json:"file_url,omitzero"`
	FileID   string `json:"file_id,omitzero"`
	Filename string `json:"filename,omitzero"`
}

type inputMessage struct {
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

type functionCall struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type functionOutput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output any    `json:"output"`
}

type inputItem struct {
	value  any
	callID string
}

type inputBuilder struct {
	items    []inputItem
	outputs  map[string]bool
	warnings []ai.Warning
}

func buildInput(msgs []ai.Message) ([]any, []ai.Warning, error) {
	b := &inputBuilder{outputs: map[string]bool{}}
	for _, m := range msgs {
		var err error
		switch m.Role {
		case ai.RoleUser:
			err = b.user(m)
		case ai.RoleAssistant:
			err = b.assistant(m)
		case ai.RoleTool:
			err = b.tool(m)
		default:
			err = fmt.Errorf("openai: unknown message role %q", m.Role)
		}
		if err != nil {
			return nil, nil, err
		}
	}
	return b.finish(), b.warnings, nil
}

func (b *inputBuilder) finish() []any {
	out := make([]any, 0, len(b.items))
	for _, it := range b.items {
		out = append(out, it.value)
		if it.callID != "" && !b.outputs[it.callID] {
			b.outputs[it.callID] = true
			out = append(out, functionOutput{Type: "function_call_output", CallID: it.callID, Output: openaiwire.MissingResult})
		}
	}
	return out
}

func (b *inputBuilder) add(v any) { b.items = append(b.items, inputItem{value: v}) }

func (b *inputBuilder) user(m ai.Message) error {
	content, err := inputContent(m.Content)
	if err != nil {
		return err
	}
	if len(content) == 0 {
		return nil
	}
	b.add(inputMessage{Role: "user", Content: content})
	return nil
}

func (b *inputBuilder) tool(m ai.Message) error {
	for _, p := range m.Content {
		res, ok := p.(ai.ToolResult)
		if !ok {
			continue
		}
		if err := b.result(res); err != nil {
			return err
		}
	}
	return nil
}

func (b *inputBuilder) result(res ai.ToolResult) error {
	if res.ProviderExecuted {
		b.foreignResult(res)
		return nil
	}
	out, err := toolOutput(res.Output)
	if err != nil {
		return err
	}
	id := openaiwire.CallID(res.CallID, maxResponsesCallID)
	b.outputs[id] = true
	b.add(functionOutput{Type: "function_call_output", CallID: id, Output: out})
	return nil
}

func (b *inputBuilder) foreignResult(res ai.ToolResult) {
	if pdata.Decode[partData](providerName, res.Provider).Derived {
		return
	}
	text := res.Output.Text
	if text == "" {
		text = string(res.Output.JSON)
	}
	b.add(assistantText(fmt.Sprintf("[%s result] %s", res.Name, text)))
}

func (b *inputBuilder) assistant(m ai.Message) error {
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			b.text(v)
		case ai.Reasoning:
			if d := pdata.Decode[partData](providerName, v.Provider); len(d.Item) > 0 {
				b.add(d.Item)
			}
		case ai.ToolCall:
			b.call(v)
		case ai.ToolResult:
			if err := b.result(v); err != nil {
				return err
			}
		case ai.Custom:
			if d := pdata.Decode[partData](providerName, v.Provider); len(d.Item) > 0 {
				b.add(d.Item)
			}
		case ai.File:
			if !pdata.Decode[partData](providerName, v.Provider).Derived {
				b.warnings = append(b.warnings, ai.Warning{Message: "openai: an assistant file was dropped because the Responses API cannot take it as history"})
			}
		default:
		}
	}
	return nil
}

func (b *inputBuilder) text(t ai.Text) {
	d := pdata.Decode[partData](providerName, t.Provider)
	switch {
	case len(d.Item) > 0:
		b.add(d.Item)
	case d.ID != "" || d.Derived:
	default:
		b.add(assistantText(t.Text))
	}
}

func (b *inputBuilder) call(c ai.ToolCall) {
	d := pdata.Decode[partData](providerName, c.Provider)
	if len(d.Item) > 0 {
		if c.ProviderExecuted {
			b.add(d.Item)
			return
		}
		b.items = append(b.items, inputItem{value: d.Item, callID: openaiwire.CallID(c.ID, maxResponsesCallID)})
		return
	}
	if c.ProviderExecuted {
		return
	}
	id := openaiwire.CallID(c.ID, maxResponsesCallID)
	b.items = append(b.items, inputItem{value: functionCall{Type: "function_call", CallID: id, Name: c.Name, Arguments: openaiwire.Arguments(c.Input)}, callID: id})
}

func assistantText(s string) inputMessage {
	return inputMessage{Role: "assistant", Content: []any{inputText{Type: "output_text", Text: s}}}
}

func inputContent(parts ai.Parts) ([]any, error) {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, inputText{Type: "input_text", Text: v.Text})
		case ai.File:
			c, err := fileContent(v)
			if err != nil {
				return nil, err
			}
			out = append(out, c)
		default:
		}
	}
	return out, nil
}

func fileContent(f ai.File) (any, error) {
	ref := openaiwire.FileRefOf(providerName, f.Provider)
	isImage := strings.HasPrefix(f.MediaType, "image/")
	switch {
	case ref.FileID != "" && isImage:
		return inputImage{Type: "input_image", FileID: ref.FileID, Detail: ref.Detail}, nil
	case ref.FileID != "":
		return inputFile{Type: "input_file", FileID: ref.FileID}, nil
	case isImage && len(f.Data) > 0:
		return inputImage{Type: "input_image", ImageURL: openaiwire.DataURL(f.MediaType, f.Data), Detail: ref.Detail}, nil
	case isImage && f.URL != "":
		return inputImage{Type: "input_image", ImageURL: f.URL, Detail: ref.Detail}, nil
	case strings.HasPrefix(f.MediaType, "audio/"):
		return nil, errors.New("openai: the Responses API does not accept audio input; use a Chat Completions model")
	case len(f.Data) > 0:
		name := f.Filename
		if name == "" {
			name = "file"
		}
		return inputFile{Type: "input_file", FileData: openaiwire.DataURL(f.MediaType, f.Data), Filename: name}, nil
	case f.URL != "":
		return inputFile{Type: "input_file", FileURL: f.URL}, nil
	default:
		return nil, errors.New("openai: file part has no data, url or file id")
	}
}

func toolOutput(o ai.ToolOutput) (any, error) {
	switch {
	case len(o.Parts) > 0:
		content, err := inputContent(o.Parts)
		if err != nil {
			return nil, err
		}
		return content, nil
	case len(o.JSON) > 0:
		return string(o.JSON), nil
	default:
		return o.Text, nil
	}
}
