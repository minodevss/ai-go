package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
)

const missingResultText = "error: this tool call has no result"

type textBlock struct {
	Type         string            `json:"type"`
	Text         string            `json:"text"`
	Citations    []json.RawMessage `json:"citations,omitzero"`
	CacheControl *wireCache        `json:"cache_control,omitzero"`
}

type toolUseBlock struct {
	Type         string          `json:"type"`
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Input        json.RawMessage `json:"input"`
	CacheControl *wireCache      `json:"cache_control,omitzero"`
}

type toolResultBlock struct {
	Type         string     `json:"type"`
	ToolUseID    string     `json:"tool_use_id"`
	Content      any        `json:"content,omitzero"`
	IsError      bool       `json:"is_error,omitzero"`
	CacheControl *wireCache `json:"cache_control,omitzero"`
}

type source struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitzero"`
	Data      string `json:"data,omitzero"`
	URL       string `json:"url,omitzero"`
	FileID    string `json:"file_id,omitzero"`
}

type imageBlock struct {
	Type         string     `json:"type"`
	Source       source     `json:"source"`
	CacheControl *wireCache `json:"cache_control,omitzero"`
}

type documentBlock struct {
	Type         string           `json:"type"`
	Source       source           `json:"source"`
	Title        string           `json:"title,omitzero"`
	Citations    *citationsConfig `json:"citations,omitzero"`
	CacheControl *wireCache       `json:"cache_control,omitzero"`
}

type input struct {
	messages  []claudewire.Message
	warnings  []ai.Warning
	citations bool
}

type inputBuilder struct {
	turns     claudewire.Turns[toolResultBlock]
	warnings  []ai.Warning
	citations bool
}

func buildInput(msgs []ai.Message) (input, error) {
	b := &inputBuilder{citations: anyCitations(msgs)}
	if b.citations && hasPlainDocument(msgs) {
		b.warn("citations are enabled on every document of the request because the API accepts all or none")
	}
	for _, m := range msgs {
		var err error
		switch m.Role {
		case ai.RoleUser:
			err = b.user(m)
		case ai.RoleAssistant:
			b.assistant(m)
		case ai.RoleTool:
			b.tool(m)
		default:
			err = fmt.Errorf("anthropic: unknown message role %q", m.Role)
		}
		if err != nil {
			return input{}, err
		}
	}
	return input{messages: b.finish(), warnings: b.warnings, citations: b.citations}, nil
}

func anyCitations(msgs []ai.Message) bool {
	for _, m := range msgs {
		if m.Role != ai.RoleUser {
			continue
		}
		for _, p := range m.Content {
			if f, ok := p.(ai.File); ok && f.Citations {
				return true
			}
		}
	}
	return false
}

func hasPlainDocument(msgs []ai.Message) bool {
	for _, m := range msgs {
		for _, p := range m.Content {
			if f, ok := p.(ai.File); ok && m.Role == ai.RoleUser && !f.Citations && isDocument(f) {
				return true
			}
		}
	}
	return false
}

func (b *inputBuilder) warn(format string, args ...any) {
	b.warnings = append(b.warnings, ai.Warning{Message: "anthropic: " + fmt.Sprintf(format, args...)})
}

func (b *inputBuilder) user(m ai.Message) error {
	var blocks []any
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			if v.Text != "" {
				blocks = append(blocks, textBlock{Type: "text", Text: v.Text, CacheControl: wireCacheOf(dataOf(v.Provider).CacheControl)})
			}
		case ai.File:
			blk, err := b.file(v)
			if err != nil {
				return err
			}
			blocks = append(blocks, blk)
		default:
		}
	}
	if len(blocks) > 0 {
		t := b.turns.For(claudewire.User)
		t.Blocks = append(t.Blocks, blocks...)
	}
	return nil
}

func (b *inputBuilder) tool(m ai.Message) {
	for _, p := range m.Content {
		if res, ok := p.(ai.ToolResult); ok {
			b.result(res)
		}
	}
}

func (b *inputBuilder) result(res ai.ToolResult) {
	if res.ProviderExecuted {
		b.foreignResult(res)
		return
	}
	blk := toolResultBlock{Type: "tool_result", ToolUseID: claudewire.CallID(res.CallID, validCallID), IsError: res.IsError, CacheControl: wireCacheOf(dataOf(res.Provider).CacheControl)}
	content, err := toolContent(res.Output)
	if err != nil {
		b.warn("tool result %q has content the API cannot take as tool output: %v", res.CallID, err)
		content = "error: unsupported tool output"
		blk.IsError = true
	}
	blk.Content = content
	t := b.turns.For(claudewire.User)
	t.Results = append(t.Results, blk)
}

func (b *inputBuilder) foreignResult(res ai.ToolResult) {
	if d := dataOf(res.Provider); len(d.Block) > 0 {
		t := b.turns.For(claudewire.Assistant)
		t.Blocks = append(t.Blocks, cachedBlock(d.Block, d.CacheControl))
		return
	}
	text := res.Output.Text
	if text == "" {
		text = string(res.Output.JSON)
	}
	t := b.turns.For(claudewire.Assistant)
	t.Blocks = append(t.Blocks, textBlock{Type: "text", Text: fmt.Sprintf("[%s result] %s", res.Name, text)})
}

func (b *inputBuilder) assistant(m ai.Message) {
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			b.text(v)
		case ai.Reasoning:
			if d := dataOf(v.Provider); len(d.Block) > 0 {
				b.add(d.Block)
			}
		case ai.ToolCall:
			b.call(v)
		case ai.ToolResult:
			b.result(v)
		case ai.Custom:
			if d := dataOf(v.Provider); len(d.Block) > 0 {
				b.add(cachedBlock(d.Block, d.CacheControl))
			}
		case ai.File:
			b.warn("an assistant file was dropped because the Messages API cannot take it as history")
		default:
		}
	}
}

func (b *inputBuilder) add(block any) {
	t := b.turns.For(claudewire.Assistant)
	t.Blocks = append(t.Blocks, block)
}

func (b *inputBuilder) text(t ai.Text) {
	blk := textBlock{Type: "text", Text: t.Text}
	if t.Text != "" {
		blk.CacheControl = wireCacheOf(dataOf(t.Provider).CacheControl)
	}
	for _, c := range t.Citations {
		if raw := dataOf(c.Provider).Block; len(raw) > 0 {
			blk.Citations = append(blk.Citations, raw)
		}
	}
	if blk.Text == "" && len(blk.Citations) == 0 {
		return
	}
	b.add(blk)
}

func (b *inputBuilder) call(c ai.ToolCall) {
	d := dataOf(c.Provider)
	if c.ProviderExecuted {
		if len(d.Block) > 0 {
			b.add(cachedBlock(d.Block, d.CacheControl))
		}
		return
	}
	id := claudewire.CallID(c.ID, validCallID)
	var block any = toolUseBlock{Type: "tool_use", ID: id, Name: c.Name, Input: claudewire.ObjectInput(c.Input), CacheControl: wireCacheOf(d.CacheControl)}
	if len(d.Block) > 0 {
		block = cachedBlock(d.Block, d.CacheControl)
	}
	t := b.turns.For(claudewire.Assistant)
	t.Blocks = append(t.Blocks, block)
	t.Calls = append(t.Calls, id)
}

func (b *inputBuilder) finish() []claudewire.Message {
	b.turns.Settle(
		func(r toolResultBlock) string { return r.ToolUseID },
		func(id string) toolResultBlock {
			return toolResultBlock{Type: "tool_result", ToolUseID: id, Content: missingResultText, IsError: true}
		},
		func(n int) { b.warn("%d tool result(s) without a matching tool call were dropped", n) },
	)
	return b.turns.Messages()
}

func toolContent(o ai.ToolOutput) (any, error) {
	switch {
	case len(o.Parts) > 0:
		blocks := make([]any, 0, len(o.Parts))
		for _, p := range o.Parts {
			switch v := p.(type) {
			case ai.Text:
				blocks = append(blocks, textBlock{Type: "text", Text: v.Text})
			case ai.File:
				blk, err := fileBlock(v, false)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, blk)
			default:
			}
		}
		return blocks, nil
	case len(o.JSON) > 0:
		return string(o.JSON), nil
	default:
		return o.Text, nil
	}
}

func (b *inputBuilder) file(f ai.File) (any, error) {
	return fileBlock(f, b.citations)
}

func isDocument(f ai.File) bool { return !strings.HasPrefix(f.MediaType, "image/") }

func fileBlock(f ai.File, citations bool) (any, error) {
	d := dataOf(f.Provider)
	cache := wireCacheOf(d.CacheControl)
	if strings.HasPrefix(f.MediaType, "image/") {
		src, err := fileSource(f, d)
		if err != nil {
			return nil, err
		}
		return imageBlock{Type: "image", Source: src, CacheControl: cache}, nil
	}
	doc := documentBlock{Type: "document", Title: f.Filename, CacheControl: cache}
	if citations {
		doc.Citations = &citationsConfig{Enabled: true}
	}
	switch {
	case d.FileID != "":
		doc.Source = source{Type: "file", FileID: d.FileID}
	case f.MediaType == "application/pdf":
		src, err := fileSource(f, d)
		if err != nil {
			return nil, err
		}
		doc.Source = src
	case strings.HasPrefix(f.MediaType, "text/") && len(f.Data) > 0:
		doc.Source = source{Type: "text", MediaType: "text/plain", Data: string(f.Data)}
	default:
		return nil, fmt.Errorf("anthropic: unsupported file media type %q: the Messages API takes images, PDFs and plain text", f.MediaType)
	}
	return doc, nil
}

func fileSource(f ai.File, d partData) (source, error) {
	switch {
	case d.FileID != "":
		return source{Type: "file", FileID: d.FileID}, nil
	case len(f.Data) > 0:
		return source{Type: "base64", MediaType: f.MediaType, Data: base64.StdEncoding.EncodeToString(f.Data)}, nil
	case f.URL != "":
		return source{Type: "url", URL: f.URL}, nil
	default:
		return source{}, errors.New("anthropic: file part has no data, url or file id")
	}
}
