package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
)

const (
	missingResultText = "error: this tool call has no result"
	maxDocumentName   = 200
)

type textBlock struct {
	Text string `json:"text"`
}

type cachePointBlock struct {
	CachePoint cachePointBody `json:"cachePoint"`
}

type cachePointBody struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitzero"`
}

type s3Location struct {
	URI string `json:"uri"`
}

type source struct {
	Bytes string      `json:"bytes,omitzero"`
	S3    *s3Location `json:"s3Location,omitzero"`
}

type imageBlock struct {
	Image struct {
		Format string `json:"format"`
		Source source `json:"source"`
	} `json:"image"`
}

type documentBlock struct {
	Document documentBody `json:"document"`
}

type documentBody struct {
	Name      string           `json:"name"`
	Format    string           `json:"format"`
	Source    source           `json:"source"`
	Citations *citationsConfig `json:"citations,omitzero"`
}

type citationsConfig struct {
	Enabled bool `json:"enabled"`
}

type toolUseBlock struct {
	ToolUse struct {
		ID    string          `json:"toolUseId"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"toolUse"`
}

type toolResultBlock struct {
	ToolResult toolResultBody `json:"toolResult"`
}

type toolResultBody struct {
	ToolUseID string `json:"toolUseId"`
	Content   []any  `json:"content"`
	Status    string `json:"status,omitzero"`
}

type toolResultJSON struct {
	JSON json.RawMessage `json:"json"`
}

type input struct {
	messages []claudewire.Message
	warnings []ai.Warning
	hasTools bool
}

type inputBuilder struct {
	model     string
	turns     claudewire.Turns[toolResultBlock]
	warnings  []ai.Warning
	documents int
}

func buildInput(model string, msgs []ai.Message) (input, error) {
	b := &inputBuilder{model: model}
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
			err = fmt.Errorf("bedrock: unknown message role %q", m.Role)
		}
		if err != nil {
			return input{}, err
		}
	}
	messages := b.finish()
	return input{messages: messages, warnings: b.warnings, hasTools: slices.ContainsFunc(b.turns, func(t *claudewire.Turn[toolResultBlock]) bool { return len(t.Calls) > 0 || len(t.Results) > 0 })}, nil
}

func (b *inputBuilder) warn(format string, args ...any) {
	b.warnings = append(b.warnings, ai.Warning{Message: "bedrock: " + fmt.Sprintf(format, args...)})
}

func (b *inputBuilder) user(m ai.Message) error {
	var blocks []any
	hasText, hasDocument := false, false
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			if v.Text != "" {
				hasText = true
				blocks = append(blocks, textBlock{Text: v.Text})
				blocks = appendCachePoint(blocks, dataOf(v.Provider).CachePoint)
			}
		case ai.File:
			blk, doc, err := b.file(v)
			if err != nil {
				return err
			}
			hasDocument = hasDocument || doc
			blocks = append(blocks, blk)
			blocks = appendCachePoint(blocks, dataOf(v.Provider).CachePoint)
		default:
		}
	}
	if hasDocument && !hasText {
		b.warn("a user message has a document but no text, which the Converse API rejects")
	}
	if len(blocks) > 0 {
		t := b.turns.For(claudewire.User)
		t.Blocks = append(t.Blocks, blocks...)
	}
	return nil
}

func appendCachePoint(blocks []any, c *cachePointData) []any {
	if c == nil {
		return blocks
	}
	return append(blocks, cachePointBlock{CachePoint: cachePointBody{Type: "default", TTL: string(c.TTL)}})
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
	body := toolResultBody{ToolUseID: claudewire.CallID(res.CallID, validCallID)}
	if res.IsError {
		body.Status = "error"
	}
	content, err := b.toolContent(res.Output)
	if err != nil {
		b.warn("tool result %q has content the API cannot take as tool output: %v", res.CallID, err)
		content = []any{textBlock{Text: "error: unsupported tool output"}}
		body.Status = "error"
	}
	body.Content = content
	t := b.turns.For(claudewire.User)
	t.Results = append(t.Results, toolResultBlock{ToolResult: body})
}

func (b *inputBuilder) foreignResult(res ai.ToolResult) {
	if raw := dataOf(res.Provider).Block; len(raw) > 0 {
		b.add(raw)
		return
	}
	text := res.Output.Text
	if text == "" {
		text = string(res.Output.JSON)
	}
	b.add(textBlock{Text: fmt.Sprintf("[%s result] %s", res.Name, text)})
}

func (b *inputBuilder) assistant(m ai.Message) {
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			b.text(v)
		case ai.Reasoning:
			if d := dataOf(v.Provider); len(d.Block) > 0 && d.Model == b.model && verifiable(d.Block) {
				b.add(d.Block)
			}
		case ai.ToolCall:
			b.call(v)
		case ai.ToolResult:
			b.result(v)
		case ai.Custom:
			if raw := dataOf(v.Provider).Block; len(raw) > 0 {
				b.add(raw)
			}
		case ai.File:
			b.warn("an assistant file was dropped because the Converse API cannot take it as history")
		default:
		}
	}
}

func verifiable(block json.RawMessage) bool {
	// Reasoning is replayed only with its signature or as redacted content: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningTextBlock.html
	var probe struct {
		Reasoning struct {
			Text struct {
				Signature string `json:"signature"`
			} `json:"reasoningText"`
			Redacted string `json:"redactedContent"`
		} `json:"reasoningContent"`
	}
	if err := json.Unmarshal(block, &probe); err != nil {
		return false
	}
	return probe.Reasoning.Text.Signature != "" || probe.Reasoning.Redacted != ""
}

func (b *inputBuilder) add(block any) {
	t := b.turns.For(claudewire.Assistant)
	t.Blocks = append(t.Blocks, block)
}

func (b *inputBuilder) text(t ai.Text) {
	if raw := dataOf(t.Provider).Block; len(raw) > 0 {
		b.add(raw)
		return
	}
	if t.Text != "" {
		b.add(textBlock{Text: t.Text})
	}
}

func (b *inputBuilder) call(c ai.ToolCall) {
	raw := dataOf(c.Provider).Block
	if c.ProviderExecuted {
		if len(raw) > 0 {
			b.add(raw)
		}
		return
	}
	id := claudewire.CallID(c.ID, validCallID)
	var block any = newToolUse(id, c.Name, claudewire.ObjectInput(c.Input))
	if len(raw) > 0 {
		block = raw
	}
	t := b.turns.For(claudewire.Assistant)
	t.Blocks = append(t.Blocks, block)
	t.Calls = append(t.Calls, id)
}

func newToolUse(id, name string, in json.RawMessage) toolUseBlock {
	var blk toolUseBlock
	blk.ToolUse.ID, blk.ToolUse.Name, blk.ToolUse.Input = id, name, in
	return blk
}

func (b *inputBuilder) finish() []claudewire.Message {
	b.turns.Settle(
		func(r toolResultBlock) string { return r.ToolResult.ToolUseID },
		func(id string) toolResultBlock {
			return toolResultBlock{ToolResult: toolResultBody{ToolUseID: id, Content: []any{textBlock{Text: missingResultText}}, Status: "error"}}
		},
		func(n int) { b.warn("%d tool result(s) without a matching tool call were dropped", n) },
	)
	return b.turns.Messages()
}

func (b *inputBuilder) toolContent(o ai.ToolOutput) ([]any, error) {
	switch {
	case len(o.Parts) > 0:
		blocks := make([]any, 0, len(o.Parts))
		for _, p := range o.Parts {
			switch v := p.(type) {
			case ai.Text:
				blocks = append(blocks, textBlock{Text: v.Text})
			case ai.File:
				blk, _, err := b.file(v)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, blk)
			default:
			}
		}
		return blocks, nil
	case len(o.JSON) > 0:
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(o.JSON, &obj); err == nil && obj != nil {
			return []any{toolResultJSON{JSON: o.JSON}}, nil
		}
		return []any{textBlock{Text: string(o.JSON)}}, nil
	default:
		return []any{textBlock{Text: o.Text}}, nil
	}
}

func imageFormat(mediaType string) (string, bool) {
	switch mediaType {
	case "image/png":
		return "png", true
	case "image/jpeg", "image/jpg":
		return "jpeg", true
	case "image/gif":
		return "gif", true
	case "image/webp":
		return "webp", true
	default:
		return "", false
	}
}

func documentFormat(mediaType string) (string, bool) {
	switch mediaType {
	case "application/pdf":
		return "pdf", true
	case "text/csv":
		return "csv", true
	case "application/msword":
		return "doc", true
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return "docx", true
	case "application/vnd.ms-excel":
		return "xls", true
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return "xlsx", true
	case "text/html":
		return "html", true
	case "text/plain":
		return "txt", true
	case "text/markdown":
		return "md", true
	default:
		return "", false
	}
}

func (b *inputBuilder) file(f ai.File) (any, bool, error) {
	src, err := fileSource(f)
	if err != nil {
		return nil, false, err
	}
	if format, ok := imageFormat(f.MediaType); ok {
		var blk imageBlock
		blk.Image.Format, blk.Image.Source = format, src
		return blk, false, nil
	}
	format, ok := documentFormat(f.MediaType)
	if !ok {
		return nil, false, fmt.Errorf("bedrock: unsupported file media type %q: Converse takes images (png, jpeg, gif, webp) and documents (pdf, csv, doc, docx, xls, xlsx, html, txt, md)", f.MediaType)
	}
	b.documents++
	doc := documentBody{Name: b.documentName(f.Filename), Format: format, Source: src}
	if f.Citations {
		doc.Citations = &citationsConfig{Enabled: true}
	}
	return documentBlock{Document: doc}, true, nil
}

func (b *inputBuilder) documentName(filename string) string {
	// Document names allow only alphanumerics, single spaces, hyphens, parentheses and square brackets: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
	cleaned := strings.Join(strings.Fields(strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("-()[]", r) {
			return r
		}
		return ' '
	}, filename)), " ")
	if len(cleaned) > maxDocumentName {
		cleaned = cleaned[:maxDocumentName]
	}
	if cleaned == "" {
		return fmt.Sprintf("document %d", b.documents)
	}
	if cleaned != filename {
		b.warn("document name %q was changed to %q because names allow only letters, digits, single spaces, hyphens, parentheses and square brackets", filename, cleaned)
	}
	return cleaned
}

func fileSource(f ai.File) (source, error) {
	switch {
	case len(f.Data) > 0:
		return source{Bytes: base64.StdEncoding.EncodeToString(f.Data)}, nil
	case strings.HasPrefix(f.URL, "s3://"):
		return source{S3: &s3Location{URI: f.URL}}, nil
	case f.URL != "":
		return source{}, fmt.Errorf("bedrock: file url %q is not an s3:// location: Converse takes inline bytes or S3 objects", f.URL)
	default:
		return source{}, errors.New("bedrock: file part has no data or s3 url")
	}
}
