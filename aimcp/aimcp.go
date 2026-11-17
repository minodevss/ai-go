// Package aimcp exposes the tools of an MCP server as ai.Tool values that ask for approval and are marked untrusted by default.
package aimcp

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/minodevss/ai-go"
)

const (
	defaultMaxTools            = 128
	defaultMaxDescriptionBytes = 4 << 10
	defaultMaxSchemaBytes      = 32 << 10
	defaultMaxOutputBytes      = 1 << 20
	maxNameBytes               = 64
)

var (
	ErrNoPrefix  = errors.New("aimcp: a tool name prefix is required")
	ErrTooLarge  = errors.New("aimcp: tool definition or output exceeds its limit")
	ErrBadName   = errors.New("aimcp: tool name cannot be exposed")
	errNoSession = errors.New("aimcp: nil session")
)

// Limits caps what a server may hand to the model; a zero field keeps its default (128 tools, 4 KiB description, 32 KiB schema, 1 MiB output). A tool over a definition limit fails Tools instead of being silently dropped.
type Limits struct {
	MaxTools            int
	MaxDescriptionBytes int
	MaxSchemaBytes      int
	MaxOutputBytes      int
}

type options struct {
	prefix string
	filter func(*mcp.Tool) bool
	trust  bool
	local  map[string]struct{}
	limits Limits
	pins   *Pins
	fresh  bool
}

type Option func(*options)

// Prefix is put in front of every tool name and must not be empty; the default is the server's initialize name followed by an underscore.
func Prefix(prefix string) Option {
	return func(o *options) { o.prefix = prefix }
}

// Filter keeps the tools for which keep returns true.
func Filter(keep func(*mcp.Tool) bool) Option {
	return func(o *options) { o.filter = keep }
}

// TrustHints lets the server's annotations relax the defaults: a read-only tool is allowed without asking, and a tool with openWorldHint set to false is no longer untrusted. Annotations are hints from the server, so use this only for servers you trust.
func TrustHints() Option {
	return func(o *options) { o.trust = true }
}

// LocalTools lists tools that win a name clash: an MCP tool whose exposed name equals one of theirs is dropped.
func LocalTools(tools ...ai.Tool) Option {
	return func(o *options) {
		o.local = make(map[string]struct{}, len(tools))
		for _, t := range tools {
			o.local[t.Name] = struct{}{}
		}
	}
}

func WithLimits(l Limits) Option {
	return func(o *options) { o.limits = l }
}

// WithPins freezes tool definitions: the first Tools call per session stores them in p and later calls return them without asking the server again.
func WithPins(p *Pins) Option {
	return func(o *options) { o.pins = p }
}

// Refresh makes Tools list the server again and replace the definitions pinned with WithPins.
func Refresh() Option {
	return func(o *options) { o.fresh = true }
}

// Pins holds pinned tool definitions per session. The zero value is ready to use.
type Pins struct {
	mu   sync.Mutex
	sets map[*mcp.ClientSession][]ai.Tool
}

// Tools lists the server's tools. Each call without WithPins reads the current definitions from the server, so a changed description reaches the model; keep the result, or use WithPins, to hold definitions fixed.
func Tools(ctx context.Context, session *mcp.ClientSession, opts ...Option) ([]ai.Tool, error) {
	if session == nil {
		return nil, errNoSession
	}
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	o.limits = o.limits.withDefaults()
	if o.pins == nil {
		return list(ctx, session, o)
	}
	o.pins.mu.Lock()
	defer o.pins.mu.Unlock()
	if pinned, ok := o.pins.sets[session]; ok && !o.fresh {
		return slices.Clone(pinned), nil
	}
	tools, err := list(ctx, session, o)
	if err != nil {
		return nil, err
	}
	if o.pins.sets == nil {
		o.pins.sets = map[*mcp.ClientSession][]ai.Tool{}
	}
	o.pins.sets[session] = slices.Clone(tools)
	return tools, nil
}

func (l Limits) withDefaults() Limits {
	l.MaxTools = cmp.Or(l.MaxTools, defaultMaxTools)
	l.MaxDescriptionBytes = cmp.Or(l.MaxDescriptionBytes, defaultMaxDescriptionBytes)
	l.MaxSchemaBytes = cmp.Or(l.MaxSchemaBytes, defaultMaxSchemaBytes)
	l.MaxOutputBytes = cmp.Or(l.MaxOutputBytes, defaultMaxOutputBytes)
	return l
}

func list(ctx context.Context, session *mcp.ClientSession, o options) ([]ai.Tool, error) {
	prefix, err := resolvePrefix(session, o.prefix)
	if err != nil {
		return nil, err
	}
	var tools []ai.Tool
	seen := map[string]string{}
	count := 0
	for t, err := range session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("aimcp: list tools: %w", err)
		}
		count++
		if count > o.limits.MaxTools {
			return nil, fmt.Errorf("%w: more than %d tools", ErrTooLarge, o.limits.MaxTools)
		}
		if o.filter != nil && !o.filter(t) {
			continue
		}
		tool, err := convert(session, t, prefix, o)
		if err != nil {
			return nil, err
		}
		if _, local := o.local[tool.Name]; local {
			continue
		}
		if first, dup := seen[tool.Name]; dup {
			return nil, fmt.Errorf("%w: %q and %q both become %q", ai.ErrDuplicateTool, first, t.Name, tool.Name)
		}
		seen[tool.Name] = t.Name
		tools = append(tools, tool)
	}
	return tools, nil
}

func resolvePrefix(session *mcp.ClientSession, prefix string) (string, error) {
	if prefix != "" {
		return prefix, nil
	}
	if init := session.InitializeResult(); init != nil && init.ServerInfo != nil {
		if name := sanitize(init.ServerInfo.Name); strings.Trim(name, "_") != "" {
			return name + "_", nil
		}
	}
	return "", ErrNoPrefix
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-':
			return r
		case r >= 'A' && r <= 'Z':
			return r + 'a' - 'A'
		default:
			return '_'
		}
	}, s)
}

func exposedName(prefix, name string) string {
	return prefix + strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			return r
		default:
			return '_'
		}
	}, name)
}

func convert(session *mcp.ClientSession, t *mcp.Tool, prefix string, o options) (ai.Tool, error) {
	name := exposedName(prefix, t.Name)
	if len(name) > maxNameBytes {
		return ai.Tool{}, fmt.Errorf("%w: %q is longer than %d bytes with its prefix", ErrBadName, name, maxNameBytes)
	}
	if len(t.Description) > o.limits.MaxDescriptionBytes {
		return ai.Tool{}, fmt.Errorf("%w: description of %q has %d bytes, limit %d", ErrTooLarge, t.Name, len(t.Description), o.limits.MaxDescriptionBytes)
	}
	schema, err := inputSchema(t)
	if err != nil {
		return ai.Tool{}, err
	}
	if len(schema) > o.limits.MaxSchemaBytes {
		return ai.Tool{}, fmt.Errorf("%w: input schema of %q has %d bytes, limit %d", ErrTooLarge, t.Name, len(schema), o.limits.MaxSchemaBytes)
	}
	tool := ai.Tool{
		Name:        name,
		Description: t.Description,
		InputSchema: schema,
		Approval:    ai.AlwaysAsk,
		Untrusted:   true,
		Execute:     executor(session, t.Name, o.limits.MaxOutputBytes),
	}
	if o.trust && t.Annotations != nil {
		if t.Annotations.ReadOnlyHint {
			tool.Approval = func(context.Context, ai.ApprovalInput) ai.Approval { return ai.Allow() }
		}
		if t.Annotations.OpenWorldHint != nil && !*t.Annotations.OpenWorldHint {
			tool.Untrusted = false
		}
	}
	return tool, nil
}

func inputSchema(t *mcp.Tool) (json.RawMessage, error) {
	if t.InputSchema == nil {
		return json.RawMessage(`{"type":"object"}`), nil
	}
	raw, err := json.Marshal(t.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("aimcp: encode input schema of %q: %w", t.Name, err)
	}
	return raw, nil
}

func executor(session *mcp.ClientSession, name string, maxOutput int) func(context.Context, json.RawMessage) (ai.ToolOutput, error) {
	return func(ctx context.Context, input json.RawMessage) (ai.ToolOutput, error) {
		params := &mcp.CallToolParams{Name: name}
		if len(strings.TrimSpace(string(input))) > 0 {
			params.Arguments = input
		}
		res, err := session.CallTool(ctx, params)
		if err != nil {
			return ai.ToolOutput{}, fmt.Errorf("aimcp: call tool %q: %w", name, err)
		}
		return output(res, maxOutput)
	}
}

func output(res *mcp.CallToolResult, maxBytes int) (ai.ToolOutput, error) {
	parts, size := convertContent(res.Content)
	if size > maxBytes {
		return ai.ToolOutput{}, ai.ToolErrorf("aimcp: tool output has %d bytes, limit %d", size, maxBytes)
	}
	if res.IsError {
		return ai.ToolOutput{}, ai.ToolErrorf("%s", errorText(parts))
	}
	if len(parts) == 0 && res.StructuredContent != nil {
		raw, err := json.Marshal(res.StructuredContent)
		if err != nil {
			return ai.ToolOutput{}, fmt.Errorf("aimcp: encode structured content: %w", err)
		}
		if len(raw) > maxBytes {
			return ai.ToolOutput{}, ai.ToolErrorf("aimcp: tool output has %d bytes, limit %d", len(raw), maxBytes)
		}
		return ai.ToolOutput{JSON: raw}, nil
	}
	if len(parts) == 1 {
		if text, ok := parts[0].(ai.Text); ok {
			return ai.ToolOutput{Text: text.Text}, nil
		}
	}
	if len(parts) == 0 {
		return ai.ToolOutput{Text: ""}, nil
	}
	return ai.ToolOutput{Parts: parts}, nil
}

func convertContent(content []mcp.Content) (ai.Parts, int) {
	var parts ai.Parts
	size := 0
	for _, c := range content {
		switch v := c.(type) {
		case *mcp.TextContent:
			parts = append(parts, ai.Text{Text: v.Text})
			size += len(v.Text)
		case *mcp.ImageContent:
			parts = append(parts, ai.File{MediaType: v.MIMEType, Data: v.Data})
			size += len(v.Data)
		case *mcp.AudioContent:
			parts = append(parts, ai.File{MediaType: v.MIMEType, Data: v.Data})
			size += len(v.Data)
		case *mcp.ResourceLink:
			text := "resource link: " + v.URI
			parts = append(parts, ai.Text{Text: text})
			size += len(text)
		case *mcp.EmbeddedResource:
			if v.Resource == nil {
				continue
			}
			if v.Resource.Blob != nil {
				parts = append(parts, ai.File{MediaType: v.Resource.MIMEType, Data: v.Resource.Blob})
				size += len(v.Resource.Blob)
				continue
			}
			parts = append(parts, ai.Text{Text: v.Resource.Text})
			size += len(v.Resource.Text)
		default:
		}
	}
	return parts, size
}

func errorText(parts ai.Parts) string {
	var texts []string
	for _, p := range parts {
		if t, ok := p.(ai.Text); ok {
			texts = append(texts, t.Text)
		}
	}
	if len(texts) == 0 {
		return "tool reported an error"
	}
	return strings.Join(texts, "\n")
}
