package ui

import (
	"encoding/base64"
	"errors"
	"fmt"
	"mime"
	"net/url"
	"path"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/toolmsg"
)

// FilePolicy decides which client files reach the model. The zero value accepts inline data URLs and rejects every remote URL.
type FilePolicy struct {
	MediaTypes []string
	MaxBytes   int
	AllowURL   func(u *url.URL) bool
}

func trustedFiles() FilePolicy {
	return FilePolicy{AllowURL: func(*url.URL) bool { return true }}
}

func (p FilePolicy) check(f FilePart) (ai.File, error) {
	if len(f.Reference) > 0 {
		return ai.File{}, requestError(400, ErrFile, "provider file references are not accepted")
	}
	if err := validMediaType(f.MediaType); err != nil {
		return ai.File{}, err
	}
	out := ai.File{MediaType: f.MediaType, Filename: f.Filename, Citations: f.Citable}
	if strings.HasPrefix(f.URL, "data:") {
		mediaType, data, err := parseDataURL(f.URL)
		if err != nil {
			return ai.File{}, requestError(400, ErrFile, "%v", err)
		}
		if err := validMediaType(mediaType); err != nil {
			return ai.File{}, err
		}
		if out.MediaType == "" {
			out.MediaType = mediaType
		}
		if mediaType != "" && !p.mediaTypeAllowed(mediaType) {
			return ai.File{}, requestError(400, ErrFile, "media type %q is not accepted", mediaType)
		}
		if p.MaxBytes > 0 && len(data) > p.MaxBytes {
			return ai.File{}, requestError(413, ErrFile, "file exceeds %d bytes", p.MaxBytes)
		}
		out.Data = data
	} else {
		u, err := url.Parse(f.URL)
		if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") {
			return ai.File{}, requestError(400, ErrFile, "unsupported file url")
		}
		if p.AllowURL == nil || !p.AllowURL(u) {
			return ai.File{}, requestError(400, ErrFile, "file urls are not accepted")
		}
		out.URL = f.URL
	}
	if !p.mediaTypeAllowed(out.MediaType) {
		return ai.File{}, requestError(400, ErrFile, "media type %q is not accepted", out.MediaType)
	}
	return out, nil
}

func validMediaType(mediaType string) error {
	if mediaType == "" {
		return nil
	}
	if strings.ContainsFunc(mediaType, func(r rune) bool { return r < ' ' || r == 0x7f }) {
		return requestError(400, ErrFile, "media type has control characters")
	}
	if _, _, err := mime.ParseMediaType(mediaType); err != nil {
		return requestError(400, ErrFile, "media type %q is invalid", mediaType)
	}
	return nil
}

func (p FilePolicy) mediaTypeAllowed(mediaType string) bool {
	if len(p.MediaTypes) == 0 {
		return true
	}
	return slices.ContainsFunc(p.MediaTypes, func(pattern string) bool {
		ok, err := path.Match(pattern, mediaType)
		return err == nil && ok
	})
}

func dataURL(mediaType string, data []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func parseDataURL(s string) (string, []byte, error) {
	meta, payload, ok := strings.Cut(strings.TrimPrefix(s, "data:"), ",")
	if !ok {
		return "", nil, errors.New("ui: malformed data url")
	}
	mediaType, params, _ := strings.Cut(meta, ";")
	if strings.HasSuffix(";"+params, ";base64") {
		data, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return "", nil, fmt.Errorf("ui: malformed data url: %w", err)
		}
		return mediaType, data, nil
	}
	text, err := url.PathUnescape(payload)
	if err != nil {
		return "", nil, fmt.Errorf("ui: malformed data url: %w", err)
	}
	return mediaType, []byte(text), nil
}

type builder struct {
	out []Message
	cur int
	n   int
}

func FromMessages(msgs []ai.Message) []Message {
	b := &builder{cur: -1}
	for _, m := range msgs {
		switch m.Role {
		case ai.RoleUser:
			b.cur = -1
			b.add(Message{Role: RoleUser, Parts: userParts(m.Content)})
		case ai.RoleAssistant:
			b.group().Parts = append(b.group().Parts, StepStartPart{Provider: cloneData(m.Provider)})
			for _, p := range m.Content {
				b.assistantPart(p)
			}
		case ai.RoleTool:
			b.group()
			for _, p := range m.Content {
				b.toolPart(p)
			}
		}
	}
	return b.out
}

func (b *builder) add(m Message) {
	b.n++
	m.ID = fmt.Sprintf("msg-%d", b.n)
	b.out = append(b.out, m)
}

func (b *builder) group() *Message {
	if b.cur < 0 {
		b.add(Message{Role: RoleAssistant})
		b.cur = len(b.out) - 1
	}
	return &b.out[b.cur]
}

func userParts(parts ai.Parts) []Part {
	out := make([]Part, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, TextPart{Text: v.Text, State: StateDone, Provider: v.Provider})
		case ai.File:
			out = append(out, fileToPart(v))
		default:
		}
	}
	return out
}

func fileToPart(f ai.File) FilePart {
	u := f.URL
	if len(f.Data) > 0 {
		u = dataURL(f.MediaType, f.Data)
	}
	return FilePart{MediaType: f.MediaType, Filename: f.Filename, URL: u, Citable: f.Citations, Provider: f.Provider}
}

func (b *builder) assistantPart(p ai.Part) {
	g := b.group()
	switch v := p.(type) {
	case ai.Text:
		g.Parts = append(g.Parts, TextPart{Text: v.Text, State: StateDone, Citations: slices.Clone(v.Citations), Provider: v.Provider})
	case ai.Reasoning:
		g.Parts = append(g.Parts, ReasoningPart{Text: v.Text, State: StateDone, Provider: v.Provider})
	case ai.File:
		g.Parts = append(g.Parts, fileToPart(v))
	case ai.Custom:
		g.Parts = append(g.Parts, CustomPart{Kind: v.Kind, Data: v.Data, Provider: v.Provider})
	case ai.ToolCall:
		g.Parts = append(g.Parts, ToolPart{Name: v.Name, CallID: v.ID, State: ToolInputAvailable, Input: v.Input, ProviderExecuted: v.ProviderExecuted, CallProvider: v.Provider})
	default:
		b.toolPart(p)
	}
}

func (b *builder) tool(callID, name string) *ToolPart {
	g := b.group()
	for _, p := range slices.Backward(g.Parts) {
		if t, ok := p.(ToolPart); ok && t.CallID == callID {
			return &t
		}
	}
	return &ToolPart{Name: name, CallID: callID, State: ToolInputAvailable}
}

func (b *builder) put(t *ToolPart) {
	g := b.group()
	for i := len(g.Parts) - 1; i >= 0; i-- {
		if old, ok := g.Parts[i].(ToolPart); ok && old.CallID == t.CallID {
			g.Parts[i] = *t
			return
		}
	}
	g.Parts = append(g.Parts, *t)
}

func (b *builder) toolPart(p ai.Part) {
	switch v := p.(type) {
	case ai.ToolApprovalRequest:
		t := b.tool(v.CallID, "")
		t.Approval = &Approval{ID: v.ApprovalID, RequestReason: v.Reason, InputHash: v.InputHash}
		if t.State == ToolInputAvailable {
			t.State = ToolApprovalRequested
		}
		b.put(t)
	case ai.ToolApprovalResponse:
		t := b.tool(v.CallID, "")
		if t.Approval == nil {
			t.Approval = &Approval{ID: v.ApprovalID}
		}
		approved := v.Approved
		t.Approval.Approved, t.Approval.Reason = &approved, v.Reason
		if t.State == ToolApprovalRequested || t.State == ToolInputAvailable {
			t.State = ToolApprovalResponded
		}
		b.put(t)
	case ai.ToolResult:
		t := b.tool(v.CallID, v.Name)
		if t.Name == "" {
			t.Name = v.Name
		}
		t.Output, t.ResultProvider = v.Output, v.Provider
		t.ProviderExecuted = t.ProviderExecuted || v.ProviderExecuted
		switch {
		case !v.IsError:
			t.State = ToolOutputAvailable
		case t.Approval != nil && t.Approval.Approved != nil && !*t.Approval.Approved:
			t.State = ToolOutputDenied
		default:
			t.State = ToolOutputError
		}
		if v.IsError {
			t.ErrorText = outputText(v.Output)
		}
		b.put(t)
	default:
	}
}

func outputText(o ai.ToolOutput) string {
	if o.Text != "" {
		return o.Text
	}
	return string(o.JSON)
}

// ToAI converts trusted UI messages, such as ones an application stored itself, back to model messages. Client input goes through Merge or ToMessages instead.
func ToAI(msgs []Message) ([]ai.Message, error) {
	var out []ai.Message
	for _, m := range msgs {
		switch m.Role {
		case RoleUser:
			um, err := userToAI(m, trustedFiles(), true)
			if err != nil {
				return nil, err
			}
			out = append(out, um)
		case RoleAssistant:
			out = append(out, assistantToAI(m.Parts)...)
		case RoleSystem:
		}
	}
	return out, nil
}

func userToAI(m Message, policy FilePolicy, keepProvider bool) (ai.Message, error) {
	content := make(ai.Parts, 0, len(m.Parts))
	for _, p := range m.Parts {
		switch v := p.(type) {
		case TextPart:
			t := ai.Text{Text: v.Text}
			if keepProvider {
				t.Provider = cloneData(v.Provider)
			}
			content = append(content, t)
		case FilePart:
			f, err := policy.check(v)
			if err != nil {
				return ai.Message{}, err
			}
			if keepProvider {
				f.Provider = cloneData(v.Provider)
			}
			content = append(content, f)
		default:
		}
	}
	return ai.Message{Role: ai.RoleUser, Content: content}, nil
}

type block struct {
	assistant ai.Message
	requests  []ai.Part
	responses []ai.Part
	results   []ai.Part
}

func assistantToAI(parts []Part) []ai.Message {
	var blocks []*block
	cur := &block{assistant: ai.Message{Role: ai.RoleAssistant}}
	started := false
	flush := func() {
		if started || len(cur.assistant.Content) > 0 || len(cur.results) > 0 {
			blocks = append(blocks, cur)
		}
	}
	for _, p := range parts {
		if sp, ok := p.(StepStartPart); ok {
			flush()
			cur = &block{assistant: ai.Message{Role: ai.RoleAssistant, Provider: cloneData(sp.Provider)}}
			started = true
			continue
		}
		cur.add(p)
	}
	flush()
	var out []ai.Message
	for _, b := range blocks {
		if len(b.assistant.Content) > 0 || len(b.requests) > 0 {
			b.assistant.Content = append(b.assistant.Content, b.requests...)
			out = append(out, b.assistant)
		}
		if tool := slices.Concat(b.responses, b.results); len(tool) > 0 {
			out = append(out, ai.Message{Role: ai.RoleTool, Content: tool})
		}
	}
	return out
}

func (b *block) add(p Part) {
	switch v := p.(type) {
	case TextPart:
		b.assistant.Content = append(b.assistant.Content, ai.Text{Text: v.Text, Citations: slices.Clone(v.Citations), Provider: cloneData(v.Provider)})
	case ReasoningPart:
		b.assistant.Content = append(b.assistant.Content, ai.Reasoning{Text: v.Text, Provider: cloneData(v.Provider)})
	case FilePart:
		if f, err := trustedFiles().check(v); err == nil {
			f.Provider = cloneData(v.Provider)
			b.assistant.Content = append(b.assistant.Content, f)
		}
	case CustomPart:
		b.assistant.Content = append(b.assistant.Content, ai.Custom{Kind: v.Kind, Data: v.Data, Provider: cloneData(v.Provider)})
	case ToolPart:
		b.tool(v)
	default:
	}
}

func (b *block) tool(t ToolPart) {
	if t.State == ToolInputStreaming {
		return
	}
	b.assistant.Content = append(b.assistant.Content, ai.ToolCall{ID: t.CallID, Name: t.Name, Input: t.Input, ProviderExecuted: t.ProviderExecuted, Provider: cloneData(t.CallProvider)})
	if t.Approval != nil && t.Approval.ID != "" {
		b.requests = append(b.requests, ai.ToolApprovalRequest{ApprovalID: t.Approval.ID, CallID: t.CallID, Reason: t.Approval.RequestReason, InputHash: t.Approval.InputHash})
		if t.Approval.Approved != nil {
			b.responses = append(b.responses, ai.ToolApprovalResponse{ApprovalID: t.Approval.ID, CallID: t.CallID, Approved: *t.Approval.Approved, Reason: t.Approval.Reason})
		}
	}
	res, ok := t.result()
	if !ok {
		return
	}
	if t.ProviderExecuted {
		b.assistant.Content = append(b.assistant.Content, res)
		return
	}
	b.results = append(b.results, res)
}

func (p ToolPart) result() (ai.ToolResult, bool) {
	var isErr bool
	switch p.State {
	case ToolOutputAvailable:
	case ToolOutputError, ToolOutputDenied:
		isErr = true
	default:
		return ai.ToolResult{}, false
	}
	out := p.Output
	if isErr && out.Text == "" && len(out.JSON) == 0 && len(out.Parts) == 0 {
		out = ai.ToolOutput{Text: p.ErrorText}
		if p.State == ToolOutputDenied && out.Text == "" {
			out.Text = toolmsg.Denied("")
		}
	}
	return ai.ToolResult{CallID: p.CallID, Name: p.Name, Output: out, IsError: isErr, ProviderExecuted: p.ProviderExecuted, Provider: cloneData(p.ResultProvider)}, true
}

func cloneData(d ai.ProviderData) ai.ProviderData {
	if len(d) == 0 {
		return nil
	}
	out := make(ai.ProviderData, len(d))
	for k, v := range d {
		out[k] = slices.Clone(v)
	}
	return out
}

// WithoutProvider returns a copy of m with all provider data removed, for messages that are sent to a client.
func (m *Message) WithoutProvider() Message {
	out := *m
	out.Parts = make([]Part, len(m.Parts))
	for i, p := range m.Parts {
		out.Parts[i] = withoutProvider(p)
	}
	return out
}

func withoutProvider(p Part) Part {
	switch v := p.(type) {
	case TextPart:
		v.Provider = nil
		v.Citations = stripCitations(v.Citations)
		return v
	case ReasoningPart:
		v.Provider = nil
		return v
	case FilePart:
		v.Provider = nil
		return v
	case ToolPart:
		v.CallProvider, v.ResultProvider = nil, nil
		return v
	case SourceURLPart:
		v.Provider = nil
		return v
	case SourceDocumentPart:
		v.Provider = nil
		return v
	case StepStartPart:
		return StepStartPart{}
	case CustomPart:
		v.Provider = nil
		return v
	default:
		return p
	}
}
