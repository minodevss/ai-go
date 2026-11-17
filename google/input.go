package google

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
)

const missingResultText = "error: this tool call has no result"

type content struct {
	Role  string            `json:"role,omitzero"`
	Parts []json.RawMessage `json:"parts"`
}

type partOut struct {
	Text             *string      `json:"text,omitzero"`
	InlineData       *blobOut     `json:"inlineData,omitzero"`
	FileData         *fileRefOut  `json:"fileData,omitzero"`
	FunctionCall     *callOut     `json:"functionCall,omitzero"`
	FunctionResponse *responseOut `json:"functionResponse,omitzero"`
	ThoughtSignature string       `json:"thoughtSignature,omitzero"`
}

type blobOut struct {
	DisplayName string `json:"displayName,omitzero"`
	MimeType    string `json:"mimeType"`
	Data        string `json:"data"`
}

type fileRefOut struct {
	MimeType string `json:"mimeType,omitzero"`
	FileURI  string `json:"fileUri"`
}

type callOut struct {
	ID   string          `json:"id,omitzero"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type responseOut struct {
	ID       string            `json:"id,omitzero"`
	Name     string            `json:"name"`
	Response json.RawMessage   `json:"response"`
	Parts    []json.RawMessage `json:"parts,omitzero"`
}

type pendingCall struct {
	aiID   string
	sendID string
	name   string
}

type contentBuilder struct {
	b        *backend
	dummy    bool
	contents []content
	pending  []pendingCall
	results  map[string]ai.ToolResult
	warnings []ai.Warning
	signed   bool
}

func (b *backend) buildContents(model string, msgs []ai.Message) ([]content, []ai.Warning, error) {
	cb := &contentBuilder{b: b, dummy: needsDummySignature(model), results: map[string]ai.ToolResult{}}
	for _, m := range msgs {
		var err error
		switch m.Role {
		case ai.RoleUser:
			err = cb.user(m)
		case ai.RoleAssistant:
			err = cb.assistant(m)
		case ai.RoleTool:
			err = cb.tool(m)
		default:
			err = fmt.Errorf("google: unknown message role %q", m.Role)
		}
		if err != nil {
			return nil, nil, err
		}
	}
	if err := cb.flush(); err != nil {
		return nil, nil, err
	}
	return cb.contents, cb.warnings, nil
}

func needsDummySignature(model string) bool {
	return strings.HasPrefix(model, "gemini-3") || strings.HasPrefix(model, "gemini-4")
}

func (cb *contentBuilder) warn(format string, args ...any) {
	cb.warnings = append(cb.warnings, ai.Warning{Message: fmt.Sprintf("google: %s", fmt.Sprintf(format, args...))})
}

func (cb *contentBuilder) add(role string, parts []json.RawMessage) {
	if len(parts) > 0 {
		cb.contents = append(cb.contents, content{Role: role, Parts: parts})
	}
}

func (cb *contentBuilder) user(m ai.Message) error {
	if err := cb.flush(); err != nil {
		return err
	}
	var parts []json.RawMessage
	for _, p := range m.Content {
		switch v := p.(type) {
		case ai.Text:
			if v.Text != "" {
				parts = append(parts, pdata.JSON(providerName, partOut{Text: &v.Text}))
			}
		case ai.File:
			raw, err := filePart(v)
			if err != nil {
				return err
			}
			parts = append(parts, raw)
		default:
		}
	}
	cb.add("user", parts)
	return nil
}

func (cb *contentBuilder) tool(m ai.Message) error {
	var foreign []json.RawMessage
	for _, p := range m.Content {
		res, ok := p.(ai.ToolResult)
		if !ok {
			continue
		}
		if res.ProviderExecuted {
			foreign = append(foreign, pdata.JSON(providerName, foreignResultPart(res)))
			continue
		}
		cb.results[res.CallID] = res
	}
	if len(foreign) > 0 {
		if err := cb.flush(); err != nil {
			return err
		}
		cb.add("model", foreign)
	}
	return nil
}

func (cb *contentBuilder) assistant(m ai.Message) error {
	if err := cb.flush(); err != nil {
		return err
	}
	cb.signed = false
	var parts []json.RawMessage
	for _, p := range m.Content {
		raw, err := cb.assistantPart(p)
		if err != nil {
			return err
		}
		if raw != nil {
			parts = append(parts, raw)
		}
	}
	cb.add("model", parts)
	return nil
}

func (cb *contentBuilder) assistantPart(p ai.Part) (json.RawMessage, error) {
	switch v := p.(type) {
	case ai.Text:
		return cb.textPart(v), nil
	case ai.Reasoning:
		return cb.b.decode(v.Provider).Part, nil
	case ai.ToolCall:
		return cb.callPart(v), nil
	case ai.ToolResult:
		return cb.resultPart(v), nil
	case ai.File:
		if d := cb.b.decode(v.Provider); len(d.Part) > 0 {
			return d.Part, nil
		}
		return filePart(v)
	case ai.Custom:
		return cb.b.decode(v.Provider).Part, nil
	default:
		return nil, nil
	}
}

func (cb *contentBuilder) textPart(t ai.Text) json.RawMessage {
	if d := cb.b.decode(t.Provider); len(d.Part) > 0 {
		if isEmptyUnsigned(d.Part) {
			return nil
		}
		return d.Part
	}
	if t.Text == "" {
		return nil
	}
	return pdata.JSON(providerName, partOut{Text: &t.Text})
}

func (cb *contentBuilder) resultPart(res ai.ToolResult) json.RawMessage {
	if !res.ProviderExecuted {
		cb.results[res.CallID] = res
		return nil
	}
	if d := cb.b.decode(res.Provider); len(d.Part) > 0 {
		return d.Part
	}
	return pdata.JSON(providerName, foreignResultPart(res))
}

func foreignResultPart(res ai.ToolResult) partOut {
	text := res.Output.Text
	if text == "" {
		text = string(res.Output.JSON)
	}
	text = fmt.Sprintf("[%s result] %s", res.Name, text)
	return partOut{Text: &text}
}

func (cb *contentBuilder) callPart(c ai.ToolCall) json.RawMessage {
	d := cb.b.decode(c.Provider)
	if c.ProviderExecuted {
		return d.Part
	}
	var raw json.RawMessage
	call := pendingCall{aiID: c.ID, name: c.Name}
	if len(d.Part) > 0 {
		raw = d.Part
		if !d.GeneratedID {
			call.sendID = c.ID
		}
	} else {
		call.sendID = sendableID(c.ID)
		raw = pdata.JSON(providerName, partOut{FunctionCall: &callOut{ID: call.sendID, Name: c.Name, Args: argsOf(c.Input)}})
	}
	cb.pending = append(cb.pending, call)
	if cb.signed {
		return raw
	}
	cb.signed = true
	return cb.ensureSignature(raw)
}

func (cb *contentBuilder) ensureSignature(raw json.RawMessage) json.RawMessage {
	var probe struct {
		Signature string `json:"thoughtSignature"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil || probe.Signature != "" || !cb.dummy {
		return raw
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return raw
	}
	fields["thoughtSignature"] = pdata.JSON(providerName, dummySignature)
	cb.warn("a function call has no thought signature from this endpoint, so %s was sent in its place; the model may answer worse", dummySignature)
	return pdata.JSON(providerName, fields)
}

func (cb *contentBuilder) flush() error {
	if len(cb.pending) == 0 {
		if len(cb.results) > 0 {
			cb.warn("%d tool results without a matching call were dropped", len(cb.results))
			cb.results = map[string]ai.ToolResult{}
		}
		return nil
	}
	parts := make([]json.RawMessage, 0, len(cb.pending))
	for _, c := range cb.pending {
		res, ok := cb.results[c.aiID]
		delete(cb.results, c.aiID)
		out := responseOut{ID: c.sendID, Name: c.name, Response: pdata.JSON(providerName, map[string]string{"error": missingResultText})}
		if ok {
			var err error
			out, err = functionResponse(c, res)
			if err != nil {
				return err
			}
		}
		parts = append(parts, pdata.JSON(providerName, partOut{FunctionResponse: &out}))
	}
	if len(cb.results) > 0 {
		cb.warn("%d tool results without a matching call were dropped", len(cb.results))
	}
	cb.add("user", parts)
	cb.pending = nil
	cb.results = map[string]ai.ToolResult{}
	return nil
}

func functionResponse(c pendingCall, res ai.ToolResult) (responseOut, error) {
	name := res.Name
	if name == "" {
		name = c.name
	}
	out := responseOut{ID: c.sendID, Name: name}
	text, files, err := resultContent(res.Output)
	if err != nil {
		return responseOut{}, err
	}
	switch {
	case res.IsError:
		out.Response = pdata.JSON(providerName, map[string]string{"error": text})
	case len(res.Output.JSON) > 0 && isObject(res.Output.JSON):
		out.Response = res.Output.JSON
	case len(res.Output.JSON) > 0:
		out.Response = pdata.JSON(providerName, map[string]json.RawMessage{"output": res.Output.JSON})
	case len(files) > 0:
		refs := make([]map[string]string, len(files))
		for i, f := range files {
			refs[i] = map[string]string{"$ref": f.name}
			out.Parts = append(out.Parts, f.raw)
		}
		out.Response = pdata.JSON(providerName, map[string]any{"output": text, "attachments": refs})
	default:
		out.Response = pdata.JSON(providerName, map[string]string{"output": text})
	}
	return out, nil
}

type resultFile struct {
	name string
	raw  json.RawMessage
}

func resultContent(o ai.ToolOutput) (string, []resultFile, error) {
	if len(o.Parts) == 0 {
		if len(o.JSON) > 0 {
			return string(o.JSON), nil, nil
		}
		return o.Text, nil, nil
	}
	var texts []string
	var files []resultFile
	for _, p := range o.Parts {
		switch v := p.(type) {
		case ai.Text:
			texts = append(texts, v.Text)
		case ai.File:
			if len(v.Data) == 0 {
				return "", nil, errors.New("google: tool result files must carry data")
			}
			name := fmt.Sprintf("file-%d", len(files)+1)
			files = append(files, resultFile{name: name, raw: pdata.JSON(providerName, partOut{InlineData: &blobOut{DisplayName: name, MimeType: v.MediaType, Data: base64.StdEncoding.EncodeToString(v.Data)}})})
		default:
		}
	}
	return strings.Join(texts, "\n"), files, nil
}

func filePart(f ai.File) (json.RawMessage, error) {
	switch {
	case len(f.Data) > 0 && f.MediaType == "":
		return nil, errors.New("google: file part with data needs a media type")
	case len(f.Data) > 0:
		return pdata.JSON(providerName, partOut{InlineData: &blobOut{MimeType: f.MediaType, Data: base64.StdEncoding.EncodeToString(f.Data)}}), nil
	case f.URL != "":
		return pdata.JSON(providerName, partOut{FileData: &fileRefOut{MimeType: f.MediaType, FileURI: f.URL}}), nil
	default:
		return nil, errors.New("google: file part has no data or url")
	}
}

func argsOf(in json.RawMessage) json.RawMessage {
	if len(in) == 0 || string(in) == "null" || !isObject(in) {
		return json.RawMessage("{}")
	}
	return in
}

func isObject(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return strings.HasPrefix(trimmed, "{")
}

func isEmptyUnsigned(raw json.RawMessage) bool {
	var probe struct {
		Text      *string `json:"text"`
		Signature string  `json:"thoughtSignature"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	return probe.Text != nil && *probe.Text == "" && probe.Signature == ""
}
