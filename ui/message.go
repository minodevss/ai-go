package ui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type PartState string

const (
	StateStreaming PartState = "streaming"
	StateDone      PartState = "done"
)

type ToolState string

const (
	ToolInputStreaming    ToolState = "input-streaming"
	ToolInputAvailable    ToolState = "input-available"
	ToolApprovalRequested ToolState = "approval-requested"
	ToolApprovalResponded ToolState = "approval-responded"
	ToolOutputAvailable   ToolState = "output-available"
	ToolOutputError       ToolState = "output-error"
	ToolOutputDenied      ToolState = "output-denied"
)

// Message mirrors the AI SDK UIMessage. Fields tagged json:"-" exist so FromMessages and ToAI round-trip without loss; they never reach the wire.
type Message struct {
	ID       string          `json:"id"`
	Role     Role            `json:"role"`
	Metadata json.RawMessage `json:"metadata,omitzero"`
	Parts    []Part          `json:"parts"`
}

type Part interface{ isPart() }

type TextPart struct {
	Text      string
	State     PartState
	Citations []ai.Citation
	Provider  ai.ProviderData
}

type ReasoningPart struct {
	Text     string
	State    PartState
	Provider ai.ProviderData
}

// FilePart.URL is a data URL for inline bytes; Reference is rejected on input because provider file ids are never accepted from the wire.
type FilePart struct {
	MediaType string
	Filename  string
	URL       string
	Reference map[string]string
	Citable   bool
	Provider  ai.ProviderData
}

type Approval struct {
	ID            string
	Approved      *bool
	Reason        string
	RequestReason string
	InputHash     [32]byte
}

type ToolPart struct {
	Name             string
	Dynamic          bool
	CallID           string
	State            ToolState
	Input            json.RawMessage
	RawInput         string
	Output           ai.ToolOutput
	ErrorText        string
	Preliminary      bool
	ProviderExecuted bool
	Approval         *Approval
	CallProvider     ai.ProviderData
	ResultProvider   ai.ProviderData
}

type SourceURLPart struct {
	SourceID string
	URL      string
	Title    string
	Provider ai.ProviderData
}

type SourceDocumentPart struct {
	SourceID  string
	MediaType string
	Title     string
	Filename  string
	Provider  ai.ProviderData
}

type DataPart struct {
	Name string
	ID   string
	Data json.RawMessage
}

// StepStartPart.Provider holds the message-level provider data of the assistant message that begins at this step.
type StepStartPart struct {
	Provider ai.ProviderData
}

type CustomPart struct {
	Kind     string
	Data     json.RawMessage
	Provider ai.ProviderData
}

// RawPart keeps part types this package does not model, such as reasoning-file and steering-message.
type RawPart struct {
	Type string
	Raw  json.RawMessage
}

func (TextPart) isPart()           {}
func (ReasoningPart) isPart()      {}
func (FilePart) isPart()           {}
func (ToolPart) isPart()           {}
func (SourceURLPart) isPart()      {}
func (SourceDocumentPart) isPart() {}
func (DataPart) isPart()           {}
func (StepStartPart) isPart()      {}
func (CustomPart) isPart()         {}
func (RawPart) isPart()            {}

type approvalWire struct {
	ID            string `json:"id"`
	Approved      *bool  `json:"approved,omitzero"`
	Reason        string `json:"reason,omitzero"`
	RequestReason string `json:"requestReason,omitzero"`
}

type citationSourceWire struct {
	Type       string  `json:"type"`
	SourceType string  `json:"sourceType"`
	ID         string  `json:"id"`
	URL        string  `json:"url,omitzero"`
	MediaType  string  `json:"mediaType,omitzero"`
	Title      *string `json:"title,omitzero"`
	Filename   string  `json:"filename,omitzero"`
}

type citationWire struct {
	Source     citationSourceWire `json:"source"`
	StartIndex *int               `json:"startIndex,omitzero"`
	EndIndex   *int               `json:"endIndex,omitzero"`
	CitedText  string             `json:"citedText,omitzero"`
}

func citationsToWire(cs []ai.Citation) []citationWire {
	out := make([]citationWire, len(cs))
	for i, c := range cs {
		src := citationSourceWire{Type: "source", ID: fmt.Sprintf("cite-%d", i), Title: &c.Title}
		if c.URL != "" {
			src.SourceType = "url"
			src.URL = c.URL
			if c.Title == "" {
				src.Title = nil
			}
		} else {
			src.SourceType = "document"
			src.MediaType = "text/plain"
		}
		w := citationWire{Source: src, CitedText: c.CitedText}
		if c.Start != 0 || c.End != 0 {
			w.StartIndex, w.EndIndex = &c.Start, &c.End
		}
		out[i] = w
	}
	return out
}

func citationsFromWire(ws []citationWire) []ai.Citation {
	out := make([]ai.Citation, len(ws))
	for i, w := range ws {
		c := ai.Citation{URL: w.Source.URL, CitedText: w.CitedText}
		if w.Source.Title != nil {
			c.Title = *w.Source.Title
		}
		if w.StartIndex != nil {
			c.Start = *w.StartIndex
		}
		if w.EndIndex != nil {
			c.End = *w.EndIndex
		}
		out[i] = c
	}
	return out
}

func (p TextPart) MarshalJSON() ([]byte, error) {
	w := struct {
		Type      string          `json:"type"`
		Text      string          `json:"text"`
		State     PartState       `json:"state,omitzero"`
		Citations []citationWire  `json:"citations,omitzero"`
		Provider  ai.ProviderData `json:"providerMetadata,omitzero"`
	}{"text", p.Text, p.State, citationsToWire(p.Citations), p.Provider}
	if len(p.Citations) == 0 {
		w.Citations = nil
	}
	return json.Marshal(w)
}

func (p ReasoningPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		State    PartState       `json:"state,omitzero"`
		Provider ai.ProviderData `json:"providerMetadata,omitzero"`
	}{"reasoning", p.Text, p.State, p.Provider})
}

func (p FilePart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type      string            `json:"type"`
		MediaType string            `json:"mediaType"`
		Filename  string            `json:"filename,omitzero"`
		URL       string            `json:"url"`
		Reference map[string]string `json:"providerReference,omitzero"`
		Provider  ai.ProviderData   `json:"providerMetadata,omitzero"`
	}{"file", p.MediaType, p.Filename, p.URL, p.Reference, p.Provider})
}

func (p SourceURLPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type     string          `json:"type"`
		SourceID string          `json:"sourceId"`
		URL      string          `json:"url"`
		Title    string          `json:"title,omitzero"`
		Provider ai.ProviderData `json:"providerMetadata,omitzero"`
	}{"source-url", p.SourceID, p.URL, p.Title, p.Provider})
}

func (p SourceDocumentPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type      string          `json:"type"`
		SourceID  string          `json:"sourceId"`
		MediaType string          `json:"mediaType"`
		Title     string          `json:"title"`
		Filename  string          `json:"filename,omitzero"`
		Provider  ai.ProviderData `json:"providerMetadata,omitzero"`
	}{"source-document", p.SourceID, p.MediaType, p.Title, p.Filename, p.Provider})
}

func (p DataPart) MarshalJSON() ([]byte, error) {
	data := p.Data
	if len(data) == 0 {
		data = json.RawMessage("null")
	}
	return json.Marshal(struct {
		Type string          `json:"type"`
		ID   string          `json:"id,omitzero"`
		Data json.RawMessage `json:"data"`
	}{"data-" + p.Name, p.ID, data})
}

func (StepStartPart) MarshalJSON() ([]byte, error) { return []byte(`{"type":"step-start"}`), nil }

func (p CustomPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type     string          `json:"type"`
		Kind     string          `json:"kind"`
		Data     json.RawMessage `json:"data,omitzero"`
		Provider ai.ProviderData `json:"providerMetadata,omitzero"`
	}{"custom", p.Kind, p.Data, p.Provider})
}

func (p RawPart) MarshalJSON() ([]byte, error) { return p.Raw, nil }

func (p ToolPart) MarshalJSON() ([]byte, error) {
	w := struct {
		Type             string          `json:"type"`
		ToolName         string          `json:"toolName,omitzero"`
		ToolCallID       string          `json:"toolCallId"`
		State            ToolState       `json:"state"`
		Input            json.RawMessage `json:"input,omitzero"`
		RawInput         string          `json:"rawInput,omitzero"`
		Output           json.RawMessage `json:"output,omitzero"`
		ErrorText        string          `json:"errorText,omitzero"`
		ProviderExecuted bool            `json:"providerExecuted,omitzero"`
		CallProvider     ai.ProviderData `json:"callProviderMetadata,omitzero"`
		ResultProvider   ai.ProviderData `json:"resultProviderMetadata,omitzero"`
		Preliminary      bool            `json:"preliminary,omitzero"`
		Approval         *approvalWire   `json:"approval,omitzero"`
	}{
		ToolCallID:       p.CallID,
		State:            p.State,
		RawInput:         p.RawInput,
		ErrorText:        p.ErrorText,
		ProviderExecuted: p.ProviderExecuted,
		CallProvider:     p.CallProvider,
		ResultProvider:   p.ResultProvider,
		Preliminary:      p.Preliminary,
	}
	if p.Dynamic {
		w.Type, w.ToolName = "dynamic-tool", p.Name
	} else {
		w.Type = "tool-" + p.Name
	}
	if p.State != ToolInputStreaming || len(p.Input) > 0 {
		w.Input = p.Input
		if len(w.Input) == 0 {
			w.Input = json.RawMessage("{}")
		}
	}
	if p.State == ToolOutputAvailable {
		out, err := outputToWire(p.Output)
		if err != nil {
			return nil, err
		}
		w.Output = out
	}
	if p.Approval != nil {
		w.Approval = &approvalWire{ID: p.Approval.ID, Approved: p.Approval.Approved, Reason: p.Approval.Reason, RequestReason: p.Approval.RequestReason}
	}
	return json.Marshal(w)
}

func outputToWire(o ai.ToolOutput) (json.RawMessage, error) {
	switch {
	case len(o.JSON) > 0:
		return o.JSON, nil
	case len(o.Parts) > 0:
		return json.Marshal(o.Parts)
	default:
		return json.Marshal(o.Text)
	}
}

func outputFromWire(raw json.RawMessage) ai.ToolOutput {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return ai.ToolOutput{}
	}
	var s string
	if trimmed[0] == '"' && json.Unmarshal(trimmed, &s) == nil {
		return ai.ToolOutput{Text: s}
	}
	return ai.ToolOutput{JSON: bytes.Clone(trimmed)}
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var w struct {
		ID       string            `json:"id"`
		Role     Role              `json:"role"`
		Metadata json.RawMessage   `json:"metadata"`
		Parts    []json.RawMessage `json:"parts"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return fmt.Errorf("ui: decode message: %w", err)
	}
	m.ID, m.Role = w.ID, w.Role
	m.Metadata = nil
	if t := bytes.TrimSpace(w.Metadata); len(t) > 0 && string(t) != "null" {
		m.Metadata = bytes.Clone(t)
	}
	m.Parts = make([]Part, 0, len(w.Parts))
	for i, raw := range w.Parts {
		p, err := decodePart(raw)
		if err != nil {
			return fmt.Errorf("ui: decode part %d: %w", i, err)
		}
		m.Parts = append(m.Parts, p)
	}
	return nil
}

func decodePart(raw json.RawMessage) (Part, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, err
	}
	switch {
	case head.Type == "text":
		return decodeText(raw)
	case head.Type == "reasoning":
		var w struct {
			Text     string          `json:"text"`
			State    PartState       `json:"state"`
			Provider ai.ProviderData `json:"providerMetadata"`
		}
		err := json.Unmarshal(raw, &w)
		return ReasoningPart{Text: w.Text, State: w.State, Provider: w.Provider}, err
	case head.Type == "file":
		var w struct {
			MediaType string            `json:"mediaType"`
			Filename  string            `json:"filename"`
			URL       string            `json:"url"`
			Reference map[string]string `json:"providerReference"`
			Provider  ai.ProviderData   `json:"providerMetadata"`
		}
		err := json.Unmarshal(raw, &w)
		return FilePart{MediaType: w.MediaType, Filename: w.Filename, URL: w.URL, Reference: w.Reference, Provider: w.Provider}, err
	case head.Type == "source-url":
		var w struct {
			SourceID string          `json:"sourceId"`
			URL      string          `json:"url"`
			Title    string          `json:"title"`
			Provider ai.ProviderData `json:"providerMetadata"`
		}
		err := json.Unmarshal(raw, &w)
		return SourceURLPart{SourceID: w.SourceID, URL: w.URL, Title: w.Title, Provider: w.Provider}, err
	case head.Type == "source-document":
		var w struct {
			SourceID  string          `json:"sourceId"`
			MediaType string          `json:"mediaType"`
			Title     string          `json:"title"`
			Filename  string          `json:"filename"`
			Provider  ai.ProviderData `json:"providerMetadata"`
		}
		err := json.Unmarshal(raw, &w)
		return SourceDocumentPart{SourceID: w.SourceID, MediaType: w.MediaType, Title: w.Title, Filename: w.Filename, Provider: w.Provider}, err
	case head.Type == "step-start":
		return StepStartPart{}, nil
	case head.Type == "custom":
		var w struct {
			Kind     string          `json:"kind"`
			Data     json.RawMessage `json:"data"`
			Provider ai.ProviderData `json:"providerMetadata"`
		}
		err := json.Unmarshal(raw, &w)
		return CustomPart{Kind: w.Kind, Data: w.Data, Provider: w.Provider}, err
	case head.Type == "dynamic-tool":
		return decodeTool(raw, "", true)
	case strings.HasPrefix(head.Type, "tool-"):
		return decodeTool(raw, strings.TrimPrefix(head.Type, "tool-"), false)
	case strings.HasPrefix(head.Type, "data-"):
		var w struct {
			ID   string          `json:"id"`
			Data json.RawMessage `json:"data"`
		}
		err := json.Unmarshal(raw, &w)
		return DataPart{Name: strings.TrimPrefix(head.Type, "data-"), ID: w.ID, Data: w.Data}, err
	default:
		return RawPart{Type: head.Type, Raw: bytes.Clone(raw)}, nil
	}
}

func decodeText(raw json.RawMessage) (Part, error) {
	var w struct {
		Text      string          `json:"text"`
		State     PartState       `json:"state"`
		Citations []citationWire  `json:"citations"`
		Provider  ai.ProviderData `json:"providerMetadata"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, err
	}
	p := TextPart{Text: w.Text, State: w.State, Provider: w.Provider}
	if len(w.Citations) > 0 {
		p.Citations = citationsFromWire(w.Citations)
	}
	return p, nil
}

func decodeTool(raw json.RawMessage, name string, dynamic bool) (Part, error) {
	var w struct {
		ToolName         string          `json:"toolName"`
		ToolCallID       string          `json:"toolCallId"`
		State            ToolState       `json:"state"`
		Input            json.RawMessage `json:"input"`
		RawInput         json.RawMessage `json:"rawInput"`
		Output           json.RawMessage `json:"output"`
		ErrorText        string          `json:"errorText"`
		ProviderExecuted bool            `json:"providerExecuted"`
		CallProvider     ai.ProviderData `json:"callProviderMetadata"`
		ResultProvider   ai.ProviderData `json:"resultProviderMetadata"`
		Preliminary      bool            `json:"preliminary"`
		Approval         *approvalWire   `json:"approval"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, err
	}
	if dynamic {
		name = w.ToolName
	}
	p := ToolPart{
		Name:             name,
		Dynamic:          dynamic,
		CallID:           w.ToolCallID,
		State:            w.State,
		Output:           outputFromWire(w.Output),
		ErrorText:        w.ErrorText,
		ProviderExecuted: w.ProviderExecuted,
		CallProvider:     w.CallProvider,
		ResultProvider:   w.ResultProvider,
		Preliminary:      w.Preliminary,
	}
	if t := bytes.TrimSpace(w.Input); len(t) > 0 && string(t) != "null" {
		p.Input = bytes.Clone(t)
	}
	var rawInput string
	if json.Unmarshal(w.RawInput, &rawInput) == nil {
		p.RawInput = rawInput
	}
	if w.Approval != nil {
		p.Approval = &Approval{ID: w.Approval.ID, Approved: w.Approval.Approved, Reason: w.Approval.Reason, RequestReason: w.Approval.RequestReason}
	}
	return p, nil
}
