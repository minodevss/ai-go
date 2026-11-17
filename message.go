// Package ai is a provider-neutral Go SDK for language, embedding and reranking models.
package ai

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role     Role         `json:"role"`
	Content  Parts        `json:"content"`
	Provider ProviderData `json:"provider,omitzero"`
}

// Parts decodes each element by its "type" field. Part pointers are accepted and normalized to values.
type Parts []Part

type Part interface{ isPart() }

type Text struct {
	Text      string       `json:"text"`
	Citations []Citation   `json:"citations,omitzero"`
	Provider  ProviderData `json:"provider,omitzero"`
}

type Reasoning struct {
	Text     string       `json:"text"`
	Provider ProviderData `json:"provider,omitzero"`
}

type File struct {
	MediaType string       `json:"mediaType,omitzero"`
	Filename  string       `json:"filename,omitzero"`
	Data      []byte       `json:"data,omitzero"`
	URL       string       `json:"url,omitzero"`
	Citations bool         `json:"citations,omitzero"`
	Provider  ProviderData `json:"provider,omitzero"`
}

type ToolCall struct {
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Input            json.RawMessage `json:"input,omitzero"`
	ProviderExecuted bool            `json:"providerExecuted,omitzero"`
	Provider         ProviderData    `json:"provider,omitzero"`
}

type ToolResult struct {
	CallID           string       `json:"callId"`
	Name             string       `json:"name"`
	Output           ToolOutput   `json:"output"`
	IsError          bool         `json:"isError,omitzero"`
	ProviderExecuted bool         `json:"providerExecuted,omitzero"`
	Provider         ProviderData `json:"provider,omitzero"`
}

type ToolApprovalRequest struct {
	ApprovalID string   `json:"approvalId"`
	CallID     string   `json:"callId"`
	Reason     string   `json:"reason,omitzero"`
	InputHash  [32]byte `json:"-"`
}

type ToolApprovalResponse struct {
	ApprovalID string `json:"approvalId"`
	CallID     string `json:"callId"`
	Approved   bool   `json:"approved"`
	Reason     string `json:"reason,omitzero"`
}

type Custom struct {
	Kind     string          `json:"kind"`
	Data     json.RawMessage `json:"data,omitzero"`
	Provider ProviderData    `json:"provider,omitzero"`
}

type Citation struct {
	Kind      string       `json:"kind,omitzero"`
	URL       string       `json:"url,omitzero"`
	Title     string       `json:"title,omitzero"`
	CitedText string       `json:"citedText,omitzero"`
	DocIndex  *int         `json:"docIndex,omitzero"`
	Start     int          `json:"start,omitzero"`
	End       int          `json:"end,omitzero"`
	Provider  ProviderData `json:"provider,omitzero"`
}

// ToolOutput carries exactly one of Text, JSON or Parts.
type ToolOutput struct {
	Text  string          `json:"text,omitzero"`
	JSON  json.RawMessage `json:"json,omitzero"`
	Parts Parts           `json:"parts,omitzero"`
}

func (Text) isPart()                 {}
func (Reasoning) isPart()            {}
func (File) isPart()                 {}
func (ToolCall) isPart()             {}
func (ToolResult) isPart()           {}
func (ToolApprovalRequest) isPart()  {}
func (ToolApprovalResponse) isPart() {}
func (Custom) isPart()               {}

const (
	typeText         = "text"
	typeReasoning    = "reasoning"
	typeFile         = "file"
	typeToolCall     = "tool-call"
	typeToolResult   = "tool-result"
	typeApprovalReq  = "tool-approval-request"
	typeApprovalResp = "tool-approval-response"
	typeCustom       = "custom"
)

func User(s string) Message {
	return Message{Role: RoleUser, Content: Parts{Text{Text: s}}}
}

func UserParts(parts ...Part) Message {
	return Message{Role: RoleUser, Content: normalizeParts(parts)}
}

func Assistant(s string) Message {
	return Message{Role: RoleAssistant, Content: Parts{Text{Text: s}}}
}

func Texts(s ...string) []Text {
	out := make([]Text, len(s))
	for i, v := range s {
		out[i] = Text{Text: v}
	}
	return out
}

func (m Message) clone() Message {
	m.Content = slices.Clone(m.Content)
	m.Provider = m.Provider.clone()
	return m
}

func cloneMessages(msgs []Message) []Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]Message, len(msgs))
	for i, m := range msgs {
		out[i] = m.clone()
	}
	return out
}

func normalizeParts(parts []Part) Parts {
	if len(parts) == 0 {
		return nil
	}
	out := make(Parts, 0, len(parts))
	for _, p := range parts {
		if v := normalizePart(p); v != nil {
			out = append(out, v)
		}
	}
	return out
}

func normalizePart(p Part) Part {
	switch v := p.(type) {
	case *Text:
		return derefPart(v)
	case *Reasoning:
		return derefPart(v)
	case *File:
		return derefPart(v)
	case *ToolCall:
		return derefPart(v)
	case *ToolResult:
		return derefPart(v)
	case *ToolApprovalRequest:
		return derefPart(v)
	case *ToolApprovalResponse:
		return derefPart(v)
	case *Custom:
		return derefPart(v)
	default:
		return p
	}
}

func derefPart[T Part](p *T) Part {
	if p == nil {
		return nil
	}
	return *p
}

func (t Text) MarshalJSON() ([]byte, error) {
	type wire Text
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeText, wire(t)})
}

func (r Reasoning) MarshalJSON() ([]byte, error) {
	type wire Reasoning
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeReasoning, wire(r)})
}

func (f File) MarshalJSON() ([]byte, error) {
	type wire File
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeFile, wire(f)})
}

func (c ToolCall) MarshalJSON() ([]byte, error) {
	type wire ToolCall
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeToolCall, wire(c)})
}

func (r ToolResult) MarshalJSON() ([]byte, error) {
	type wire ToolResult
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeToolResult, wire(r)})
}

func (r ToolApprovalRequest) MarshalJSON() ([]byte, error) {
	type wire ToolApprovalRequest
	return json.Marshal(struct {
		Type      string `json:"type"`
		InputHash string `json:"inputHash"`
		wire
	}{typeApprovalReq, hex.EncodeToString(r.InputHash[:]), wire(r)})
}

func (r ToolApprovalResponse) MarshalJSON() ([]byte, error) {
	type wire ToolApprovalResponse
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeApprovalResp, wire(r)})
}

func (c Custom) MarshalJSON() ([]byte, error) {
	type wire Custom
	return json.Marshal(struct {
		Type string `json:"type"`
		wire
	}{typeCustom, wire(c)})
}

func (p *Parts) UnmarshalJSON(b []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(b, &raws); err != nil {
		return fmt.Errorf("ai: decode parts: %w", err)
	}
	if raws == nil {
		*p = nil
		return nil
	}
	out := make(Parts, 0, len(raws))
	for i, raw := range raws {
		part, err := decodePart(raw)
		if err != nil {
			return fmt.Errorf("ai: decode part %d: %w", i, err)
		}
		out = append(out, part)
	}
	*p = out
	return nil
}

func decodePart(raw json.RawMessage) (Part, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	var kind string
	if t, ok := fields["type"]; ok {
		if err := json.Unmarshal(t, &kind); err != nil {
			return nil, fmt.Errorf("ai: decode type: %w", err)
		}
	}
	switch kind {
	case typeText:
		return decodeKnown[Text](raw, fields, "text", "citations", "provider")
	case typeReasoning:
		return decodeKnown[Reasoning](raw, fields, "text", "provider")
	case typeFile:
		return decodeKnown[File](raw, fields, "mediaType", "filename", "data", "url", "citations", "provider")
	case typeToolCall:
		return decodeKnown[ToolCall](raw, fields, "id", "name", "input", "providerExecuted", "provider")
	case typeToolResult:
		return decodeKnown[ToolResult](raw, fields, "callId", "name", "output", "isError", "providerExecuted", "provider")
	case typeApprovalReq:
		return decodeApprovalRequest(raw)
	case typeApprovalResp:
		return decodeKnown[ToolApprovalResponse](raw, fields, "approvalId", "callId", "approved", "reason")
	case typeCustom:
		return decodeKnown[Custom](raw, fields, "kind", "data", "provider")
	default:
		return Custom{Kind: kind, Data: slices.Clone(raw)}, nil
	}
}

func decodeKnown[T Part](raw json.RawMessage, fields map[string]json.RawMessage, known ...string) (Part, error) {
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	extra := unknownFields(fields, known)
	if len(extra) == 0 {
		return v, nil
	}
	encoded, err := json.Marshal(extra)
	if err != nil {
		return nil, err
	}
	switch p := any(&v).(type) {
	case *Text:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	case *Reasoning:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	case *File:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	case *ToolCall:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	case *ToolResult:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	case *Custom:
		p.Provider = p.Provider.WithRaw(unknownFieldsKey, encoded)
	default:
	}
	return v, nil
}

func decodeApprovalRequest(raw json.RawMessage) (Part, error) {
	var wire struct {
		ApprovalID string `json:"approvalId"`
		CallID     string `json:"callId"`
		Reason     string `json:"reason"`
		InputHash  string `json:"inputHash"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, err
	}
	req := ToolApprovalRequest{ApprovalID: wire.ApprovalID, CallID: wire.CallID, Reason: wire.Reason}
	if wire.InputHash != "" {
		b, err := hex.DecodeString(wire.InputHash)
		if err != nil || len(b) != len(req.InputHash) {
			return nil, errors.New("ai: inputHash: invalid value")
		}
		copy(req.InputHash[:], b)
	}
	return req, nil
}

func unknownFields(fields map[string]json.RawMessage, known []string) map[string]json.RawMessage {
	var extra map[string]json.RawMessage
	for k, v := range fields {
		if k == "type" || slices.Contains(known, k) {
			continue
		}
		if extra == nil {
			extra = map[string]json.RawMessage{}
		}
		extra[k] = v
	}
	return extra
}
