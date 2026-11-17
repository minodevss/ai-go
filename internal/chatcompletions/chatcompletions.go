// Package chatcompletions implements the Chat Completions wire format for OpenAI and compatible providers.
package chatcompletions

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const defaultMaxCallID = 40

type TokenField string

const (
	MaxCompletionTokens TokenField = "max_completion_tokens"
	MaxTokens           TokenField = "max_tokens"
)

// Replay says how earlier reasoning is sent back. A zero Field sends none.
type Replay struct {
	Field         string
	ToolCallsOnly bool
	Chunks        bool
}

type ResponseExtras struct {
	Data      map[string]json.RawMessage
	Citations []ai.Citation
	Warnings  []ai.Warning
}

// Dialect describes one provider's Chat Completions variant. The zero value of every field except Provider is the OpenAI behavior.
type Dialect struct {
	Provider        string
	Path            string
	MaxTokens       TokenField
	NoStreamOptions bool
	TopK            bool
	ToolStrict      bool
	ToolResultName  bool
	ReasoningFields []string
	Replay          Replay
	Verbatim        []string
	CallID          func(id string) string
	Request         func(c *ai.Call, r *Request) ([]ai.Warning, error)
	Usage           func(raw json.RawMessage) ai.Usage
	Response        func(body map[string]json.RawMessage) ResponseExtras
}

type Model struct {
	client  *transport.Client
	id      string
	dialect *Dialect
}

func New(client *transport.Client, id string, d *Dialect) *Model {
	return &Model{client: client, id: id, dialect: d}
}

func (m *Model) Provider() string { return m.dialect.Provider }

func (m *Model) ModelID() string { return m.id }

func (m *Model) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	req, warnings, err := m.request(c, false)
	if err != nil {
		return nil, err
	}
	var raw json.RawMessage
	header, err := m.client.JSON(ctx, transport.Request{Path: m.path(), Body: req, Header: c.Headers}, &raw)
	if err != nil {
		return nil, err
	}
	return m.decode(raw, warnings, header)
}

func (m *Model) decode(raw json.RawMessage, warnings []ai.Warning, header map[string][]string) (*ai.Response, error) {
	var body response
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("%s: decode response: %w", m.dialect.Provider, err)
	}
	if len(body.Choices) == 0 {
		return nil, fmt.Errorf("%s: chat response %s has no choices", m.dialect.Provider, body.ID)
	}
	extras := m.extras(raw)
	choice := body.Choices[0]
	var msg message
	if err := json.Unmarshal(choice.Message, &msg); err != nil {
		return nil, fmt.Errorf("%s: decode response message: %w", m.dialect.Provider, err)
	}
	parts, hasRefusal := m.parts(msg, verbatimOf(m.dialect.Verbatim, choice.Message), extras.Citations, choice.Message)
	finish, finishRaw := finishReason(choice.FinishReason, hasRefusal)
	return &ai.Response{
		Content:      parts,
		Usage:        m.usage(body.Usage),
		FinishReason: finish,
		FinishRaw:    finishRaw,
		Warnings:     slices.Concat(warnings, extras.Warnings),
		Provider:     m.responseProvider(body.ID, body.ServiceTier, extras.Data),
		Header:       header,
	}, nil
}

func (m *Model) path() string {
	if m.dialect.Path != "" {
		return m.dialect.Path
	}
	return "/chat/completions"
}

func (m *Model) usage(raw json.RawMessage) ai.Usage {
	if m.dialect.Usage != nil {
		return m.dialect.Usage(raw)
	}
	return openaiwire.Usage(raw)
}

func (m *Model) extras(raw json.RawMessage) ResponseExtras {
	if m.dialect.Response == nil {
		return ResponseExtras{}
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return ResponseExtras{}
	}
	return m.dialect.Response(top)
}

func (m *Model) callID(id string) string {
	if m.dialect.CallID != nil {
		return m.dialect.CallID(id)
	}
	return openaiwire.CallID(id, defaultMaxCallID)
}

func (m *Model) responseProvider(id, tier string, extra map[string]json.RawMessage) ai.ProviderData {
	data := map[string]json.RawMessage{}
	if id != "" {
		data["id"] = pdata.JSON(m.dialect.Provider, id)
	}
	if tier != "" {
		data["serviceTier"] = pdata.JSON(m.dialect.Provider, tier)
	}
	maps.Copy(data, extra)
	if len(data) == 0 {
		return nil
	}
	return pdata.Encode(m.dialect.Provider, data)
}

func OptionsOf[T any](provider string, d ai.ProviderData) (T, error) {
	var out T
	raw, ok := d[provider]
	if !ok {
		return out, nil
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return *new(T), fmt.Errorf("%s: decode provider options: %w", provider, err)
	}
	return out, nil
}

var _ ai.LanguageModel = (*Model)(nil)
