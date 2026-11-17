package openaicompat_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
	"github.com/minodevss/ai-go/openaicompat"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

var callIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

type wireCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireMessage struct {
	Content   *string
	Fields    map[string]json.RawMessage
	ToolCalls []wireCall
	Refusal   string
}

func (w wireMessage) canonical(keys []string) (json.RawMessage, error) {
	out := map[string]any{"role": "assistant"}
	if w.Content != nil && *w.Content != "" {
		out["content"] = *w.Content
	}
	if w.Refusal != "" {
		out["refusal"] = w.Refusal
	}
	for _, k := range keys {
		if v, ok := w.Fields[k]; ok && string(v) != "null" && string(v) != `""` {
			out[k] = v
		}
	}
	if len(w.ToolCalls) > 0 {
		out["tool_calls"] = w.ToolCalls
	}
	return transport.EncodeJSON(out)
}

func requestMessages(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	return req.Messages, nil
}

func responseMessages(keys []string) func([]byte) ([]json.RawMessage, error) {
	return func(body []byte) ([]json.RawMessage, error) {
		var resp struct {
			Choices []struct {
				Message json.RawMessage `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return nil, err
		}
		var out []json.RawMessage
		for _, c := range resp.Choices {
			var w wireMessage
			if err := json.Unmarshal(c.Message, &w.Fields); err != nil {
				return nil, err
			}
			var typed struct {
				Content   *string    `json:"content"`
				Refusal   string     `json:"refusal"`
				ToolCalls []wireCall `json:"tool_calls"`
			}
			if err := json.Unmarshal(c.Message, &typed); err != nil {
				return nil, err
			}
			w.Content, w.Refusal, w.ToolCalls = typed.Content, typed.Refusal, typed.ToolCalls
			item, err := w.canonical(keys)
			if err != nil {
				return nil, err
			}
			out = append(out, item)
		}
		return out, nil
	}
}

func streamMessages(keys []string) func([]byte) ([]json.RawMessage, error) {
	return func(body []byte) ([]json.RawMessage, error) {
		r := sse.NewReader(bytes.NewReader(body))
		var content, refusal strings.Builder
		fields := map[string][]json.RawMessage{}
		calls := map[int]*wireCall{}
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, err
			}
			if ev.Data == "[DONE]" {
				break
			}
			var chunk struct {
				Choices []struct {
					Delta json.RawMessage `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
				return nil, err
			}
			for _, c := range chunk.Choices {
				if err := collectDelta(c.Delta, keys, &content, &refusal, fields, calls); err != nil {
					return nil, err
				}
			}
		}
		w := wireMessage{Fields: map[string]json.RawMessage{}, Refusal: refusal.String()}
		text := content.String()
		w.Content = &text
		for k, parts := range fields {
			joined, err := joinFields(parts)
			if err != nil {
				return nil, err
			}
			w.Fields[k] = joined
		}
		for i := range len(calls) {
			w.ToolCalls = append(w.ToolCalls, *calls[i])
		}
		item, err := w.canonical(keys)
		if err != nil {
			return nil, err
		}
		return []json.RawMessage{item}, nil
	}
}

func collectDelta(raw json.RawMessage, keys []string, content, refusal *strings.Builder, fields map[string][]json.RawMessage, calls map[int]*wireCall) error {
	var delta struct {
		Content   string `json:"content"`
		Refusal   string `json:"refusal"`
		ToolCalls []struct {
			Index int `json:"index"`
			wireCall
		} `json:"tool_calls"`
	}
	if err := json.Unmarshal(raw, &delta); err != nil {
		return err
	}
	content.WriteString(delta.Content)
	refusal.WriteString(delta.Refusal)
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		return err
	}
	for _, k := range keys {
		if v, ok := all[k]; ok {
			fields[k] = append(fields[k], v)
		}
	}
	for _, tc := range delta.ToolCalls {
		call, ok := calls[tc.Index]
		if !ok {
			call = &wireCall{}
			calls[tc.Index] = call
		}
		if tc.ID != "" {
			call.ID, call.Type, call.Function.Name = tc.ID, tc.Type, tc.Function.Name
		}
		call.Function.Arguments += tc.Function.Arguments
	}
	return nil
}

func joinFields(parts []json.RawMessage) (json.RawMessage, error) {
	var lists []json.RawMessage
	var text strings.Builder
	isList := false
	for _, p := range parts {
		var list []json.RawMessage
		if json.Unmarshal(p, &list) == nil {
			isList = true
			lists = append(lists, list...)
			continue
		}
		var s string
		if json.Unmarshal(p, &s) == nil {
			text.WriteString(s)
		}
	}
	if isList {
		return transport.EncodeJSON(lists)
	}
	return transport.EncodeJSON(text.String())
}

func requestCallIDs(body []byte) ([]string, []string, error) {
	msgs, err := requestMessages(body)
	if err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, raw := range msgs {
		var m struct {
			Role       string     `json:"role"`
			ToolCallID string     `json:"tool_call_id"`
			ToolCalls  []wireCall `json:"tool_calls"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, nil, err
		}
		for _, c := range m.ToolCalls {
			calls = append(calls, c.ID)
		}
		if m.Role == "tool" {
			results = append(results, m.ToolCallID)
		}
	}
	return calls, results, nil
}

func chatWire(keys []string) providertest.Wire {
	return providertest.Wire{Request: requestMessages, Response: responseMessages(keys), StreamResponse: streamMessages(keys), CallIDs: requestCallIDs}
}

func schemaCase() *providertest.SchemaCase {
	return &providertest.SchemaCase{
		Name:         "schema",
		Schema:       json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1},"b":{"type":"integer"}},"required":["a"],"additionalProperties":false}`),
		WantWarnings: []string{`"minLength"`, `property "b"`, "not strict"},
	}
}

func TestOpenRouter_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel: func(t *testing.T, client *http.Client) ai.LanguageModel {
			t.Helper()
			return openaicompat.OpenRouter(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(client)).Model("anthropic/claude-sonnet-4.5")
		},
		CallIDValid: callIDPattern.MatchString,
		Wire:        chatWire([]string{"content", "reasoning_details", "tool_calls"}),
	}, providertest.Config{
		Dir:    "testdata/contract/openrouter",
		Replay: "replay",
		Schema: schemaCase(),
		Cases: []providertest.Case{
			{
				Name:       "reasoning_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"reasoning", "tool-call"},
				WantIDs:    []string{"rs-aaa1", "rs-aaa2", "SIG-OR-1", "gAAAAABoENC", "toolu_01ABC", "gen-r1"},
				WantUsage:  &ai.Usage{Input: 120, Output: 85, CacheRead: 64, Reasoning: 70},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "message_citations",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"reasoning", "text"},
				WantIDs:    []string{"rs-bbb1", "gen-c1"},
				WantUsage:  &ai.Usage{Input: 40, Output: 30},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "refusal",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantFinish: ai.FinishRefusal,
			},
			{
				Name:       "incomplete",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantFinish: ai.FinishLength,
			},
		},
	})
}

func TestDeepSeek_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel: func(t *testing.T, client *http.Client) ai.LanguageModel {
			t.Helper()
			return openaicompat.DeepSeek(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(client)).Model("deepseek-v4-pro")
		},
		CallIDValid: callIDPattern.MatchString,
		Wire:        chatWire([]string{"content", "reasoning_content", "tool_calls"}),
	}, providertest.Config{
		Dir:    "testdata/contract/deepseek",
		Replay: "replay",
		Schema: schemaCase(),
		Cases: []providertest.Case{
			{
				Name:       "reasoning_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"reasoning", "tool-call"},
				WantIDs:    []string{"call_00_XyZ", "ds-r1"},
				WantUsage:  &ai.Usage{Input: 100, Output: 30, CacheRead: 64, Reasoning: 20},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "answer",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"reasoning", "text"},
				WantUsage:  &ai.Usage{Input: 40, Output: 12, Reasoning: 5},
				WantFinish: ai.FinishStop,
			},
		},
	})
}
