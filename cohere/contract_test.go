package cohere_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

func newChatModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	return cohere.New(cohere.WithAPIKey("test-key"), cohere.WithBaseURL("https://api.cohere.com"), cohere.WithHTTPClient(client)).Model("command-a-reasoning-08-2025")
}

type wireMsg struct {
	Role       string            `json:"role"`
	Content    []json.RawMessage `json:"content"`
	ToolPlan   string            `json:"tool_plan"`
	ToolCalls  []json.RawMessage `json:"tool_calls"`
	ToolCallID string            `json:"tool_call_id"`
}

func canonical(v any) (json.RawMessage, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var generic any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return nil, err
	}
	return json.Marshal(generic)
}

func assistantItems(m wireMsg) ([]json.RawMessage, error) {
	var out []json.RawMessage
	for _, c := range m.Content {
		item, err := canonical(c)
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	if m.ToolPlan != "" {
		item, err := canonical(map[string]string{"tool_plan": m.ToolPlan})
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	for _, c := range m.ToolCalls {
		item, err := canonical(c)
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, nil
}

func requestItems(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Messages []wireMsg `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	var out []json.RawMessage
	for _, m := range req.Messages {
		if m.Role != "assistant" {
			continue
		}
		items, err := assistantItems(m)
		if err != nil {
			return nil, err
		}
		out = append(out, items...)
	}
	return out, nil
}

func responseItems(body []byte) ([]json.RawMessage, error) {
	var resp struct {
		Message wireMsg `json:"message"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return assistantItems(resp.Message)
}

func streamItems(body []byte) ([]json.RawMessage, error) {
	r := sse.NewReader(bytes.NewReader(body))
	var msg wireMsg
	blocks := map[int]map[string]string{}
	var order []int
	calls := map[int]map[string]string{}
	var callOrder []int
	args := map[int]*strings.Builder{}
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		var e struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Delta struct {
				Message struct {
					Content   json.RawMessage `json:"content"`
					ToolPlan  string          `json:"tool_plan"`
					ToolCalls json.RawMessage `json:"tool_calls"`
				} `json:"message"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
			return nil, err
		}
		var content struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		}
		var call struct {
			ID       string `json:"id"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		}
		m := e.Delta.Message
		if len(m.Content) > 0 && m.Content[0] == '{' {
			if err := json.Unmarshal(m.Content, &content); err != nil {
				return nil, err
			}
		}
		if len(m.ToolCalls) > 0 && m.ToolCalls[0] == '{' {
			if err := json.Unmarshal(m.ToolCalls, &call); err != nil {
				return nil, err
			}
		}
		switch e.Type {
		case "content-start":
			blocks[e.Index] = map[string]string{"type": content.Type}
			order = append(order, e.Index)
		case "content-delta":
			key := "text"
			if blocks[e.Index]["type"] == "thinking" {
				key = "thinking"
			}
			blocks[e.Index][key] += content.Text + content.Thinking
		case "tool-plan-delta":
			msg.ToolPlan += m.ToolPlan
		case "tool-call-start":
			calls[e.Index] = map[string]string{"id": call.ID, "name": call.Function.Name}
			callOrder = append(callOrder, e.Index)
			args[e.Index] = &strings.Builder{}
		case "tool-call-delta":
			args[e.Index].WriteString(call.Function.Arguments)
		}
	}
	for _, i := range order {
		raw, err := json.Marshal(blocks[i])
		if err != nil {
			return nil, err
		}
		msg.Content = append(msg.Content, raw)
	}
	for _, i := range callOrder {
		c := calls[i]
		raw, err := json.Marshal(map[string]any{"id": c["id"], "type": "function", "function": map[string]string{"name": c["name"], "arguments": args[i].String()}})
		if err != nil {
			return nil, err
		}
		msg.ToolCalls = append(msg.ToolCalls, raw)
	}
	return assistantItems(msg)
}

func callIDs(body []byte) ([]string, []string, error) {
	var req struct {
		Messages []wireMsg `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, m := range req.Messages {
		for _, c := range m.ToolCalls {
			var tc struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(c, &tc); err != nil {
				return nil, nil, err
			}
			calls = append(calls, tc.ID)
		}
		if m.Role == "tool" {
			results = append(results, m.ToolCallID)
		}
	}
	return calls, results, nil
}

func TestChatModel_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel: newChatModel,
		Wire:     providertest.Wire{Request: requestItems, Response: responseItems, StreamResponse: streamItems, CallIDs: callIDs},
	}, providertest.Config{
		Dir:    "testdata/contract",
		Replay: "replay",
		Schema: &providertest.SchemaCase{
			Name:   "schema",
			Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1,"format":"phone"},"b":{"type":"integer","minimum":0},"c":{"oneOf":[{"type":"string"},{"type":"integer"}]}},"required":["a"]}`),
			WantWarnings: []string{
				`"minLength"`, `format "phone"`, `"minimum"`, `"oneOf" at /properties/c is not supported and was replaced by "anyOf"`,
			},
		},
		Cases: []providertest.Case{
			{
				Name:       "thinking_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}, Reasoning: ai.ReasoningMedium},
				WantKinds:  []string{"reasoning", "reasoning", "tool-call"},
				WantIDs:    []string{"get_weather_k5mq2x8v1c9d", "c14c80c3-18eb-4519-9460-6c92edd8cfb4"},
				WantUsage:  &ai.Usage{Input: 106, Output: 85, CacheRead: 64},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "parallel_tools",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"reasoning", "tool-call", "tool-call"},
				WantIDs:    []string{"get_weather_a1b2c3d4e5f6", "get_weather_g7h8i9j0k1l2"},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "citations",
				Request:    ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "text/plain", Filename: "colors.txt", Data: []byte("The grass is green."), Citations: true}, ai.Text{Text: "colors?"})}},
				WantKinds:  []string{"text"},
				WantIDs:    []string{"The grass is green.", "lookup_x9y8z7w6v5u4:0"},
				WantFinish: ai.FinishStop,
			},
			{Name: "text", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantUsage: &ai.Usage{Input: 9, Output: 5}, WantFinish: ai.FinishStop},
			{Name: "max_tokens", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishLength},
			{Name: "stop_sequence", Request: ai.TextRequest{Messages: user, StopSequences: []string{"three"}}, WantKinds: []string{"text"}, WantFinish: ai.FinishStop},
			{Name: "timeout", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishError},
		},
	})
}
