package ollama_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ollama"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

func newChatModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	return ollama.New(ollama.WithBaseURL("http://localhost:11434"), ollama.WithHTTPClient(client)).Model("qwen3")
}

func canonicalMessage(raw json.RawMessage) (json.RawMessage, error) {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	for k, v := range m {
		switch x := v.(type) {
		case string:
			if x == "" {
				delete(m, k)
			}
		case []any:
			if len(x) == 0 {
				delete(m, k)
			}
		case nil:
			delete(m, k)
		}
	}
	return json.Marshal(m)
}

func requestMessages(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	out := make([]json.RawMessage, len(req.Messages))
	for i, m := range req.Messages {
		c, err := canonicalMessage(m)
		if err != nil {
			return nil, err
		}
		out[i] = c
	}
	return out, nil
}

func responseMessage(body []byte) ([]json.RawMessage, error) {
	var resp struct {
		Message json.RawMessage `json:"message"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	c, err := canonicalMessage(resp.Message)
	if err != nil {
		return nil, err
	}
	return []json.RawMessage{c}, nil
}

func streamMessage(body []byte) ([]json.RawMessage, error) {
	var content, thinking strings.Builder
	var calls []json.RawMessage
	sc := bufio.NewScanner(bytes.NewReader(body))
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var chunk struct {
			Message struct {
				Content   string            `json:"content"`
				Thinking  string            `json:"thinking"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
		}
		if err := json.Unmarshal(sc.Bytes(), &chunk); err != nil {
			return nil, err
		}
		content.WriteString(chunk.Message.Content)
		thinking.WriteString(chunk.Message.Thinking)
		calls = append(calls, chunk.Message.ToolCalls...)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	raw, err := json.Marshal(map[string]any{"role": "assistant", "content": content.String(), "thinking": thinking.String(), "tool_calls": calls})
	if err != nil {
		return nil, err
	}
	c, err := canonicalMessage(raw)
	if err != nil {
		return nil, err
	}
	return []json.RawMessage{c}, nil
}

func toolNames(body []byte) ([]string, []string, error) {
	var req struct {
		Messages []struct {
			Role      string `json:"role"`
			ToolName  string `json:"tool_name"`
			ToolCalls []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, m := range req.Messages {
		for _, c := range m.ToolCalls {
			calls = append(calls, c.Function.Name)
		}
		if m.Role == "tool" {
			results = append(results, m.ToolName)
		}
	}
	return calls, results, nil
}

func TestChatModel_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel: newChatModel,
		Wire:     providertest.Wire{Request: requestMessages, Response: responseMessage, StreamResponse: streamMessage, CallIDs: toolNames},
	}, providertest.Config{
		Dir:    "testdata/contract",
		Replay: "replay",
		Cases: []providertest.Case{
			{
				Name:       "text",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantUsage:  &ai.Usage{Input: 26, Output: 298, CacheRead: 16},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "thinking_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}, Reasoning: ai.ReasoningMedium},
				WantKinds:  []string{"reasoning", "tool-call"},
				WantIDs:    []string{"Seoul, so I should call get_weather.", `"index":0`},
				WantUsage:  &ai.Usage{Input: 141, Output: 62},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "parallel_tools",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"tool-call", "tool-call"},
				WantIDs:    []string{"Seoul", "Busan"},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "length",
				Request:    ai.TextRequest{Messages: user, MaxOutputTokens: 4},
				WantKinds:  []string{"text"},
				WantFinish: ai.FinishLength,
			},
		},
	})
}
