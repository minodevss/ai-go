package openai_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/openai"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

var callIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func newResponsesModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	return openai.New(openai.WithAPIKey("test-key"), openai.WithBaseURL("https://api.openai.com/v1"), openai.WithHTTPClient(client)).Model("gpt-5-mini")
}

func requestItems(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	return req.Input, nil
}

func responseItems(body []byte) ([]json.RawMessage, error) {
	var resp struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return resp.Output, nil
}

func streamItems(body []byte) ([]json.RawMessage, error) {
	r := sse.NewReader(bytes.NewReader(body))
	var items []json.RawMessage
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			return items, nil
		}
		if err != nil {
			return nil, err
		}
		var e struct {
			Type string          `json:"type"`
			Item json.RawMessage `json:"item"`
		}
		if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
			return nil, err
		}
		if e.Type == "response.output_item.done" {
			items = append(items, e.Item)
		}
	}
}

func callIDs(body []byte) ([]string, []string, error) {
	items, err := requestItems(body)
	if err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, raw := range items {
		var it struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		}
		if err := json.Unmarshal(raw, &it); err != nil {
			return nil, nil, err
		}
		switch it.Type {
		case "function_call":
			calls = append(calls, it.CallID)
		case "function_call_output":
			results = append(results, it.CallID)
		}
	}
	return calls, results, nil
}

func TestResponsesModel_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	noStore := ai.Providers(openai.Options{Store: new(false), ReasoningSummary: "auto"})
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel:    newResponsesModel,
		CallIDValid: callIDPattern.MatchString,
		Wire:        providertest.Wire{Request: requestItems, Response: responseItems, StreamResponse: streamItems, CallIDs: callIDs},
	}, providertest.Config{
		Dir:    "testdata/contract",
		Replay: "replay",
		Schema: &providertest.SchemaCase{
			Name:         "schema",
			Schema:       json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1},"b":{"type":"integer"}},"required":["a"],"additionalProperties":false}`),
			WantWarnings: []string{`"minLength"`, `property "b"`, "not strict"},
		},
		Cases: []providertest.Case{
			{
				Name:       "reasoning_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}, Provider: noStore},
				WantKinds:  []string{"reasoning", "tool-call"},
				WantIDs:    []string{"rs_aaa1", "fc_bbb2", "call_ccc3", "gAAAAABo", "resp_r1"},
				WantUsage:  &ai.Usage{Input: 120, Output: 85, CacheRead: 64, Reasoning: 70},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "message_citations",
				Request:    ai.TextRequest{Messages: user, Provider: noStore},
				WantKinds:  []string{"reasoning", "text", "text"},
				WantIDs:    []string{"rs_ddd4", "msg_eee5"},
				WantUsage:  &ai.Usage{Input: 40, Output: 30},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "refusal",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantIDs:    []string{"msg_fff6"},
				WantFinish: ai.FinishRefusal,
			},
			{
				Name:    "provider_tools",
				Request: ai.TextRequest{Messages: user, Tools: []ai.Tool{openai.WebSearch(openai.WebSearchOptions{})}},
				WantKinds: []string{
					"tool-call", "tool-result", "tool-call", "tool-result", "tool-call", "tool-result",
					"tool-call", "tool-result", "file", "custom", "tool-call", "tool-result", "text",
				},
				WantIDs:    []string{"ws_111", "fs_222", "ci_333", "ig_444", "mcpl_555", "mcp_666", "msg_777"},
				WantUsage:  &ai.Usage{Input: 300, Output: 120, CacheRead: 200},
				WantFinish: ai.FinishStop,
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
