package anthropic_test

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
	"github.com/minodevss/ai-go/anthropic"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

var callIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func newLanguageModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	return anthropic.New(anthropic.WithAPIKey("test-key"), anthropic.WithBaseURL("https://api.anthropic.com"), anthropic.WithHTTPClient(client)).Model("claude-sonnet-5-5")
}

func requestBlocks(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	var out []json.RawMessage
	for _, m := range req.Messages {
		out = append(out, m.Content...)
	}
	return out, nil
}

func responseBlocks(body []byte) ([]json.RawMessage, error) {
	var resp struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return resp.Content, nil
}

type streamBlock struct {
	fields    map[string]json.RawMessage
	text      strings.Builder
	thinking  strings.Builder
	input     strings.Builder
	signature string
	citations []json.RawMessage
}

func streamBlocks(body []byte) ([]json.RawMessage, error) {
	r := sse.NewReader(bytes.NewReader(body))
	blocks := map[int]*streamBlock{}
	var order []int
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		var e struct {
			Type         string                     `json:"type"`
			Index        int                        `json:"index"`
			ContentBlock map[string]json.RawMessage `json:"content_block"`
			Delta        struct {
				Type        string          `json:"type"`
				Text        string          `json:"text"`
				Thinking    string          `json:"thinking"`
				Signature   string          `json:"signature"`
				PartialJSON string          `json:"partial_json"`
				Citation    json.RawMessage `json:"citation"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(ev.Data), &e); err != nil {
			return nil, err
		}
		switch e.Type {
		case "content_block_start":
			blocks[e.Index] = &streamBlock{fields: e.ContentBlock}
			order = append(order, e.Index)
		case "content_block_delta":
			b := blocks[e.Index]
			b.text.WriteString(e.Delta.Text)
			b.thinking.WriteString(e.Delta.Thinking)
			b.input.WriteString(e.Delta.PartialJSON)
			if e.Delta.Signature != "" {
				b.signature = e.Delta.Signature
			}
			if len(e.Delta.Citation) > 0 {
				b.citations = append(b.citations, e.Delta.Citation)
			}
		}
	}
	var out []json.RawMessage
	for _, i := range order {
		raw, err := b2json(blocks[i])
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return out, nil
}

func marshalRaw(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSpace(buf.Bytes()), nil
}

func b2json(b *streamBlock) (json.RawMessage, error) {
	var kind string
	if err := json.Unmarshal(b.fields["type"], &kind); err != nil {
		return nil, err
	}
	switch kind {
	case "text":
		return marshalRaw(struct {
			Type      string            `json:"type"`
			Text      string            `json:"text"`
			Citations []json.RawMessage `json:"citations,omitempty"`
		}{kind, b.text.String(), b.citations})
	case "thinking":
		var err error
		if b.fields["thinking"], err = marshalRaw(b.thinking.String()); err != nil {
			return nil, err
		}
		if b.fields["signature"], err = marshalRaw(b.signature); err != nil {
			return nil, err
		}
	case "tool_use", "server_tool_use", "mcp_tool_use":
		if b.input.Len() > 0 {
			b.fields["input"] = json.RawMessage(b.input.String())
		}
	}
	return marshalRaw(b.fields)
}

func callIDs(body []byte) ([]string, []string, error) {
	var req struct {
		Messages []struct {
			Content []struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, m := range req.Messages {
		for _, b := range m.Content {
			switch b.Type {
			case "tool_use":
				calls = append(calls, b.ID)
			case "tool_result":
				results = append(results, b.ToolUseID)
			}
		}
	}
	return calls, results, nil
}

func TestLanguageModel_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel:    newLanguageModel,
		CallIDValid: callIDPattern.MatchString,
		Wire:        providertest.Wire{Request: requestBlocks, Response: responseBlocks, StreamResponse: streamBlocks, CallIDs: callIDs},
	}, providertest.Config{
		Dir:    "testdata/contract",
		Replay: "replay",
		Schema: &providertest.SchemaCase{
			Name:   "schema",
			Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1,"format":"phone"},"b":{"type":"array","items":{"type":"integer","minimum":0},"minItems":3,"maxItems":9},"c":{"type":"object","additionalProperties":{"type":"string"}}},"required":["a"]}`),
			WantWarnings: []string{
				`"minLength"`, `format "phone"`, `"minimum"`, `"maxItems"`, `"minItems"`,
				"additionalProperties at the root was set to false",
				"additionalProperties at /properties/c was changed from a schema to false",
			},
		},
		Cases: []providertest.Case{
			{
				Name:       "thinking_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"reasoning", "reasoning", "tool-call"},
				WantIDs:    []string{"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds", "Es8CCkYICxIMa7dEx0qk2Yh4PmZSGgyRq9kTn4Gx1LbcqZEiMN8s", "toolu_01A09q90qw90lq917835lq9", "msg_01Tool1"},
				WantUsage:  &ai.Usage{Input: 106, Output: 85, CacheRead: 64, CacheWrite: 30, Reasoning: 70},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "omitted_thinking",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"reasoning", "reasoning", "text"},
				WantIDs:    []string{"Es8CCkYICxIMa7dEx0qk2Yh4PmZSGgyRq9kTn4Gx1LbcqZEiMN8s", "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP"},
				WantUsage:  &ai.Usage{Input: 20, Output: 40, Reasoning: 33},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "citations",
				Request:    ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "text/plain", Data: []byte("The grass is green."), Citations: true}, ai.Text{Text: "colors?"})}},
				WantKinds:  []string{"text", "text", "text", "text", "text", "text"},
				WantIDs:    []string{"char_location", "page_location", "content_block_location", "search_result_location"},
				WantFinish: ai.FinishStop,
			},
			{
				Name:    "server_tools",
				Request: ai.TextRequest{Messages: user, Tools: []ai.Tool{anthropic.WebSearch(anthropic.WebSearchOptions{})}},
				WantKinds: []string{
					"text", "tool-call", "tool-result", "tool-call", "tool-result", "tool-call", "tool-result",
					"tool-call", "tool-result", "tool-call", "tool-result", "text",
				},
				WantIDs:    []string{"srvtoolu_01WYG3ziw53XMcoyKL4XcZmE", "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5LTQ1NGUtODgxNC1kNjNjNTk1ZWI3Y", "srvtoolu_03Code", "mcptoolu_014Q35RayjACSWkSj4X2yov1", "Eo8BCioIAhgBIiQyYjQ0OWJmZi1lNm..", "container_011CZmPrtWd9XpVk3nZq8Lfg"},
				WantUsage:  &ai.Usage{Input: 6039, Output: 931},
				WantFinish: ai.FinishStop,
			},
			{
				Name:      "pause_turn",
				Request:   ai.TextRequest{Messages: user, Tools: []ai.Tool{anthropic.WebSearch(anthropic.WebSearchOptions{})}},
				WantKinds: []string{"text", "tool-call"},
				WantIDs:   []string{"srvtoolu_05Pause"},
			},
			{
				Name:       "mixed_tool_use",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather, anthropic.WebFetch(anthropic.WebFetchOptions{})}},
				WantKinds:  []string{"text", "tool-call", "tool-call"},
				WantIDs:    []string{"srvtoolu_01HxbWnMRmbWyMfUtJKC45rA", "toolu_01PjgRJLbXrXEMZwDNYLnBqk", "direct"},
				WantFinish: ai.FinishToolCalls,
			},
			{Name: "refusal", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishRefusal},
			{Name: "max_tokens", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishLength},
			{Name: "context_window", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishLength},
			{Name: "stop_sequence", Request: ai.TextRequest{Messages: user, StopSequences: []string{"three"}}, WantKinds: []string{"text"}, WantFinish: ai.FinishStop},
		},
	})
}
