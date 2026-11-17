package bedrock_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

var callIDPattern = regexp.MustCompile(`^[A-Za-z0-9_.:-]{1,64}$`)

type framingTransport struct {
	base http.RoundTripper
}

func (f framingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := f.base.RoundTrip(req)
	if err != nil || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/vnd.amazon.eventstream") {
		return resp, err
	}
	raw, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}
	var recorded struct {
		Events []event `json:"events"`
	}
	if err := json.Unmarshal(raw, &recorded); err != nil {
		return nil, err
	}
	frames, err := encodeEventsErr(recorded.Events)
	if err != nil {
		return nil, err
	}
	resp.Body = io.NopCloser(bytes.NewReader(frames))
	resp.ContentLength = int64(len(frames))
	return resp, nil
}

func newLanguageModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	p := bedrock.New(
		bedrock.WithRegion("us-east-1"),
		bedrock.WithAPIKey("test-key"),
		bedrock.WithHTTPClient(&http.Client{Transport: framingTransport{base: client.Transport}}),
	)
	return p.Model(testModel)
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
		Output struct {
			Message struct {
				Content []json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return resp.Output.Message.Content, nil
}

func callIDs(body []byte) ([]string, []string, error) {
	blocks, err := requestBlocks(body)
	if err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, raw := range blocks {
		var b struct {
			ToolUse *struct {
				ID string `json:"toolUseId"`
			} `json:"toolUse"`
			ToolResult *struct {
				ID string `json:"toolUseId"`
			} `json:"toolResult"`
		}
		if err := json.Unmarshal(raw, &b); err != nil {
			return nil, nil, err
		}
		if b.ToolUse != nil {
			calls = append(calls, b.ToolUse.ID)
		}
		if b.ToolResult != nil {
			results = append(results, b.ToolResult.ID)
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
				WantIDs:    []string{"EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds", "RW5jcnlwdGVkIHJlYXNvbmluZyB0aGF0IHRoZSBtb2RlbCBwcm92aWRlciBoaWQ=", "tooluse_vN7jKq2fR8a1Zp0Xb3Tg5w"},
				WantUsage:  &ai.Usage{Input: 114, Output: 85, CacheRead: 64, CacheWrite: 30},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "text_cached",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantUsage:  &ai.Usage{Input: 2062, Output: 12, CacheRead: 2048},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "citations",
				Request:    ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "text/plain", Filename: "notes.txt", Data: []byte("The grass is green."), Citations: true}, ai.Text{Text: "colors?"})}},
				WantKinds:  []string{"text", "text"},
				WantIDs:    []string{"documentChar", "notes.txt"},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "server_tools",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text", "tool-call", "tool-result", "text"},
				WantIDs:    []string{"srvtooluse_01Search"},
				WantUsage:  &ai.Usage{Input: 300, Output: 120},
				WantFinish: ai.FinishStop,
			},
			{Name: "max_tokens", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishLength},
			{Name: "context_window", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishLength},
			{Name: "stop_sequence", Request: ai.TextRequest{Messages: user, StopSequences: []string{"three"}}, WantKinds: []string{"text"}, WantFinish: ai.FinishStop},
			{Name: "guardrail", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishContentFilter},
			{Name: "content_filtered", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishContentFilter},
			{Name: "malformed_tool_use", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishError},
			{Name: "malformed_model_output", Request: ai.TextRequest{Messages: user}, WantKinds: []string{"text"}, WantFinish: ai.FinishError},
		},
	})
}
