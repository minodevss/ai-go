package openaicompat_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openaicompat"
)

func providerMap(t *testing.T, d ai.ProviderData, key string) map[string]json.RawMessage {
	t.Helper()
	got, ok := ai.ProviderValue[map[string]json.RawMessage](d, key)
	if !ok {
		t.Fatalf("provider data %v has no %q", d, key)
	}
	return got
}

func messagesOf(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var req struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	return req.Messages
}

func TestMistral_toolCallIDs(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	long := strings.Repeat("x", 70) + "/bad"
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.ToolCall{ID: long, Name: "f", Input: json.RawMessage(`{}`)},
			ai.ToolCall{ID: "abcDEF123", Name: "g", Input: json.RawMessage(`{}`)},
			ai.ToolCall{ID: "toolu_orphan", Name: "h", Input: json.RawMessage(`{}`)},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{
			ai.ToolResult{CallID: long, Name: "f", Output: ai.ToolOutput{Text: "r1"}},
			ai.ToolResult{CallID: "abcDEF123", Name: "g", Output: ai.ToolOutput{Text: "r2"}},
		}},
	}
	send := func() []map[string]any {
		if _, err := openaicompat.Mistral(openaicompat.WithBaseURL(srv.URL)).Model("m").Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		return messagesOf(t, srv.last(t).Body)
	}
	msgs := send()
	valid := regexp.MustCompile(`^[a-zA-Z0-9]{9}$`)
	var calls, results []string
	for _, m := range msgs {
		if m["role"] == "tool" {
			results = append(results, as[string](t, m["tool_call_id"]))
			if m["name"] == nil {
				t.Errorf("tool message %v has no name", m)
			}
		}
		if cs, ok := m["tool_calls"].([]any); ok {
			for _, c := range cs {
				calls = append(calls, as[string](t, as[map[string]any](t, c)["id"]))
			}
		}
	}
	for _, id := range append(append([]string{}, calls...), results...) {
		if !valid.MatchString(id) {
			t.Errorf("id %q is not 9 alphanumeric characters", id)
		}
	}
	if len(calls) != 3 || len(results) != 3 || calls[1] != "abcDEF123" {
		t.Errorf("calls = %v, results = %v, want 3 each with the valid id kept", calls, results)
	}
	if again := send(); !cmp.Equal(msgs, again) {
		t.Errorf("mapping is not stable:\n%v\n%v", msgs, again)
	}
}

const mistralThinking = `{"id":"m1","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":[
{"type":"thinking","thinking":[{"type":"text","text":"Let me think."}],"signature":"sig-1","closed":true},{"type":"text","text":"Calling."}],
"tool_calls":[{"id":"aB3dE5gH7","type":"function","function":{"name":"f","arguments":"{\"a\":1}"},"index":0}]}}],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}}`

func TestMistral_thinkingChunks(t *testing.T) {
	srv := newFakeServer(t, jsonReply(mistralThinking))
	model := openaicompat.Mistral(openaicompat.WithBaseURL(srv.URL)).Model("magistral-medium-latest")
	resp, err := model.Generate(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := len(resp.Content); got != 3 {
		t.Fatalf("content has %d parts, want reasoning, text, tool call", got)
	}
	if r, ok := resp.Content[0].(ai.Reasoning); !ok || r.Text != "Let me think." {
		t.Errorf("content[0] = %#v, want reasoning text", resp.Content[0])
	}
	if tx, ok := resp.Content[1].(ai.Text); !ok || tx.Text != "Calling." {
		t.Errorf("content[1] = %#v, want text", resp.Content[1])
	}
	if resp.FinishReason != ai.FinishToolCalls {
		t.Errorf("FinishReason = %q, want tool-calls", resp.FinishReason)
	}
	history := []ai.Message{ai.User("q"), {Role: ai.RoleAssistant, Content: resp.Content}, {Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "aB3dE5gH7", Name: "f", Output: ai.ToolOutput{Text: "done"}}}}}
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := messagesOf(t, srv.last(t).Body)[1]
	wantContent := `[{"type":"thinking","thinking":[{"type":"text","text":"Let me think."}],"signature":"sig-1","closed":true},{"type":"text","text":"Calling."}]`
	raw, err := json.Marshal(got["content"])
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	equalJSON(t, wantContent, raw)
}

func TestMistral_streamThinkingChunks(t *testing.T) {
	srv := newFakeServer(t, sseReply(
		`{"id":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"Let "}]}]}}]}`,
		`{"id":"m1","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"me think."}],"signature":"sig-1","closed":true},{"type":"text","text":"Hi"}]}}]}`,
		`{"id":"m1","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":3}}`,
	))
	model := openaicompat.Mistral(openaicompat.WithBaseURL(srv.URL)).Model("magistral-medium-latest")
	parts := streamAll(t, model, &ai.Call{Messages: hi()})
	want := []string{"ReasoningStart", "ReasoningDelta", "ReasoningDelta", "ReasoningEnd", "TextStart", "TextDelta", "TextDelta", "TextEnd", "StepEnd"}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Fatalf("part types mismatch (-want +got):\n%s", diff)
	}
	end, ok := parts[3].(ai.ReasoningEnd)
	if !ok {
		t.Fatalf("parts[3] = %T, want ReasoningEnd", parts[3])
	}
	chunk := providerMap(t, end.Provider, "mistral")["chunk"]
	equalJSON(t, `{"type":"thinking","thinking":[{"type":"text","text":"Let me think."}],"closed":true,"signature":"sig-1"}`, chunk)
	if r := srv.last(t); bodyOf(r.Body)["stream_options"] != nil {
		t.Errorf("request has stream_options, which Mistral does not document: %s", r.Body)
	}
}

func TestDeepSeek_reasoningReplayAndUsage(t *testing.T) {
	body := `{"id":"d1","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"reasoning_content":"need the tool",
	"tool_calls":[{"id":"call_00_a","type":"function","function":{"name":"f","arguments":"{}"}}]}}],
	"usage":{"prompt_tokens":100,"completion_tokens":30,"prompt_cache_hit_tokens":64,"prompt_cache_miss_tokens":36,"completion_tokens_details":{"reasoning_tokens":20}}}`
	srv := newFakeServer(t, jsonReply(body))
	model := openaicompat.DeepSeek(openaicompat.WithBaseURL(srv.URL)).Model("deepseek-v4-pro")
	resp, err := model.Generate(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(ai.Usage{Input: 100, Output: 30, CacheRead: 64, Reasoning: 20}, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: resp.Content},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "call_00_a", Name: "f", Output: ai.ToolOutput{Text: "ok"}}}},
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "plain answer"}}},
		ai.User("again"),
	}
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	msgs := messagesOf(t, srv.last(t).Body)
	if msgs[1]["reasoning_content"] != "need the tool" {
		t.Errorf("tool-call turn = %v, want reasoning_content replayed", msgs[1])
	}
	if msgs[3]["reasoning_content"] != nil {
		t.Errorf("plain turn = %v, want no reasoning_content", msgs[3])
	}
}

func TestDeepSeek_warnings(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	model := openaicompat.DeepSeek(openaicompat.WithBaseURL(srv.URL)).Model("m")
	resp, err := model.Generate(t.Context(), &ai.Call{
		Messages: hi(), Seed: new(3),
		Output: &ai.OutputSpec{Name: "o", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	var texts []string
	for _, w := range resp.Warnings {
		texts = append(texts, w.Message)
	}
	joined := strings.Join(texts, "\n")
	for _, want := range []string{"seed", "no JSON Schema mode"} {
		if !strings.Contains(joined, want) {
			t.Errorf("warnings %q do not mention %q", joined, want)
		}
	}
	req := bodyOf(srv.last(t).Body)
	if req["seed"] != nil || as[map[string]any](t, req["response_format"])["type"] != "json_object" {
		t.Errorf("request = %v, want no seed and json_object", req)
	}
	first := messagesOf(t, srv.last(t).Body)[0]
	if first["role"] != "system" || !strings.Contains(as[string](t, first["content"]), `"required":["a"]`) {
		t.Errorf("first message = %v, want a system message with the schema", first)
	}
}

const perplexityBody = `{"id":"p1","model":"sonar-pro","citations":["https://a.test/x","https://b.test/y"],
"search_results":[{"title":"A","url":"https://a.test/x","date":"2025-01-02","last_updated":null,"snippet":"s","source":"web"},{"title":"B","url":"https://b.test/y","date":null,"last_updated":null,"snippet":"t","source":"web"}],
"related_questions":["more?"],
"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Answer [1][2]."}}],
"usage":{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16,"citation_tokens":3,"num_search_queries":2,"cost":{"input_tokens_cost":0.001,"output_tokens_cost":0.002,"total_cost":0.003}}}`

func checkPerplexity(t *testing.T, content ai.Parts, data ai.ProviderData, usage ai.Usage) {
	t.Helper()
	text, ok := content[0].(ai.Text)
	if !ok || text.Text != "Answer [1][2]." {
		t.Fatalf("content[0] = %#v, want the answer text", content[0])
	}
	want := []ai.Citation{{Kind: "url", URL: "https://a.test/x", Title: "A"}, {Kind: "url", URL: "https://b.test/y", Title: "B"}}
	if diff := cmp.Diff(want, text.Citations); diff != "" {
		t.Errorf("citations mismatch (-want +got):\n%s", diff)
	}
	fields := providerMap(t, data, "perplexity")
	if !strings.Contains(string(fields["searchResults"]), `"snippet":"s"`) || string(fields["relatedQuestions"]) != `["more?"]` {
		t.Errorf("provider data = %v, want search results and related questions kept", fields)
	}
	if usage.Input != 7 || usage.Output != 9 || !strings.Contains(string(usage.Raw), `"num_search_queries":2`) {
		t.Errorf("usage = %+v, want 7/9 with the raw usage kept", usage)
	}
}

func TestPerplexity_citationsGenerate(t *testing.T) {
	srv := newFakeServer(t, jsonReply(perplexityBody))
	resp, err := openaicompat.Perplexity(openaicompat.WithBaseURL(srv.URL)).Model("sonar-pro").Generate(t.Context(), &ai.Call{Messages: hi(), Tools: []ai.Tool{{Name: "f"}}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	checkPerplexity(t, resp.Content, resp.Provider, resp.Usage)
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "tools") {
		t.Errorf("warnings = %v, want one about tools", resp.Warnings)
	}
	if srv.last(t).Path != "/v1/sonar" {
		t.Errorf("path = %s, want /v1/sonar", srv.last(t).Path)
	}
}

func TestPerplexity_citationsStream(t *testing.T) {
	cites := `"citations":["https://a.test/x","https://b.test/y"],"search_results":[{"title":"A","url":"https://a.test/x"},{"title":"B","url":"https://b.test/y"}]`
	srv := newFakeServer(t, sseReply(
		`{"id":"p1",`+cites+`,"choices":[{"index":0,"delta":{"role":"assistant","content":"Answer "}}]}`,
		`{"id":"p1",`+cites+`,"choices":[{"index":0,"delta":{"content":"[1][2]."},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":9}}`,
	))
	parts := streamAll(t, openaicompat.Perplexity(openaicompat.WithBaseURL(srv.URL)).Model("sonar"), &ai.Call{Messages: hi()})
	want := []string{"TextStart", "TextDelta", "TextDelta", "Citation", "Citation", "TextEnd", "StepEnd"}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Errorf("part types mismatch (-want +got):\n%s", diff)
	}
	if c, ok := parts[3].(ai.Citation); !ok || c.URL != "https://a.test/x" || c.Title != "A" {
		t.Errorf("parts[3] = %#v, want the first citation with its title", parts[3])
	}
}

const openRouterDetails = `[{"type":"reasoning.text","text":"First,","signature":"sig-a","id":"rs-1","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"ENC==","id":"rs-2","format":"anthropic-claude-v1","index":1}]`

func TestOpenRouter_reasoningDetailsRoundTrip(t *testing.T) {
	body := `{"id":"gen-1","provider":"Anthropic","model":"anthropic/claude-sonnet-4.5","choices":[{"index":0,"finish_reason":"tool_calls","native_finish_reason":"tool_use","message":{"role":"assistant","content":"","reasoning":"First,",
	"reasoning_details":` + openRouterDetails + `,"tool_calls":[{"id":"toolu_01A","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]}}],
	"usage":{"prompt_tokens":50,"completion_tokens":20,"cost":0.0012,"prompt_tokens_details":{"cached_tokens":10},"completion_tokens_details":{"reasoning_tokens":8}}}`
	srv := newFakeServer(t, jsonReply(body))
	model := openaicompat.OpenRouter(openaicompat.WithBaseURL(srv.URL)).Model("anthropic/claude-sonnet-4.5")
	resp, err := model.Generate(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if r, ok := resp.Content[0].(ai.Reasoning); !ok || r.Text != "First," {
		t.Fatalf("content[0] = %#v, want reasoning", resp.Content[0])
	}
	if got := providerMap(t, resp.Provider, "openrouter"); string(got["provider"]) != `"Anthropic"` {
		t.Errorf("response provider data = %v, want the serving provider", got)
	}
	raw, err := json.Marshal(resp.Content)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	var persisted ai.Parts
	if err := json.Unmarshal(raw, &persisted); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: persisted},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "toolu_01A", Name: "f", Output: ai.ToolOutput{Text: "ok"}}}},
	}
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	sent := messagesOf(t, srv.last(t).Body)[1]
	details, err := json.Marshal(sent["reasoning_details"])
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if string(details) != compactJSON(t, openRouterDetails) {
		t.Errorf("reasoning_details = %s, want the original array unchanged and in order", details)
	}
	if sent["reasoning"] != nil {
		t.Errorf("assistant message = %v, want reasoning_details to replace the reasoning string", sent)
	}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(ai.Usage{Input: 50, Output: 20, CacheRead: 10, Reasoning: 8}, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
}

func compactJSON(t *testing.T, s string) string {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	return string(out)
}

func TestOpenRouter_reasoningStringReplay(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	model := openaicompat.OpenRouter(openaicompat.WithBaseURL(srv.URL)).Model("m")
	history := []ai.Message{ai.User("q"), {Role: ai.RoleAssistant, Content: ai.Parts{
		ai.Reasoning{Text: "thought", Provider: ai.ProviderData{"openrouter": json.RawMessage(`{"field":"reasoning"}`)}}, ai.Text{Text: "a"},
	}}, ai.User("n")}
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := messagesOf(t, srv.last(t).Body)[1]; got["reasoning"] != "thought" {
		t.Errorf("assistant message = %v, want the reasoning string", got)
	}
}

func TestOpenRouter_streamReasoningDetails(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, ": OPENROUTER PROCESSING\n\n"+
			`data: {"id":"g1","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"First,","reasoning_details":[{"type":"reasoning.text","text":"First,","id":"rs-1","format":"anthropic-claude-v1","index":0}]}}]}`+"\n\n"+
			`data: {"id":"g1","choices":[{"index":0,"delta":{"content":"","reasoning":"","reasoning_details":[{"type":"reasoning.encrypted","data":"ENC==","id":"rs-2","format":"anthropic-claude-v1","index":1}]}}]}`+"\n\n"+
			`data: {"id":"g1","choices":[{"index":0,"delta":{"content":"Done."},"finish_reason":"stop"}]}`+"\n\n"+
			`data: {"id":"g1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"cost":0.001}}`+"\n\n"+
			"data: [DONE]\n\n")
	})
	res := ai.StreamText(t.Context(), openaicompat.OpenRouter(openaicompat.WithBaseURL(srv.URL)).Model("m"), ai.TextRequest{Messages: hi(), MaxSteps: 1})
	got, err := res.Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	reasoning, ok := got.Content[0].(ai.Reasoning)
	if !ok {
		t.Fatalf("content[0] = %#v, want reasoning", got.Content[0])
	}
	fields := providerMap(t, reasoning.Provider, "openrouter")
	var data struct {
		Fields map[string]json.RawMessage `json:"fields"`
	}
	raw, err := json.Marshal(map[string]any{"fields": fields["fields"]})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	want := `[{"type":"reasoning.text","text":"First,","id":"rs-1","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"ENC==","id":"rs-2","format":"anthropic-claude-v1","index":1}]`
	equalJSON(t, want, data.Fields["reasoning_details"])
	if got.Text != "Done." {
		t.Errorf("Text = %q, want Done.", got.Text)
	}
}

func TestOpenRouter_midStreamError(t *testing.T) {
	srv := newFakeServer(t, sseReply(
		`{"id":"g1","choices":[{"index":0,"delta":{"content":"par"}}]}`,
		`{"id":"g1","provider":"openai","error":{"code":502,"message":"Provider disconnected"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`,
	))
	resp, err := openaicompat.OpenRouter(openaicompat.WithBaseURL(srv.URL)).Model("m").Stream(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var streamErr error
	for _, err := range resp.Parts {
		streamErr = err
	}
	var apiErr *ai.APIError
	if !errors.As(streamErr, &apiErr) || apiErr.StatusCode != http.StatusBadGateway || !apiErr.Retryable() {
		t.Errorf("stream error = %v, want a retryable 502 APIError", streamErr)
	}
}

func TestXAI_usageAndCitations(t *testing.T) {
	body := `{"id":"x1","citations":["https://x.test/1"],"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","reasoning_content":"hmm"}}],
	"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":45,"num_sources_used":1,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":20}}}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := openaicompat.XAI(openaicompat.WithBaseURL(srv.URL)).Model("grok-4.7").Generate(t.Context(), &ai.Call{Messages: hi(), Reasoning: ai.ReasoningNone})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(ai.Usage{Input: 20, Output: 25, CacheRead: 4, Reasoning: 20}, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	text, ok := resp.Content[1].(ai.Text)
	if !ok || len(text.Citations) != 1 || text.Citations[0].URL != "https://x.test/1" {
		t.Errorf("content[1] = %#v, want text with the citation", resp.Content[1])
	}
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "cannot be disabled") {
		t.Errorf("warnings = %v, want the reasoning none warning", resp.Warnings)
	}
}

func TestTogether_eosAndWarnings(t *testing.T) {
	body := `{"id":"t1","warnings":[{"message":"seed is ignored"}],"choices":[{"index":0,"finish_reason":"eos","message":{"role":"assistant","content":"hi","reasoning":"r"}}],
	"usage":{"prompt_tokens":9,"completion_tokens":4,"cached_tokens":6,"completion_tokens_details":{"reasoning_tokens":2}}}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := openaicompat.Together(openaicompat.WithBaseURL(srv.URL)).Model("m").Generate(t.Context(), &ai.Call{Messages: hi(), Seed: new(1)})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if resp.FinishReason != ai.FinishStop || resp.FinishRaw != "eos" {
		t.Errorf("finish = %q/%q, want stop/eos", resp.FinishReason, resp.FinishRaw)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Message != "together: seed is ignored" {
		t.Errorf("warnings = %v, want the response warning", resp.Warnings)
	}
	if resp.Usage.CacheRead != 6 || resp.Usage.Reasoning != 2 {
		t.Errorf("usage = %+v, want cache 6 and reasoning 2", resp.Usage)
	}
	if bodyOf(srv.last(t).Body)["stream_options"] != nil {
		t.Error("request has stream_options")
	}
}

func TestGroq_penaltiesDropped(t *testing.T) {
	body := `{"id":"g1","x_groq":{"id":"req_1"},"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","reasoning":"because"}}]}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := openaicompat.Groq(openaicompat.WithBaseURL(srv.URL)).Model("m").Generate(t.Context(), &ai.Call{Messages: hi(), PresencePenalty: new(0.1), FrequencyPenalty: new(0.1)})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(resp.Warnings) != 2 {
		t.Errorf("warnings = %v, want two", resp.Warnings)
	}
	if r, ok := resp.Content[0].(ai.Reasoning); !ok || r.Text != "because" {
		t.Errorf("content[0] = %#v, want reasoning from message.reasoning", resp.Content[0])
	}
	if got := providerMap(t, resp.Provider, "groq"); string(got["xGroq"]) != `{"id":"req_1"}` {
		t.Errorf("provider data = %v, want x_groq kept", got)
	}
	if b := bodyOf(srv.last(t).Body); b["presence_penalty"] != nil || b["frequency_penalty"] != nil {
		t.Errorf("request = %v, want no penalties", b)
	}
}

func TestExtrasInProviderData(t *testing.T) {
	tests := []struct {
		name  string
		model func(url string) *openaicompat.LanguageModel
		extra string
		key   string
		field string
	}{
		{name: "azure", model: func(u string) *openaicompat.LanguageModel {
			return openaicompat.Azure("r", openaicompat.WithBaseURL(u)).Model("d")
		}, extra: `"prompt_filter_results":[{"prompt_index":0}]`, key: "azure", field: "promptFilterResults"},
		{name: "cerebras", model: func(u string) *openaicompat.LanguageModel {
			return openaicompat.Cerebras(openaicompat.WithBaseURL(u)).Model("m")
		}, extra: `"time_info":{"total_time":0.1}`, key: "cerebras", field: "timeInfo"},
		{name: "fireworks", model: func(u string) *openaicompat.LanguageModel {
			return openaicompat.Fireworks(openaicompat.WithBaseURL(u)).Model("m")
		}, extra: `"perf_metrics":{"x":1}`, key: "fireworks", field: "perfMetrics"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"id":"c1",` + tt.extra + `,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
			srv := newFakeServer(t, jsonReply(body))
			resp, err := tt.model(srv.URL).Generate(t.Context(), &ai.Call{Messages: hi()})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got := providerMap(t, resp.Provider, tt.key); got[tt.field] == nil {
				t.Errorf("provider data = %v, want %s", got, tt.field)
			}
		})
	}
}

func TestVLLM_reasoning(t *testing.T) {
	body := `{"id":"v1","choices":[{"index":0,"message":{"role":"assistant","content":"4","reasoning":"2+2"},"finish_reason":"stop"}]}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := openaicompat.VLLM(srv.URL).Model("m").Generate(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if r, ok := resp.Content[0].(ai.Reasoning); !ok || r.Text != "2+2" {
		t.Errorf("content[0] = %#v, want reasoning from message.reasoning", resp.Content[0])
	}
}

func TestAzure_embeddingBatchLimit(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[]}`))
	model := openaicompat.Azure("r", openaicompat.WithBaseURL(srv.URL)).EmbeddingModel("emb")
	if model.MaxBatch() != 2048 {
		t.Errorf("MaxBatch() = %d, want 2048", model.MaxBatch())
	}
}

func TestMistral_embeddingDimensions(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":1}}`))
	model := openaicompat.Mistral(openaicompat.WithBaseURL(srv.URL)).EmbeddingModel("mistral-embed")
	if _, err := ai.Embed(t.Context(), model, ai.EmbedRequest{Texts: []string{"a"}, Dimensions: 256}); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"mistral-embed","input":["a"],"output_dimension":256,"encoding_format":"float"}`, srv.last(t).Body)
}
