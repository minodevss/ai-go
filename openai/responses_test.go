package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

type filterInput struct {
	Query string `json:"query"`
	Limit int    `json:"limit,omitzero"`
}

func TestResponsesModel_request(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{
			name: "settings and options",
			call: ai.Call{
				Instructions:    ai.Texts("Be brief.", "Use Korean."),
				Messages:        []ai.Message{ai.User("hi")},
				MaxOutputTokens: 100,
				Temperature:     new(0.5),
				TopP:            new(0.9),
				Reasoning:       ai.ReasoningMax,
				Provider: ai.Providers(openai.Options{
					Store:                new(false),
					PreviousResponseID:   "resp_prev",
					ReasoningSummary:     "auto",
					ServiceTier:          "flex",
					PromptCacheKey:       "k1",
					PromptCacheRetention: "24h",
					ParallelToolCalls:    new(false),
					SafetyIdentifier:     "u1",
					Metadata:             map[string]string{"a": "b"},
					Truncation:           "auto",
					MaxToolCalls:         3,
					TextVerbosity:        "low",
					Include:              []string{"web_search_call.action.sources"},
				}),
			},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"instructions":"Be brief.\n\nUse Korean.",
			"max_output_tokens":100,"temperature":0.5,"top_p":0.9,"reasoning":{"effort":"max","summary":"auto"},"text":{"verbosity":"low"},
			"include":["web_search_call.action.sources","reasoning.encrypted_content"],"store":false,"previous_response_id":"resp_prev","service_tier":"flex",
			"prompt_cache_key":"k1","prompt_cache_retention":"24h","parallel_tool_calls":false,"safety_identifier":"u1","metadata":{"a":"b"},"truncation":"auto","max_tool_calls":3}`,
		},
		{
			name: "store true does not request encrypted reasoning",
			call: ai.Call{Messages: []ai.Message{ai.User("hi")}, Provider: ai.Providers(openai.Options{Store: new(true)})},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"store":true}`,
		},
		{
			name: "files",
			call: ai.Call{Messages: []ai.Message{ai.UserParts(
				ai.Text{Text: "look"},
				ai.File{MediaType: "image/png", Data: png},
				ai.File{MediaType: "image/jpeg", URL: "https://x.test/a.jpg"},
				ai.File{MediaType: "image/png", Provider: openai.FileID("file-img")},
				ai.File{MediaType: "application/pdf", Data: []byte("pdf"), Filename: "a.pdf"},
				ai.File{MediaType: "application/pdf", URL: "https://x.test/a.pdf"},
				ai.File{MediaType: "application/pdf", Provider: openai.FileID("file-pdf")},
			)}},
			want: `{"model":"m","input":[{"role":"user","content":[
				{"type":"input_text","text":"look"},
				{"type":"input_image","image_url":"data:image/png;base64,iVBORw=="},
				{"type":"input_image","image_url":"https://x.test/a.jpg"},
				{"type":"input_image","file_id":"file-img"},
				{"type":"input_file","file_data":"data:application/pdf;base64,cGRm","filename":"a.pdf"},
				{"type":"input_file","file_url":"https://x.test/a.pdf"},
				{"type":"input_file","file_id":"file-pdf"}]}]}`,
		},
		{
			name: "function tools",
			call: ai.Call{
				Messages: []ai.Message{ai.User("hi")},
				Tools: []ai.Tool{
					ai.NewClientTool[filterInput]("search", "Search things"),
					{Name: "strict", Description: "d", Strict: true, InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1}},"required":["a"],"additionalProperties":false}`)},
				},
				ToolChoice: ai.ToolNamed("search"),
			},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
			"tools":[{"type":"function","name":"search","description":"Search things","strict":false,"parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"],"additionalProperties":false}},
			{"type":"function","name":"strict","description":"d","strict":true,"parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}],
			"tool_choice":{"type":"function","name":"search"}}`,
		},
		{
			name: "hosted tools",
			call: ai.Call{
				Messages:   []ai.Message{ai.User("hi")},
				Tools:      []ai.Tool{openai.WebSearch(openai.WebSearchOptions{SearchContextSize: "low"}), openai.FileSearch("vs_1"), openai.CodeInterpreter(), openai.ImageGeneration(), openai.MCP(openai.MCPOptions{Label: "docs", URL: "https://mcp.test", RequireApproval: "never"})},
				ToolChoice: ai.ToolNamed("web_search"),
			},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
			"tools":[{"type":"web_search","search_context_size":"low"},{"type":"file_search","vector_store_ids":["vs_1"]},{"type":"code_interpreter","container":{"type":"auto"}},{"type":"image_generation"},
			{"type":"mcp","server_label":"docs","server_url":"https://mcp.test","require_approval":"never"}],"tool_choice":{"type":"web_search"}}`,
		},
		{
			name: "tool choice modes",
			call: ai.Call{Messages: []ai.Message{ai.User("hi")}, ToolChoice: ai.ToolRequired()},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"tool_choice":"required"}`,
		},
		{
			name: "structured output",
			call: ai.Call{
				Messages: []ai.Message{ai.User("hi")},
				Output:   &ai.OutputSpec{Name: "my out!", Description: "d", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)},
			},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
			"text":{"format":{"type":"json_schema","name":"myout","description":"d","strict":true,"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}}}`,
		},
		{
			name: "tool history",
			call: ai.Call{Messages: []ai.Message{
				ai.User("q"),
				{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "call_1", Name: "search", Input: json.RawMessage(`{"query":"x"}`)}, ai.ToolCall{ID: "call_2", Name: "other"}}},
				{Role: ai.RoleTool, Content: ai.Parts{
					ai.ToolResult{CallID: "call_1", Name: "search", Output: ai.ToolOutput{JSON: json.RawMessage(`{"n":1}`)}},
					ai.ToolResult{CallID: "call_2", Name: "other", Output: ai.ToolOutput{Parts: ai.Parts{ai.Text{Text: "t"}, ai.File{MediaType: "image/png", URL: "https://x.test/i.png"}}}},
				}},
				ai.Assistant("done"),
			}},
			want: `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"q"}]},
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"x\"}"},
			{"type":"function_call","call_id":"call_2","name":"other","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"{\"n\":1}"},
			{"type":"function_call_output","call_id":"call_2","output":[{"type":"input_text","text":"t"},{"type":"input_image","image_url":"https://x.test/i.png"}]},
			{"role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalResponse))
			if _, err := srv.provider().Model("m").Generate(t.Context(), &tt.call); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			equalJSON(t, tt.want, srv.last(t).Body)
		})
	}
}

func TestResponsesModel_requestErrors(t *testing.T) {
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{name: "audio", call: ai.Call{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "audio/wav", Data: []byte("x")})}}, want: "audio"},
		{name: "empty file", call: ai.Call{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "application/pdf"})}}, want: "no data"},
		{name: "bad role", call: ai.Call{Messages: []ai.Message{{Role: "system"}}}, want: "unknown message role"},
		{name: "bad options", call: ai.Call{Messages: []ai.Message{ai.User("x")}, Provider: ai.ProviderData{"openai": json.RawMessage(`{"store":"yes"}`)}}, want: "provider options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalResponse))
			_, err := srv.provider().Model("m").Generate(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) || !strings.HasPrefix(err.Error(), "openai:") {
				t.Errorf("Generate() error = %v, want openai: error containing %q", err, tt.want)
			}
			if srv.count() != 0 {
				t.Errorf("sent %d requests, want 0", srv.count())
			}
		})
	}
}

func TestResponsesModel_warnings(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	resp, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{
		Messages:         []ai.Message{ai.User("hi")},
		TopK:             5,
		Seed:             new(1),
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.1),
		StopSequences:    []string{"x"},
		Tools:            []ai.Tool{{Name: "t", Strict: true, InputSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","maxLength":3}}}`)}},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	var got []string
	for _, w := range resp.Warnings {
		got = append(got, w.Message)
	}
	joined := strings.Join(got, "\n")
	for _, want := range []string{"topK", "seed", "presencePenalty", "frequencyPenalty", "stopSequences", `"maxLength"`, `tool "t" is not strict`} {
		if !strings.Contains(joined, want) {
			t.Errorf("warnings do not mention %q:\n%s", want, joined)
		}
	}
}

func TestResponsesModel_headers(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	p := srv.provider(openai.WithOrganization("org-1"), openai.WithProject("proj-1"), openai.WithHeaders(map[string][]string{"X-Team": {"a"}, "x-override": {"provider"}}))
	call := &ai.Call{Messages: []ai.Message{ai.User("hi")}, Headers: map[string][]string{"X-Override": {"call"}}}
	if _, err := p.Model("m").Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	h := srv.last(t).Header
	for key, want := range map[string]string{"Authorization": "Bearer test-key", "Openai-Organization": "org-1", "Openai-Project": "proj-1", "X-Team": "a", "X-Override": "call", "Content-Type": "application/json"} {
		if got := h.Get(key); got != want {
			t.Errorf("header %s = %q, want %q", key, got, want)
		}
	}
}

func TestModel_environment(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", srv.URL+"/v1/")
	res, err := ai.GenerateText(t.Context(), openai.Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Text != "hi" {
		t.Errorf("Text = %q, want hi", res.Text)
	}
	got := srv.last(t)
	if got.Path != "/v1/responses" || got.Header.Get("Authorization") != "Bearer env-key" {
		t.Errorf("request = %s with auth %q, want /v1/responses with env key", got.Path, got.Header.Get("Authorization"))
	}
}

type countingLimiter struct{ n int }

func (l *countingLimiter) Wait(context.Context) error {
	l.n++
	return nil
}

func TestProvider_limiter(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	lim := &countingLimiter{}
	p := srv.provider(openai.WithLimiter(lim))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if lim.n != 2 {
		t.Errorf("limiter waited %d times, want 2", lim.n)
	}
}

func TestProvider_limiterError(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	p := srv.provider(openai.WithLimiter(failingLimiter{}))
	_, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if !errors.Is(err, errLimited) {
		t.Errorf("Generate() error = %v, want %v", err, errLimited)
	}
	if srv.count() != 0 {
		t.Errorf("sent %d requests, want 0", srv.count())
	}
}

var errLimited = errors.New("limited")

type failingLimiter struct{}

func (failingLimiter) Wait(context.Context) error { return errLimited }

func TestResponsesModel_response(t *testing.T) {
	body := `{"id":"resp_9","status":"completed","service_tier":"flex","output":[
		{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"a\":1}","call_id":"call_1","name":"f"},
		{"id":"fc_2","type":"function_call","status":"completed","arguments":"{\"a\":","call_id":"call_2","name":"g"},
		{"id":"fc_3","type":"function_call","status":"completed","arguments":"","call_id":"call_3","name":"h"},
		{"id":"cu_1","type":"computer_call","status":"completed"}],
		"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":6,"output_tokens_details":{"reasoning_tokens":2}}}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	var calls []string
	for _, p := range resp.Content {
		if c, ok := p.(ai.ToolCall); ok {
			calls = append(calls, c.ID+"="+string(c.Input))
		}
	}
	wantCalls := []string{"call_1={\"a\":1}", "call_2=\"{\\\"a\\\":\"", "call_3={}"}
	if diff := cmp.Diff(wantCalls, calls); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}
	custom, ok := resp.Content[3].(ai.Custom)
	if !ok || custom.Kind != "openai.computer_call" {
		t.Errorf("last part = %#v, want openai.computer_call custom part", resp.Content[3])
	}
	wantUsage := ai.Usage{Input: 10, Output: 6, CacheRead: 4, Reasoning: 2}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(wantUsage, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	data, ok := ai.ProviderValue[openai.ResponseData](resp.Provider, "openai")
	if !ok || data.ID != "resp_9" || data.ServiceTier != "flex" {
		t.Errorf("response data = %+v, %v; want id resp_9 and tier flex", data, ok)
	}
	if resp.FinishReason != ai.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishToolCalls)
	}
}

func TestResponsesModel_failedResponse(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"r","status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}`))
	_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.Code != "server_error" || !apiErr.Retryable() {
		t.Errorf("Generate() error = %v, want retryable server_error APIError", err)
	}
}

func TestResponsesModel_replaysRawItems(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalResponse))
	rawReasoning := `{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"ENC","status":"completed"}`
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Reasoning{Text: "edited", Provider: ai.ProviderData{"openai": json.RawMessage(`{"id":"rs_1","item":` + rawReasoning + `}`)}},
			ai.Text{Text: "split", Provider: ai.ProviderData{"openai": json.RawMessage(`{"id":"msg_1","index":1}`)}},
		}},
	}
	if _, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	equalJSON(t, `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"q"}]},`+rawReasoning+`]}`, srv.last(t).Body)
}

func TestProvider_LanguageModel(t *testing.T) {
	m, err := openai.New().LanguageModel("gpt-5-mini")
	if err != nil {
		t.Fatalf("LanguageModel() error = %v", err)
	}
	if m.Provider() != "openai" || m.ModelID() != "gpt-5-mini" {
		t.Errorf("model = %s/%s, want openai/gpt-5-mini", m.Provider(), m.ModelID())
	}
	reg := ai.Registry{Providers: map[string]ai.Provider{"openai": openai.New()}}
	if _, err := reg.LanguageModel("openai:gpt-5-mini"); err != nil {
		t.Errorf("Registry.LanguageModel() error = %v", err)
	}
}
