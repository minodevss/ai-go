package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ollama"
)

func TestChatModel_request(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalReply))
	m := srv.provider().Model("llama3.2")
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`)
	call := &ai.Call{
		Instructions:    ai.Texts("Be brief.", "Answer in English."),
		Messages:        []ai.Message{ai.UserParts(ai.Text{Text: "what is this?"}, ai.File{MediaType: "image/png", Data: []byte("img")})},
		Tools:           []ai.Tool{ai.NewClientTool[weatherInput]("get_weather", "Weather by city")},
		Output:          &ai.OutputSpec{Name: "out", Schema: schema},
		MaxOutputTokens: 64,
		Temperature:     new(0.2),
		TopP:            new(0.9),
		TopK:            40,
		Seed:            new(7),
		StopSequences:   []string{"END"},
		Reasoning:       ai.ReasoningNone,
		Provider: ai.Providers(ollama.Options{
			NumCtx:     8192,
			KeepAlive:  "10m",
			MinP:       new(0.05),
			Parameters: map[string]any{"repeat_last_n": 64, "num_ctx": 1},
		}),
	}
	if _, err := m.Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := srv.last(t)
	if got.Path != "/api/chat" {
		t.Errorf("path = %s, want /api/chat", got.Path)
	}
	if v := got.Header.Get("Accept"); v != "application/json" {
		t.Errorf("Accept = %q, want application/json", v)
	}
	equalJSON(t, `{
		"model":"llama3.2",
		"stream":false,
		"think":false,
		"keep_alive":"10m",
		"messages":[
			{"role":"system","content":"Be brief.\n\nAnswer in English."},
			{"role":"user","content":"what is this?","images":["aW1n"]}
		],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"Weather by city","parameters":`+string(toolSchema(t))+`}}],
		"format":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]},
		"options":{"temperature":0.2,"top_p":0.9,"top_k":40,"seed":7,"stop":["END"],"num_predict":64,"num_ctx":8192,"min_p":0.05,"repeat_last_n":64}
	}`, got.Body)
}

func toolSchema(t *testing.T) json.RawMessage {
	t.Helper()
	return ai.NewClientTool[weatherInput]("x", "").InputSchema
}

func TestChatModel_thinkAndFormat(t *testing.T) {
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{name: "reasoning on", call: ai.Call{Reasoning: ai.ReasoningHigh}, want: `"think":true`},
		{name: "level option", call: ai.Call{Reasoning: ai.ReasoningHigh, Provider: ai.Providers(ollama.Options{Think: "low"})}, want: `"think":"low"`},
		{name: "json mode", call: ai.Call{Output: &ai.OutputSpec{Name: "o"}}, want: `"format":"json"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalReply))
			tt.call.Messages = []ai.Message{ai.User("x")}
			if _, err := srv.provider().Model("m").Generate(t.Context(), &tt.call); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if body := string(srv.last(t).Body); !strings.Contains(body, tt.want) {
				t.Errorf("body = %s, want it to contain %s", body, tt.want)
			}
		})
	}
}

func TestChatModel_replayMessages(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalReply))
	call := json.RawMessage(`{"function":{"index":0,"name":"get_weather","arguments":{"city":"Seoul"}}}`)
	history := []ai.Message{
		ai.User("weather?"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Reasoning{Text: "think", Provider: ai.ProviderData{"ollama": json.RawMessage(`{}`)}},
			ai.ToolCall{ID: "c1", Name: "get_weather", Input: json.RawMessage(`{"city":"Seoul"}`), Provider: ai.ProviderData{"ollama": json.RawMessage(`{"call":` + string(call) + `}`)}},
			ai.ToolCall{ID: "c2", Name: "get_time", Input: json.RawMessage(`{}`)},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "c1", Name: "get_weather", Output: ai.ToolOutput{JSON: json.RawMessage(`{"temp":21}`)}}}},
		ai.User("and?"),
	}
	if _, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	equalJSON(t, `{"model":"m","stream":false,"messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":"","thinking":"think","tool_calls":[
			{"function":{"index":0,"name":"get_weather","arguments":{"city":"Seoul"}}},
			{"function":{"name":"get_time","arguments":{}}}
		]},
		{"role":"tool","content":"{\"temp\":21}","tool_name":"get_weather"},
		{"role":"tool","content":"error: this tool call has no result","tool_name":"get_time"},
		{"role":"user","content":"and?"}
	]}`, srv.last(t).Body)
}

func TestChatModel_generateResponse(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"model":"m","created_at":"t","message":{"role":"assistant","content":"x"},"done":true,"done_reason":"load","total_duration":1000,"prompt_eval_count":10,"prompt_eval_cached_count":4,"eval_count":3}`))
	resp, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("x")}, ToolChoice: ai.ToolRequired(), PresencePenalty: new(0.1)})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if resp.FinishReason != ai.FinishOther || resp.FinishRaw != "load" {
		t.Errorf("finish = %q (%q), want other (load)", resp.FinishReason, resp.FinishRaw)
	}
	want := ai.Usage{Input: 10, Output: 3, CacheRead: 4}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	if len(resp.Warnings) != 2 {
		t.Errorf("warnings = %v, want tool choice and presence penalty warnings", resp.Warnings)
	}
	data, ok := ai.ProviderValue[ollama.ResponseData](resp.Provider, "ollama")
	if !ok || data.Model != "m" || data.TotalDuration != time.Microsecond {
		t.Errorf("provider data = %+v, %v", data, ok)
	}
}

func TestChatModel_streamHeadersAndErrors(t *testing.T) {
	srv := newFakeServer(t, ndjsonReply(
		`{"model":"m","created_at":"t","message":{"role":"assistant","content":"par"},"done":false}`,
		``,
		`{"error":"an error was encountered while running the model"}`,
	))
	stream := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("x")}, MaxRetries: new(0)})
	for range stream.Parts() {
	}
	_, err := stream.Result()
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || !strings.Contains(apiErr.Message, "while running the model") {
		t.Fatalf("Result() error = %v, want the stream error", err)
	}
	if v := srv.last(t).Header.Get("Accept"); v != "application/x-ndjson" {
		t.Errorf("Accept = %q, want application/x-ndjson", v)
	}
}

func TestChatModel_streamTruncated(t *testing.T) {
	srv := newFakeServer(t, ndjsonReply(`{"model":"m","created_at":"t","message":{"role":"assistant","content":"par"},"done":false}`))
	stream := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("x")}, MaxRetries: new(0)})
	for range stream.Parts() {
	}
	if _, err := stream.Result(); err == nil || !strings.Contains(err.Error(), "ended before") {
		t.Errorf("Result() error = %v, want a truncated stream error", err)
	}
}

func TestChatModel_inputErrors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalReply))
	tests := []struct {
		name string
		part ai.Part
		want string
	}{
		{name: "pdf", part: ai.File{MediaType: "application/pdf", Data: []byte("x")}, want: "only images"},
		{name: "image url", part: ai.File{MediaType: "image/png", URL: "https://example.com/a.png"}, want: "URLs are not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.UserParts(tt.part)}})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Generate() error = %v, want %q", err, tt.want)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("sent %d requests, want 0", srv.count())
	}
}

func TestChatModel_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		body          string
		wantIs        error
		wantRetryable bool
		wantMessage   string
	}{
		{name: "model not found", status: http.StatusNotFound, body: `{"error":"model 'x' not found"}`, wantMessage: "model 'x' not found"},
		{name: "rate limit", status: http.StatusTooManyRequests, body: `{"error":"too many requests"}`, wantIs: ai.ErrRateLimit, wantRetryable: true, wantMessage: "too many requests"},
		{name: "unauthorized", status: http.StatusUnauthorized, body: `{"error":"unauthorized"}`, wantIs: ai.ErrAuth, wantMessage: "unauthorized"},
		{name: "bad gateway", status: http.StatusBadGateway, body: `{"error":"cloud model unreachable"}`, wantRetryable: true, wantMessage: "cloud model unreachable"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("x")}})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Generate() error = %v, want *ai.APIError", err)
			}
			if apiErr.Message != tt.wantMessage || apiErr.Retryable() != tt.wantRetryable || apiErr.Provider != "ollama" {
				t.Errorf("error = %+v, want message %q retryable %v", apiErr, tt.wantMessage, tt.wantRetryable)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(%v, %v) = false", err, tt.wantIs)
			}
		})
	}
}

func TestProvider_hostAndHeaders(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalReply))
	host := strings.TrimPrefix(srv.URL, "http://")
	t.Setenv("OLLAMA_HOST", host)
	t.Setenv("OLLAMA_API_KEY", "")
	p := ollama.New(ollama.WithHeaders(http.Header{"X-Test": {"1"}}))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("x")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := srv.last(t)
	if got.Header.Get("X-Test") != "1" || got.Header.Get("Authorization") != "" || got.Header.Get("User-Agent") != "ai-go" {
		t.Errorf("headers = %v", got.Header)
	}

	authed := srv.provider(ollama.WithAPIKey("secret"))
	if _, err := authed.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("x")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if v := srv.last(t).Header.Get("Authorization"); v != "Bearer secret" {
		t.Errorf("Authorization = %q, want Bearer secret", v)
	}
}

type countingLimiter struct{ n atomic.Int32 }

func (l *countingLimiter) Wait(context.Context) error {
	l.n.Add(1)
	return nil
}

func TestProvider_limiter(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalReply))
	l := &countingLimiter{}
	if _, err := srv.provider(ollama.WithLimiter(l)).Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("x")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := l.n.Load(); got != 1 {
		t.Errorf("limiter waited %d times, want 1", got)
	}
}
