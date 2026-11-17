package cohere_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
)

func generate(t *testing.T, srv *fakeServer, call *ai.Call) *ai.Response {
	t.Helper()
	resp, err := srv.provider().Model("command-a-03-2025").Generate(t.Context(), call)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	return resp
}

func TestChatModel_requestMapping(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	weather.Strict = true
	resp := generate(t, srv, &ai.Call{
		Instructions:     ai.Texts("Be brief."),
		Messages:         []ai.Message{ai.UserParts(ai.Text{Text: "hi"}, ai.File{MediaType: "image/png", Data: []byte("png")})},
		Tools:            []ai.Tool{weather},
		ToolChoice:       ai.ToolRequired(),
		MaxOutputTokens:  100,
		Temperature:      new(0.2),
		TopP:             new(0.9),
		TopK:             40,
		Seed:             new(7),
		StopSequences:    []string{"END"},
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.2),
		Reasoning:        ai.ReasoningNone,
		Headers:          http.Header{"X-Trace": {"abc"}},
		Provider: ai.Providers(cohere.Options{
			SafetyMode:   cohere.SafetyStrict,
			CitationMode: cohere.CitationsFast,
			Priority:     2,
			Documents:    []cohere.Document{{Data: map[string]any{"title": "T", "text": "body"}}},
		}),
	})
	got := srv.last(t)
	if got.Path != "/v2/chat" {
		t.Errorf("path = %s, want /v2/chat", got.Path)
	}
	if got.Header.Get("Authorization") != "Bearer test-key" || got.Header.Get("X-Client-Name") != "ai-go" || got.Header.Get("X-Trace") != "abc" {
		t.Errorf("headers = %v", got.Header)
	}
	equalJSON(t, `{
		"model":"command-a-03-2025",
		"messages":[
			{"role":"system","content":"Be brief."},
			{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}}]}
		],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"Weather by city","parameters":`+string(weather.InputSchema)+`}}],
		"strict_tools":true,
		"tool_choice":"REQUIRED",
		"documents":[{"id":"0","data":{"title":"T","text":"body"}}],
		"citation_options":{"mode":"FAST"},
		"safety_mode":"STRICT",
		"max_tokens":100,
		"stop_sequences":["END"],
		"temperature":0.2,
		"seed":7,
		"frequency_penalty":0.2,
		"presence_penalty":0.1,
		"k":40,
		"p":0.9,
		"thinking":{"type":"disabled"},
		"priority":2
	}`, got.Body)
	want := ai.Usage{Input: 7, Output: 2, CacheRead: 4}
	resp.Usage.Raw = nil
	if diff := cmp.Diff(want, resp.Usage); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
}

func TestChatModel_thinkingOptions(t *testing.T) {
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{name: "effort enables", call: ai.Call{Reasoning: ai.ReasoningHigh}, want: `"thinking":{"type":"enabled"}`},
		{name: "option budget wins", call: ai.Call{Reasoning: ai.ReasoningNone, Provider: ai.Providers(cohere.Options{Thinking: &cohere.Thinking{Type: cohere.ThinkingEnabled, TokenBudget: 2048}})}, want: `"thinking":{"type":"enabled","token_budget":2048}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalChat))
			tt.call.Messages = []ai.Message{ai.User("hi")}
			generate(t, srv, &tt.call)
			if body := string(srv.last(t).Body); !strings.Contains(body, tt.want) {
				t.Errorf("body = %s, want it to contain %s", body, tt.want)
			}
		})
	}
}

func TestChatModel_historyReplay(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	cite := ai.Citation{Kind: "document", Provider: ai.ProviderData{"cohere": json.RawMessage(`{"citation":{"start":0,"end":2,"text":"hi","content_index":0,"type":"TEXT_CONTENT","sources":[{"type":"document","id":"0"}]}}`)}}
	history := []ai.Message{
		ai.User("question"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Reasoning{Text: "hmm", Provider: ai.ProviderData{"cohere": json.RawMessage(`{"thinking":true}`)}},
			ai.Text{Text: "hi", Citations: []ai.Citation{cite, cite}},
			ai.Reasoning{Text: "plan", Provider: ai.ProviderData{"cohere": json.RawMessage(`{"toolPlan":true}`)}},
			ai.ToolCall{ID: "a_1", Name: "a", Input: json.RawMessage(`{ "x": 1 }`)},
			ai.ToolCall{ID: "b_2", Name: "b"},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{
			ai.ToolResult{CallID: "a_1", Name: "a", Output: ai.ToolOutput{JSON: json.RawMessage(`{"ok": true}`)}},
		}},
		ai.User("next"),
	}
	generate(t, srv, &ai.Call{Messages: history})
	equalJSON(t, `{
		"model":"command-a-03-2025",
		"messages":[
			{"role":"user","content":[{"type":"text","text":"question"}]},
			{"role":"assistant",
			 "content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"hi"}],
			 "tool_plan":"plan",
			 "citations":[{"start":0,"end":2,"text":"hi","content_index":0,"type":"TEXT_CONTENT","sources":[{"type":"document","id":"0"}]}],
			 "tool_calls":[
				{"id":"a_1","type":"function","function":{"name":"a","arguments":"{\"x\":1}"}},
				{"id":"b_2","type":"function","function":{"name":"b","arguments":"{}"}}
			 ]},
			{"role":"tool","tool_call_id":"a_1","content":[{"type":"document","document":{"data":"{\"ok\":true}"}}]},
			{"role":"tool","tool_call_id":"b_2","content":[{"type":"document","document":{"data":"error: this tool call has no result"}}]},
			{"role":"user","content":[{"type":"text","text":"next"}]}
		]
	}`, srv.last(t).Body)
}

func TestChatModel_toolResultError(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	generate(t, srv, &ai.Call{Messages: []ai.Message{
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "a_1", Name: "a", Input: json.RawMessage(`{}`)}}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "a_1", Name: "a", IsError: true, Output: ai.ToolOutput{Text: "boom"}}}},
	}})
	if body := string(srv.last(t).Body); !strings.Contains(body, `"data":"error: boom"`) {
		t.Errorf("body = %s, want the error prefix", body)
	}
}

func TestChatModel_documents(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"c","finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"text","text":"Green."}],"citations":[{"start":0,"end":5,"text":"Green","content_index":0,"type":"TEXT_CONTENT","sources":[{"type":"document","id":"1","document":{"id":"1","title":"b.txt","snippet":"sky is blue"}}]}]}}`))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.UserParts(
		ai.File{MediaType: "text/plain", Filename: "a.txt", Data: []byte("grass is green"), Citations: true},
		ai.File{MediaType: "text/markdown", Filename: "b.txt", Data: []byte("sky is blue")},
		ai.Text{Text: "colors?"},
	)}})
	equalJSON(t, `{"model":"command-a-03-2025","messages":[{"role":"user","content":[{"type":"text","text":"colors?"}]}],"documents":[{"id":"0","data":{"title":"a.txt","text":"grass is green"}},{"id":"1","data":{"title":"b.txt","text":"sky is blue"}}]}`, srv.last(t).Body)
	text, ok := resp.Content[0].(ai.Text)
	if !ok || len(text.Citations) != 1 {
		t.Fatalf("content = %#v, want one citation", resp.Content)
	}
	c := text.Citations[0]
	if c.DocIndex == nil || *c.DocIndex != 1 || c.Title != "b.txt" || c.CitedText != "sky is blue" || c.Start != 0 || c.End != 5 || c.Kind != "document" {
		t.Errorf("citation = %+v", c)
	}
}

func TestChatModel_unsupportedInput(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	m := srv.provider().Model("m")
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{name: "pdf", call: ai.Call{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "application/pdf", Data: []byte("x")})}}, want: "not supported"},
		{name: "output with tools", call: ai.Call{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{ai.NewClientTool[weatherInput]("w", "d")}, Output: &ai.OutputSpec{Name: "o"}}, want: "cannot be combined"},
		{name: "bad role", call: ai.Call{Messages: []ai.Message{{Role: "x"}}}, want: "unknown role"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Generate(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Generate() error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("sent %d requests, want 0", srv.count())
	}
}

func TestChatModel_structuredOutput(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	resp := generate(t, srv, &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Output:   &ai.OutputSpec{Name: "o", Schema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer","minimum":1}},"required":["n"]}`)},
	})
	equalJSON(t, `{"model":"command-a-03-2025","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"response_format":{"type":"json_object","json_schema":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}}`, srv.last(t).Body)
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, `"minimum"`) {
		t.Errorf("warnings = %v, want the minimum warning", resp.Warnings)
	}
}

func TestChatModel_namedToolChoiceWarns(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{ai.NewClientTool[weatherInput]("w", "d")}, ToolChoice: ai.ToolNamed("w")})
	if !strings.Contains(string(srv.last(t).Body), `"tool_choice":"REQUIRED"`) || len(resp.Warnings) != 1 {
		t.Errorf("body = %s, warnings = %v", srv.last(t).Body, resp.Warnings)
	}
}

func TestChatModel_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		header        map[string]string
		body          string
		wantIs        error
		wantRetryable bool
		wantRetry     time.Duration
	}{
		{name: "rate limit", status: http.StatusTooManyRequests, header: map[string]string{"retry-after": "3"}, body: `{"message":"too many requests"}`, wantIs: ai.ErrRateLimit, wantRetryable: true, wantRetry: 3 * time.Second},
		{name: "unauthorized", status: http.StatusUnauthorized, body: `{"message":"invalid api token"}`, wantIs: ai.ErrAuth},
		{name: "forbidden", status: http.StatusForbidden, body: `{"message":"no access"}`, wantIs: ai.ErrAuth},
		{name: "gateway timeout", status: http.StatusGatewayTimeout, body: `{"message":"timeout"}`, wantRetryable: true},
		{name: "service unavailable", status: http.StatusServiceUnavailable, body: `{"message":"down"}`, wantRetryable: true},
		{name: "validation", status: http.StatusUnprocessableEntity, body: `{"id":"x","message":"bad field"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				for k, v := range tt.header {
					w.Header().Set(k, v)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Generate() error = %v, want *ai.APIError", err)
			}
			if apiErr.Provider != "cohere" || apiErr.StatusCode != tt.status || apiErr.Retryable() != tt.wantRetryable || apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("error = %+v, retryable %v", apiErr, apiErr.Retryable())
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(%v, %v) = false", err, tt.wantIs)
			}
			if apiErr.Message == "" {
				t.Errorf("message is empty, body %q", apiErr.Body)
			}
		})
	}
}

func TestChatModel_streamErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "ends early", body: "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"1\"}\n\n", want: "ended before"},
		{name: "error in message end", body: "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"ERROR\",\"error\":\"model overloaded\"}}\n\n", want: "model overloaded"},
		{name: "bad event", body: "data: {nope\n\n", want: "decode stream event"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, sseReply(tt.body))
			stream := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
			_, err := stream.Result()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Result() error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestChatModel_streamRequestBody(t *testing.T) {
	srv := newFakeServer(t, sseReply("event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":1,\"output_tokens\":1}}}}\n\n"))
	res, err := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if !strings.Contains(string(srv.last(t).Body), `"stream":true`) || res.FinishReason != ai.FinishStop || res.Usage.Input != 1 {
		t.Errorf("body = %s, result = %+v", srv.last(t).Body, res)
	}
}

func TestChatModel_loop(t *testing.T) {
	calls := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			jsonReply(`{"id":"1","finish_reason":"TOOL_CALL","message":{"role":"assistant","tool_plan":"look it up","tool_calls":[{"id":"get_weather_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Seoul\"}"}}]}}`)(w, r)
			return
		}
		jsonReply(minimalChat)(w, r)
	})
	weather := ai.NewTool("get_weather", "Weather", func(_ context.Context, in weatherInput) (string, error) { return "Sunny in " + in.City, nil })
	res, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("weather?")}, Tools: []ai.Tool{weather}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Text != "hi" || len(res.Steps) != 2 {
		t.Errorf("text = %q, steps = %d, want %q and 2", res.Text, len(res.Steps), "hi")
	}
	if body := string(srv.last(t).Body); !strings.Contains(body, `"tool_call_id":"get_weather_1"`) || !strings.Contains(body, `"tool_plan":"look it up"`) {
		t.Errorf("second request body = %s", body)
	}
}

func TestProvider_options(t *testing.T) {
	t.Setenv("CO_API_KEY", "env-key")
	srv := newFakeServer(t, jsonReply(minimalChat))
	t.Setenv("CO_API_URL", srv.URL)
	if _, err := cohere.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := srv.last(t).Header.Get("Authorization"); got != "Bearer env-key" {
		t.Errorf("Authorization = %q, want the environment key", got)
	}
}

type countingLimiter struct{ n int }

func (l *countingLimiter) Wait(context.Context) error {
	l.n++
	return nil
}

func TestProvider_limiterCoversEveryKind(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/embed":
			jsonReply(`{"embeddings":{"float":[[1]]}}`)(w, r)
		case "/v2/rerank":
			jsonReply(`{"results":[{"index":0,"relevance_score":0.5}]}`)(w, r)
		default:
			jsonReply(minimalChat)(w, r)
		}
	})
	l := &countingLimiter{}
	p := srv.provider(cohere.WithLimiter(l))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if _, err := ai.Embed(t.Context(), p.EmbeddingModel("e"), ai.EmbedRequest{Texts: []string{"a"}}); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if _, err := ai.Rerank(t.Context(), p.RerankModel("r"), ai.RerankRequest{Query: "q", Documents: []string{"d"}}); err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	if l.n != 3 {
		t.Errorf("limiter waits = %d, want 3", l.n)
	}
}
