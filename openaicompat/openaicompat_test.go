package openaicompat_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openaicompat"
)

func TestNew_request(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	p := openaicompat.New(srv.URL+"/v1", openaicompat.WithAPIKey("k"), openaicompat.WithHeaders(http.Header{"X-Team": {"a"}}))
	call := &ai.Call{
		Instructions:    ai.Texts("Be brief."),
		Messages:        hi(),
		MaxOutputTokens: 50,
		Temperature:     new(0.2),
		TopK:            4,
		Seed:            new(7),
		StopSequences:   []string{"END"},
		Reasoning:       ai.ReasoningLow,
	}
	if _, err := p.Model("m").Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := srv.last(t)
	if got.Path != "/v1/chat/completions" {
		t.Errorf("path = %s, want /v1/chat/completions", got.Path)
	}
	equalJSON(t, `{"model":"m","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"hi"}],"max_tokens":50,"temperature":0.2,"top_k":4,"seed":7,"stop":["END"],"reasoning_effort":"low"}`, got.Body)
	if h := got.Header.Get("Authorization"); h != "Bearer k" {
		t.Errorf("Authorization = %q, want Bearer k", h)
	}
	if h := got.Header.Get("X-Team"); h != "a" {
		t.Errorf("X-Team = %q, want a", h)
	}
}

func TestNew_withoutKeySendsNoAuthorization(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	if _, err := openaicompat.New(srv.URL).Model("m").Generate(t.Context(), &ai.Call{Messages: hi()}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if h := srv.last(t).Header.Get("Authorization"); h != "" {
		t.Errorf("Authorization = %q, want none", h)
	}
}

func TestNew_name(t *testing.T) {
	body := `{"id":"c","choices":[{"message":{"role":"assistant","reasoning_content":"why","content":"ok"},"finish_reason":"stop"}]}`
	srv := newFakeServer(t, jsonReply(body))
	model := openaicompat.New(srv.URL, openaicompat.WithName("acme")).Model("m")
	if model.Provider() != "acme" {
		t.Errorf("Provider() = %q, want acme", model.Provider())
	}
	resp, err := model.Generate(t.Context(), &ai.Call{Messages: hi()})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	reasoning, ok := resp.Content[0].(ai.Reasoning)
	if !ok {
		t.Fatalf("content[0] = %#v, want reasoning", resp.Content[0])
	}
	if _, ok := reasoning.Provider["acme"]; !ok {
		t.Errorf("reasoning provider data = %v, want key acme", reasoning.Provider)
	}
	history := []ai.Message{ai.User("hi"), {Role: ai.RoleAssistant, Content: resp.Content}, ai.User("next")}
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: history}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := as[map[string]any](t, as[[]any](t, bodyOf(srv.last(t).Body)["messages"])[1]); got["reasoning_content"] != nil {
		t.Errorf("generic provider replayed reasoning: %v", got)
	}
}

func TestNew_streamAndResponse(t *testing.T) {
	srv := newFakeServer(t, sseReply(
		`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"id":"c1","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		`{"id":"c1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":5}}`,
	))
	parts := streamAll(t, openaicompat.New(srv.URL).Model("m"), &ai.Call{Messages: hi()})
	want := []string{"TextStart", "TextDelta", "TextDelta", "TextEnd", "StepEnd"}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Errorf("part types mismatch (-want +got):\n%s", diff)
	}
	end, ok := parts[len(parts)-1].(ai.StepEnd)
	if !ok || end.Usage.Input != 7 || end.Usage.Output != 5 || end.FinishReason != ai.FinishStop {
		t.Errorf("StepEnd = %#v, want stop with usage 7/5", parts[len(parts)-1])
	}
	equalJSON(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`, srv.last(t).Body)
}

func TestNew_embeddings(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"usage":{"prompt_tokens":4,"total_tokens":4}}`))
	model := openaicompat.New(srv.URL).EmbeddingModel("e")
	res, err := ai.Embed(t.Context(), model, ai.EmbedRequest{Texts: []string{"a", "b"}, Dimensions: 2, InputType: ai.InputQuery})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{1, 2}, {3, 4}}, res.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if len(res.Warnings) != 1 {
		t.Errorf("warnings = %v, want one about the input type", res.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/embeddings" {
		t.Errorf("path = %s, want /embeddings", got.Path)
	}
	equalJSON(t, `{"model":"e","input":["a","b"],"dimensions":2,"encoding_format":"float"}`, got.Body)
}

func TestWithTokenSource(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	calls := 0
	token := func(context.Context) (string, error) {
		calls++
		return "entra-token", nil
	}
	p := openaicompat.New(srv.URL, openaicompat.WithAPIKey("ignored"), openaicompat.WithTokenSource(token))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: hi()}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if h := srv.last(t).Header.Get("Authorization"); h != "Bearer entra-token" || calls != 1 {
		t.Errorf("Authorization = %q after %d token calls, want Bearer entra-token after 1", h, calls)
	}
}

func TestWithTokenSource_error(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	boom := errors.New("no token")
	p := openaicompat.New(srv.URL, openaicompat.WithTokenSource(func(context.Context) (string, error) { return "", boom }))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: hi()}); !errors.Is(err, boom) {
		t.Errorf("Generate() error = %v, want the token error", err)
	}
}

func TestProvider_LanguageModel(t *testing.T) {
	m, err := openaicompat.New("http://localhost").LanguageModel("x")
	if err != nil || m.ModelID() != "x" {
		t.Errorf("LanguageModel() = %v, %v, want model x", m, err)
	}
}

func TestModel_errorBodies(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "openai shape", body: `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`, want: "invalid_api_key"},
		{name: "mistral numeric code", body: `{"object":"error","message":"Tool call id was x but must be a-z, A-Z, 0-9, with a length of 9.","type":"invalid_request_invalid_args","code":3280}`, want: "3280"},
		{name: "detail only", body: `{"detail":"model not found"}`, want: "model not found"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := openaicompat.New(srv.URL).Model("m").Generate(t.Context(), &ai.Call{Messages: hi()})
			var apiErr *ai.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("error = %v, want *ai.APIError", err)
			}
			if !strings.Contains(apiErr.Code+" "+apiErr.Message, tt.want) {
				t.Errorf("code/message = %q/%q, want %q", apiErr.Code, apiErr.Message, tt.want)
			}
		})
	}
}

type orderLimiter struct {
	srv        *fakeServer
	seenAtWait []int
}

func (l *orderLimiter) Wait(context.Context) error {
	l.seenAtWait = append(l.seenAtWait, l.srv.count())
	return nil
}

func TestWithLimiter(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	lim := &orderLimiter{srv: srv}
	p := openaicompat.New(srv.URL, openaicompat.WithLimiter(lim))
	for range 2 {
		if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: hi()}); err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
	}
	if diff := cmp.Diff([]int{0, 1}, lim.seenAtWait); diff != "" {
		t.Errorf("requests sent before each Wait (-want +got):\n%s", diff)
	}
}

type failingLimiter struct{ err error }

func (l failingLimiter) Wait(context.Context) error { return l.err }

func TestWithLimiter_error(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	boom := errors.New("limited")
	p := openaicompat.New(srv.URL, openaicompat.WithLimiter(failingLimiter{err: boom}))
	if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: hi()}); !errors.Is(err, boom) {
		t.Errorf("Generate() error = %v, want the limiter error", err)
	}
	if srv.count() != 0 {
		t.Errorf("sent %d requests, want 0", srv.count())
	}
}
