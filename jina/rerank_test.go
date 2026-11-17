package jina_test

import (
	"net/http"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/jina"
)

func TestReranker_rerank(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"model":"jina-reranker-v3","usage":{"total_tokens":42},"results":[{"index":2,"relevance_score":0.9},{"index":1,"relevance_score":0.5},{"index":0,"relevance_score":0.2}]}`))
	m := srv.provider().RerankModel("jina-reranker-v3")
	res, err := ai.Rerank(t.Context(), m, ai.RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b", "c"},
		TopN:      2,
		Provider:  ai.Providers(jina.RerankOptions{MaxDocLength: 1024}),
	})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	want := []ai.Ranked{{Index: 2, Score: 0.9}, {Index: 1, Score: 0.5}}
	if diff := cmp.Diff(want, res.Ranking); diff != "" {
		t.Errorf("ranking mismatch (-want +got):\n%s", diff)
	}
	if res.Usage.Input != 42 {
		t.Errorf("usage input = %d, want 42", res.Usage.Input)
	}
	got := srv.last(t)
	if got.Path != "/v1/rerank" {
		t.Errorf("path = %s, want /v1/rerank", got.Path)
	}
	equalJSON(t, `{"model":"jina-reranker-v3","query":"q","documents":["a","b","c"],"top_n":2,"return_documents":false,"max_doc_length":1024}`, got.Body)
}

func TestReranker_minimalRequest(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"results":[{"index":0,"relevance_score":0.4}],"usage":{"total_tokens":2}}`))
	if _, err := srv.provider().RerankModel("jina-reranker-v2-base-multilingual").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}}); err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	equalJSON(t, `{"model":"jina-reranker-v2-base-multilingual","query":"q","documents":["a"],"return_documents":false}`, srv.last(t).Body)
}

func TestProvider_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		header        map[string]string
		body          string
		wantIs        error
		wantNotIs     error
		wantCode      string
		wantMessage   string
		wantRetryable bool
		wantRetry     time.Duration
	}{
		{name: "invalid key", status: http.StatusUnauthorized, body: `{"detail":"Invalid API key.","code":"AUTH_INVALID_API_KEY","request_id":"abc"}`, wantIs: ai.ErrAuth, wantCode: "AUTH_INVALID_API_KEY", wantMessage: "Invalid API key."},
		{name: "insufficient balance is quota", status: http.StatusForbidden, body: `{"detail":"Insufficient balance","code":"AUTHZ_INSUFFICIENT_BALANCE"}`, wantIs: ai.ErrQuota, wantCode: "billing_AUTHZ_INSUFFICIENT_BALANCE", wantMessage: "Insufficient balance"},
		{name: "rate limit", status: http.StatusTooManyRequests, header: map[string]string{"retry-after": "2"}, body: `{"detail":"slow down","code":"RATE_REQUEST_LIMIT_EXCEEDED"}`, wantIs: ai.ErrRateLimit, wantNotIs: ai.ErrQuota, wantCode: "RATE_REQUEST_LIMIT_EXCEEDED", wantMessage: "slow down", wantRetryable: true, wantRetry: 2 * time.Second},
		{name: "token limit", status: http.StatusBadRequest, body: `{"detail":"too long","code":"INPUT_TOKEN_LIMIT_EXCEEDED"}`, wantCode: "INPUT_TOKEN_LIMIT_EXCEEDED", wantMessage: "too long"},
		{name: "validation errors keep the summary", status: http.StatusUnprocessableEntity, body: `{"detail":"1 validation error","errors":[{"field":"body -> input","message":"bad","type":"string_type"}]}`, wantMessage: "1 validation error"},
		{name: "unavailable", status: http.StatusServiceUnavailable, body: `{"detail":"down","code":"SERVICE_UNAVAILABLE"}`, wantCode: "SERVICE_UNAVAILABLE", wantMessage: "down", wantRetryable: true},
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
			_, err := srv.provider().RerankModel("jina-reranker-v3").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}})
			apiErr, ok := errorsAsAPI(err)
			if !ok {
				t.Fatalf("Rerank() error = %v, want *ai.APIError", err)
			}
			if apiErr.Provider != "jina" || apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode || apiErr.Message != tt.wantMessage {
				t.Errorf("APIError = %+v, want status %d, code %q, message %q", apiErr, tt.status, tt.wantCode, tt.wantMessage)
			}
			if tt.wantIs != nil && !isErr(err, tt.wantIs) {
				t.Errorf("error %v is not %v", err, tt.wantIs)
			}
			if tt.wantNotIs != nil && isErr(err, tt.wantNotIs) {
				t.Errorf("error %v is %v", err, tt.wantNotIs)
			}
			if apiErr.Retryable() != tt.wantRetryable || apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("Retryable() = %v, RetryAfter = %v; want %v, %v", apiErr.Retryable(), apiErr.RetryAfter, tt.wantRetryable, tt.wantRetry)
			}
		})
	}
}

func TestProvider_options(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"results":[{"index":0,"relevance_score":1}],"usage":{"total_tokens":1}}`))
	limiter := &countingLimiter{}
	p := srv.provider(jina.WithHeaders(http.Header{"X-Team": {"a"}}), jina.WithLimiter(limiter), jina.WithHTTPClient(srv.Client()))
	_, err := p.RerankModel("jina-reranker-v3").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}, Headers: http.Header{"X-Call": {"b"}}})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	got := srv.last(t)
	if got.Header.Get("X-Team") != "a" || got.Header.Get("X-Call") != "b" || got.Header.Get("User-Agent") != "ai-go" {
		t.Errorf("headers = %v, want X-Team, X-Call and the ai-go user agent", got.Header)
	}
	if limiter.waits != 1 {
		t.Errorf("limiter waited %d times, want 1", limiter.waits)
	}
}

func TestNew_environmentKey(t *testing.T) {
	t.Setenv("JINA_API_KEY", "env-key")
	srv := newFakeServer(t, jsonReply(`{"results":[{"index":0,"relevance_score":1}],"usage":{"total_tokens":1}}`))
	p := jina.New(jina.WithBaseURL(srv.URL))
	if _, err := p.RerankModel("jina-reranker-v3").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}}); err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	if auth := srv.last(t).Header.Get("Authorization"); auth != "Bearer env-key" {
		t.Errorf("Authorization = %q, want Bearer env-key", auth)
	}
}
