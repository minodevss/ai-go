package voyage_test

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/voyage"
)

func TestReranker_rerank(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"object":"list","data":[{"relevance_score":0.2,"index":0},{"relevance_score":0.9,"index":2},{"relevance_score":0.5,"index":1}],"model":"rerank-2.5","usage":{"total_tokens":31}}`))
	m := srv.provider().RerankModel("rerank-2.5")
	res, err := ai.Rerank(t.Context(), m, ai.RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b", "c"},
		TopN:      2,
		Provider:  ai.Providers(voyage.RerankOptions{Truncation: new(true)}),
	})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	want := []ai.Ranked{{Index: 2, Score: 0.9}, {Index: 1, Score: 0.5}}
	if diff := cmp.Diff(want, res.Ranking); diff != "" {
		t.Errorf("ranking mismatch (-want +got):\n%s", diff)
	}
	if res.Usage.Input != 31 {
		t.Errorf("usage input = %d, want 31", res.Usage.Input)
	}
	got := srv.last(t)
	if got.Path != "/v1/rerank" {
		t.Errorf("path = %s, want /v1/rerank", got.Path)
	}
	equalJSON(t, `{"model":"rerank-2.5","query":"q","documents":["a","b","c"],"top_k":2,"truncation":true}`, got.Body)
}

func TestReranker_noTopN(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"relevance_score":0.4,"index":0}],"usage":{"total_tokens":2}}`))
	m := srv.provider().RerankModel("rerank-2.5-lite")
	if _, err := m.Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}}); err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	equalJSON(t, `{"model":"rerank-2.5-lite","query":"q","documents":["a"]}`, srv.last(t).Body)
}

func TestReranker_tooManyDocuments(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{}`))
	m := srv.provider().RerankModel("rerank-2.5")
	_, err := m.Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: make([]string, 1001)})
	if err == nil || !strings.Contains(err.Error(), "exceed the limit of 1000") {
		t.Fatalf("Rerank() error = %v, want the document limit", err)
	}
	if srv.count() != 0 {
		t.Errorf("Rerank() sent %d requests, want 0", srv.count())
	}
}

func TestProvider_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		header        map[string]string
		body          string
		wantIs        error
		wantMessage   string
		wantRetryable bool
		wantRetry     time.Duration
	}{
		{name: "unauthorized", status: http.StatusUnauthorized, body: `{"detail":"Provided API key is invalid."}`, wantIs: ai.ErrAuth, wantMessage: "Provided API key is invalid."},
		{name: "forbidden", status: http.StatusForbidden, body: `{"detail":"IP blocked"}`, wantIs: ai.ErrAuth, wantMessage: "IP blocked"},
		{name: "rate limit", status: http.StatusTooManyRequests, header: map[string]string{"retry-after": "3"}, body: `{"detail":"Rate limit reached"}`, wantIs: ai.ErrRateLimit, wantMessage: "Rate limit reached", wantRetryable: true, wantRetry: 3 * time.Second},
		{name: "bad request", status: http.StatusBadRequest, body: `{"detail":"Request to model voyage-4 exceeds the batch token limit"}`, wantMessage: "Request to model voyage-4 exceeds the batch token limit"},
		{name: "server error without body", status: http.StatusServiceUnavailable, body: ``, wantRetryable: true},
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
			_, err := srv.provider().RerankModel("rerank-2.5").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}})
			apiErr, ok := errorsAsAPI(err)
			if !ok {
				t.Fatalf("Rerank() error = %v, want *ai.APIError", err)
			}
			if apiErr.Provider != "voyage" || apiErr.StatusCode != tt.status || apiErr.Message != tt.wantMessage {
				t.Errorf("APIError = %+v, want provider voyage, status %d, message %q", apiErr, tt.status, tt.wantMessage)
			}
			if tt.wantIs != nil && !isErr(err, tt.wantIs) {
				t.Errorf("error %v is not %v", err, tt.wantIs)
			}
			if apiErr.Retryable() != tt.wantRetryable || apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("Retryable() = %v, RetryAfter = %v; want %v, %v", apiErr.Retryable(), apiErr.RetryAfter, tt.wantRetryable, tt.wantRetry)
			}
		})
	}
}

func TestProvider_options(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"relevance_score":1,"index":0}],"usage":{"total_tokens":1}}`))
	limiter := &countingLimiter{}
	p := srv.provider(voyage.WithHeaders(http.Header{"X-Team": {"a"}}), voyage.WithLimiter(limiter), voyage.WithHTTPClient(srv.Client()))
	_, err := p.RerankModel("rerank-2.5").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}, Headers: http.Header{"X-Call": {"b"}}})
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
	t.Setenv("VOYAGE_API_KEY", "env-key")
	srv := newFakeServer(t, jsonReply(`{"data":[{"relevance_score":1,"index":0}],"usage":{"total_tokens":1}}`))
	p := voyage.New(voyage.WithBaseURL(srv.URL))
	if _, err := p.RerankModel("rerank-2.5").Rerank(t.Context(), &ai.RerankCall{Query: "q", Documents: []string{"a"}}); err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	if auth := srv.last(t).Header.Get("Authorization"); auth != "Bearer env-key" {
		t.Errorf("Authorization = %q, want Bearer env-key", auth)
	}
}
