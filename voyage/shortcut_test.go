package voyage_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/voyage"
)

func TestNewModelShortcuts(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")
	tests := []struct {
		name     string
		wantPath string
		reply    string
		call     func(ctx context.Context) (string, error)
	}{
		{
			name:     "embedding",
			wantPath: "/v1/embeddings",
			reply:    `{"data":[{"index":0,"embedding":[0.1]}],"usage":{"total_tokens":1}}`,
			call: func(ctx context.Context) (string, error) {
				m := voyage.NewEmbeddingModel("voyage-4")
				if _, err := m.Embed(ctx, &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "x"}}}}); err != nil {
					return "", err
				}
				return m.ModelID(), nil
			},
		},
		{
			name:     "rerank",
			wantPath: "/v1/rerank",
			reply:    `{"data":[{"index":0,"relevance_score":1}],"usage":{"total_tokens":1}}`,
			call: func(ctx context.Context) (string, error) {
				m := voyage.NewRerankModel("rerank-2.5")
				if _, err := m.Rerank(ctx, &ai.RerankCall{Query: "q", Documents: []string{"a"}}); err != nil {
					return "", err
				}
				return m.ModelID(), nil
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got *http.Request
			original := http.DefaultTransport
			http.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				got = r
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"application/json"}},
					Body:       io.NopCloser(strings.NewReader(tt.reply)),
					Request:    r,
				}, nil
			})
			t.Cleanup(func() { http.DefaultTransport = original })

			gotID, err := tt.call(t.Context())
			if err != nil {
				t.Fatalf("call error = %v", err)
			}
			wantID := map[string]string{"embedding": "voyage-4", "rerank": "rerank-2.5"}[tt.name]
			if gotID != wantID {
				t.Errorf("ModelID() = %q, want %q", gotID, wantID)
			}
			if got == nil || got.URL.Path != tt.wantPath {
				t.Fatalf("request = %v, want path %s", got, tt.wantPath)
			}
			if auth := got.Header.Get("Authorization"); auth != "Bearer env-key" {
				t.Errorf("Authorization = %q, want Bearer env-key", auth)
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
