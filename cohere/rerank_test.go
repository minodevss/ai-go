package cohere_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
)

func TestRerankModel_rerank(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"r1","results":[{"index":2,"relevance_score":0.91},{"index":0,"relevance_score":0.40}],"meta":{"billed_units":{"search_units":1,"input_tokens":120}}}`))
	m := srv.provider().RerankModel("rerank-v3.5")
	resp, err := m.Rerank(t.Context(), &ai.RerankCall{
		Query:     "capital of the US",
		Documents: []string{"a", "b", "c"},
		TopN:      2,
		Provider:  ai.Providers(cohere.RerankOptions{MaxTokensPerDoc: 1024}),
	})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	want := []ai.Ranked{{Index: 2, Score: 0.91}, {Index: 0, Score: 0.40}}
	if diff := cmp.Diff(want, resp.Ranking); diff != "" {
		t.Errorf("ranking mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 120 {
		t.Errorf("usage input = %d, want 120", resp.Usage.Input)
	}
	got := srv.last(t)
	if got.Path != "/v2/rerank" {
		t.Errorf("path = %s, want /v2/rerank", got.Path)
	}
	equalJSON(t, `{"model":"rerank-v3.5","query":"capital of the US","documents":["a","b","c"],"top_n":2,"max_tokens_per_doc":1024}`, got.Body)
}

func TestRerankModel_throughCore(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"results":[{"index":0,"relevance_score":0.2},{"index":1,"relevance_score":0.9},{"index":2,"relevance_score":0.5}]}`))
	res, err := ai.Rerank(t.Context(), srv.provider().RerankModel("rerank-v3.5"), ai.RerankRequest{Query: "q", Documents: []string{"a", "b", "c"}, TopN: 2})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}
	want := []ai.Ranked{{Index: 1, Score: 0.9}, {Index: 2, Score: 0.5}}
	if diff := cmp.Diff(want, res.Ranking); diff != "" {
		t.Errorf("ranking mismatch (-want +got):\n%s", diff)
	}
}

func TestRerankModel_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"results":[{"index":5,"relevance_score":0.2}]}`))
	m := srv.provider().RerankModel("r")
	tests := []struct {
		name string
		call ai.RerankCall
		want string
	}{
		{name: "no documents", call: ai.RerankCall{Query: "q"}, want: "no documents"},
		{name: "index out of range", call: ai.RerankCall{Query: "q", Documents: []string{"a"}}, want: "out of range"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Rerank(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Rerank() error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}
