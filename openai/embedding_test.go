package openai_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
)

func TestEmbeddingModel_embed(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"object":"list","data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}],"model":"m","usage":{"prompt_tokens":6,"total_tokens":6}}`))
	m := srv.provider().EmbeddingModel("text-embedding-3-small")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}, ai.Text{Text: "c"}}},
		Dimensions: 2,
		InputType:  ai.InputQuery,
		Provider:   ai.ProviderData{"openai": []byte(`{"user":"u1"}`)},
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{0.1, 0.2}, {0.3, 0.4}}, resp.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 6 {
		t.Errorf("usage input = %d, want 6", resp.Usage.Input)
	}
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "input type") {
		t.Errorf("warnings = %v, want input type warning", resp.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/v1/embeddings" {
		t.Errorf("path = %s, want /v1/embeddings", got.Path)
	}
	equalJSON(t, `{"model":"text-embedding-3-small","input":["a","bc"],"dimensions":2,"encoding_format":"float","user":"u1"}`, got.Body)
}

func TestEmbeddingModel_batching(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		n := len(req.Input)
		var b strings.Builder
		b.WriteString(`{"data":[`)
		for i := range n {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(`{"index":` + itoa(i) + `,"embedding":[1]}`)
		}
		b.WriteString(`],"usage":{"prompt_tokens":` + itoa(n) + `}}`)
		jsonReply(b.String())(w, r)
	})
	m := srv.provider().EmbeddingModel("e")
	if m.MaxBatch() != 2048 {
		t.Fatalf("MaxBatch() = %d, want 2048", m.MaxBatch())
	}
	texts := make([]string, 2100)
	for i := range texts {
		texts[i] = "t" + itoa(i)
	}
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts, Concurrency: 2})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 2100 || res.Usage.Input != 2100 || srv.count() != 2 {
		t.Errorf("got %d embeddings, %d tokens, %d requests; want 2100, 2100, 2", len(res.Embeddings), res.Usage.Input, srv.count())
	}
}

func TestEmbeddingModel_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}]}`))
	m := srv.provider().EmbeddingModel("e")
	tests := []struct {
		name string
		call ai.EmbedCall
		want string
	}{
		{name: "image input", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "image/png", Data: []byte("x")}}}}, want: "only text"},
		{name: "too many", call: ai.EmbedCall{Inputs: make([]ai.Parts, 2049)}, want: "exceed the limit"},
		{name: "count mismatch", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}}, want: "got 1 embeddings for 2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) || !strings.HasPrefix(err.Error(), "openai:") {
				t.Errorf("Embed() error = %v, want openai error containing %q", err, tt.want)
			}
		})
	}
}
