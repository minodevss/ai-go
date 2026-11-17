package cohere_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
)

func TestEmbeddingModel_embed(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"e1","embeddings":{"float":[[0.1,0.2],[0.3,0.4]]},"texts":["a","bc"],"meta":{"billed_units":{"input_tokens":6},"warnings":["input truncated"]}}`))
	m := srv.provider().EmbeddingModel("embed-v4.0")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}, ai.Text{Text: "c"}}},
		InputType:  ai.InputQuery,
		Dimensions: 256,
		Provider:   ai.Providers(cohere.EmbeddingOptions{Truncate: cohere.TruncateStart, MaxTokens: 512}),
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
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "input truncated") {
		t.Errorf("warnings = %v", resp.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/v2/embed" {
		t.Errorf("path = %s, want /v2/embed", got.Path)
	}
	equalJSON(t, `{"model":"embed-v4.0","input_type":"search_query","embedding_types":["float"],"texts":["a","bc"],"max_tokens":512,"output_dimension":256,"truncate":"START"}`, got.Body)
}

func TestEmbeddingModel_inputTypes(t *testing.T) {
	tests := []struct {
		name string
		call ai.EmbedCall
		want string
	}{
		{name: "default is document", call: ai.EmbedCall{}, want: `"input_type":"search_document"`},
		{name: "document", call: ai.EmbedCall{InputType: ai.InputDocument}, want: `"input_type":"search_document"`},
		{name: "option override", call: ai.EmbedCall{InputType: ai.InputQuery, Provider: ai.Providers(cohere.EmbeddingOptions{InputType: cohere.InputClustering})}, want: `"input_type":"clustering"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(`{"embeddings":{"float":[[1]]}}`))
			tt.call.Inputs = []ai.Parts{{ai.Text{Text: "x"}}}
			if _, err := srv.provider().EmbeddingModel("e").Embed(t.Context(), &tt.call); err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			if body := string(srv.last(t).Body); !strings.Contains(body, tt.want) {
				t.Errorf("body = %s, want %s", body, tt.want)
			}
		})
	}
}

func TestEmbeddingModel_images(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"embeddings":{"float":[[1],[2]]},"meta":{"billed_units":{"image_tokens":50,"text_tokens":4}}}`))
	m := srv.provider().EmbeddingModel("embed-v4.0")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{
		{ai.File{MediaType: "image/png", Data: []byte("png")}},
		{ai.Text{Text: "caption"}, ai.File{MediaType: "image/jpeg", Data: []byte("jpg")}},
	}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if resp.Usage.Input != 54 {
		t.Errorf("usage input = %d, want 54", resp.Usage.Input)
	}
	equalJSON(t, `{"model":"embed-v4.0","input_type":"search_document","embedding_types":["float"],"inputs":[
		{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}}]},
		{"content":[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,anBn"}}]}
	]}`, srv.last(t).Body)
}

func TestEmbeddingModel_batching(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Texts []string `json:"texts"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		vectors := make([]string, len(req.Texts))
		for i, text := range req.Texts {
			vectors[i] = "[" + strings.TrimPrefix(text, "t") + "]"
		}
		jsonReply(`{"embeddings":{"float":[`+strings.Join(vectors, ",")+`]},"meta":{"billed_units":{"input_tokens":`+itoa(len(req.Texts))+`}}}`)(w, r)
	})
	m := srv.provider().EmbeddingModel("e")
	if m.MaxBatch() != 96 {
		t.Fatalf("MaxBatch() = %d, want 96", m.MaxBatch())
	}
	texts := make([]string, 200)
	for i := range texts {
		texts[i] = "t" + itoa(i)
	}
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts, Concurrency: 3})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 200 || res.Usage.Input != 200 || srv.count() != 3 {
		t.Fatalf("got %d embeddings, %d tokens, %d requests; want 200, 200, 3", len(res.Embeddings), res.Usage.Input, srv.count())
	}
	for i, e := range res.Embeddings {
		if len(e) != 1 || int(e[0]) != i {
			t.Fatalf("embedding %d = %v, want [%d]", i, e, i)
		}
	}
}

func TestEmbeddingModel_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"embeddings":{"float":[[1]]}}`))
	m := srv.provider().EmbeddingModel("e")
	tests := []struct {
		name string
		call ai.EmbedCall
		want string
	}{
		{name: "image without data", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "image/png", URL: "https://x/y.png"}}}}, want: "needs Data"},
		{name: "too many", call: ai.EmbedCall{Inputs: make([]ai.Parts, 97)}, want: "exceed the limit"},
		{name: "count mismatch", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}}, want: "got 1 embeddings for 2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Embed() error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func itoa(n int) string {
	b, err := json.Marshal(n)
	if err != nil {
		return ""
	}
	return string(b)
}
