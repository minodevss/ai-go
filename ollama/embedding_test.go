package ollama_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ollama"
)

func TestEmbeddingModel_embed(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"model":"all-minilm","embeddings":[[0.1,0.2],[0.3,0.4]],"total_duration":14143917,"load_duration":1019500,"prompt_eval_count":8}`))
	m := srv.provider().EmbeddingModel("all-minilm")
	if m.MaxBatch() != 0 {
		t.Errorf("MaxBatch() = %d, want 0", m.MaxBatch())
	}
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}, ai.Text{Text: "c"}}},
		Dimensions: 2,
		InputType:  ai.InputQuery,
		Provider:   ai.Providers(ollama.EmbeddingOptions{Truncate: new(false), KeepAlive: "0", NumCtx: 512}),
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{0.1, 0.2}, {0.3, 0.4}}, resp.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 8 {
		t.Errorf("usage input = %d, want 8", resp.Usage.Input)
	}
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "input type") {
		t.Errorf("warnings = %v, want input type warning", resp.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/api/embed" {
		t.Errorf("path = %s, want /api/embed", got.Path)
	}
	equalJSON(t, `{"model":"all-minilm","input":["a","bc"],"dimensions":2,"truncate":false,"keep_alive":"0","options":{"num_ctx":512}}`, got.Body)
}

func TestEmbeddingModel_multimodal(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"model":"m","embeddings":[[1],[2]]}`))
	m := srv.provider().EmbeddingModel("m")
	_, err := m.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{
		{ai.Text{Text: "a cat"}, ai.File{MediaType: "image/png", Data: []byte("img")}},
		{ai.File{MediaType: "audio/wav", Data: []byte("snd")}},
	}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"m","input":[{"text":"a cat","image":"aW1n"},{"audio":"c25k"}]}`, srv.last(t).Body)
}

func TestEmbeddingModel_ordering(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		var vecs []string
		for _, in := range req.Input {
			vecs = append(vecs, "["+strings.TrimPrefix(in, "t")+"]")
		}
		jsonReply(`{"embeddings":[`+strings.Join(vecs, ",")+`],"prompt_eval_count":`+itoa(len(req.Input))+`}`)(w, r)
	})
	texts := make([]string, 25)
	want := make([][]float32, 25)
	for i := range texts {
		texts[i] = "t" + itoa(i)
		want[i] = []float32{float32(i)}
	}
	res, err := ai.Embed(t.Context(), srv.provider().EmbeddingModel("e"), ai.EmbedRequest{Texts: texts})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff(want, res.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if srv.count() != 1 || res.Usage.Input != 25 {
		t.Errorf("got %d requests and %d tokens, want 1 and 25", srv.count(), res.Usage.Input)
	}
}

func TestEmbeddingModel_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"embeddings":[[1]]}`))
	m := srv.provider().EmbeddingModel("e")
	tests := []struct {
		name string
		call ai.EmbedCall
		want string
	}{
		{name: "pdf input", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "application/pdf", Data: []byte("x")}}}}, want: "unsupported or repeated"},
		{name: "two images", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "image/png", Data: []byte("x")}, ai.File{MediaType: "image/png", Data: []byte("y")}}}}, want: "at most one image"},
		{name: "count mismatch", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}}, want: "got 1 embeddings for 2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Embed() error = %v, want %q", err, tt.want)
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
