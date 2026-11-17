package voyage_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/voyage"
)

func TestEmbedding_embed(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"object":"list","data":[{"object":"embedding","embedding":[0.3,0.4],"index":1},{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"voyage-4-large","usage":{"total_tokens":8}}`))
	m := srv.provider().EmbeddingModel("voyage-4-large")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}, ai.Text{Text: "c"}}},
		InputType:  ai.InputQuery,
		Dimensions: 512,
		Provider:   ai.Providers(voyage.EmbeddingOptions{Truncation: new(false)}),
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{0.1, 0.2}, {0.3, 0.4}}, resp.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 8 || string(resp.Usage.Raw) != `{"total_tokens":8}` {
		t.Errorf("usage = %+v, want input 8 and raw usage", resp.Usage)
	}
	got := srv.last(t)
	if got.Path != "/v1/embeddings" {
		t.Errorf("path = %s, want /v1/embeddings", got.Path)
	}
	if auth := got.Header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", auth)
	}
	equalJSON(t, `{"model":"voyage-4-large","input":["a","bc"],"input_type":"query","output_dimension":512,"truncation":false}`, got.Body)
}

func TestEmbedding_documentInputType(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"embedding":[1],"index":0}],"usage":{"total_tokens":1}}`))
	m := srv.provider().EmbeddingModel("voyage-3.5")
	if _, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: []string{"x"}, InputType: ai.InputDocument}); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"voyage-3.5","input":["x"],"input_type":"document"}`, srv.last(t).Body)
}

func TestEmbedding_batching(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		var b strings.Builder
		b.WriteString(`{"data":[`)
		for i, text := range req.Input {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(`{"index":` + itoa(i) + `,"embedding":[` + itoa(len(text)) + `]}`)
		}
		b.WriteString(`],"usage":{"total_tokens":` + itoa(len(req.Input)) + `}}`)
		jsonReply(b.String())(w, r)
	})
	m := srv.provider().EmbeddingModel("voyage-4-lite")
	if m.MaxBatch() != 1000 {
		t.Fatalf("MaxBatch() = %d, want 1000", m.MaxBatch())
	}
	texts := make([]string, 2100)
	for i := range texts {
		texts[i] = strings.Repeat("x", i%7+1)
	}
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts, Concurrency: 3})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 2100 || res.Usage.Input != 2100 || srv.count() != 3 {
		t.Fatalf("got %d embeddings, %d tokens, %d requests; want 2100, 2100, 3", len(res.Embeddings), res.Usage.Input, srv.count())
	}
	for i, e := range res.Embeddings {
		if want := float32(i%7 + 1); e[0] != want {
			t.Fatalf("embedding %d = %v, want %v: batches were reordered", i, e[0], want)
		}
	}
}

func TestEmbedding_multimodal(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"object":"list","data":[{"object":"embedding","embedding":[1,2],"index":0},{"object":"embedding","embedding":[3,4],"index":1}],"model":"voyage-multimodal-3.5","usage":{"text_tokens":5,"image_pixels":2000000,"total_tokens":3577}}`))
	m := srv.provider().EmbeddingModel("voyage-multimodal-3.5")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs: []ai.Parts{
			{ai.Text{Text: "a cat"}, ai.File{MediaType: "image/png", Data: []byte("png")}},
			{ai.File{MediaType: "video/mp4", Data: []byte("mp4")}},
		},
		InputType:  ai.InputDocument,
		Dimensions: 256,
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{1, 2}, {3, 4}}, resp.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 3577 {
		t.Errorf("usage input = %d, want 3577", resp.Usage.Input)
	}
	if len(resp.Warnings) != 1 || !strings.Contains(resp.Warnings[0].Message, "dimensions") {
		t.Errorf("warnings = %v, want a dimensions warning", resp.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/v1/multimodalembeddings" {
		t.Errorf("path = %s, want /v1/multimodalembeddings", got.Path)
	}
	equalJSON(t, `{"model":"voyage-multimodal-3.5","input_type":"document","inputs":[{"content":[{"type":"text","text":"a cat"},{"type":"image_base64","image_base64":"data:image/png;base64,cG5n"}]},{"content":[{"type":"video_base64","video_base64":"data:video/mp4;base64,bXA0"}]}]}`, got.Body)
}

func TestEmbedding_multimodalURLs(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"embedding":[1],"index":0}],"usage":{"total_tokens":1}}`))
	m := srv.provider().EmbeddingModel("voyage-multimodal-3")
	_, err := m.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{
		ai.File{MediaType: "image/jpeg", URL: "https://example.com/a.jpg"},
		ai.File{MediaType: "video/mp4", URL: "https://example.com/a.mp4"},
	}}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"voyage-multimodal-3","inputs":[{"content":[{"type":"image_url","image_url":"https://example.com/a.jpg"},{"type":"video_url","video_url":"https://example.com/a.mp4"}]}]}`, srv.last(t).Body)
}

func TestEmbedding_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}]}`))
	text := srv.provider().EmbeddingModel("voyage-4")
	multimodal := srv.provider().EmbeddingModel("voyage-multimodal-3.5")
	image := ai.File{MediaType: "image/png", Data: []byte("x")}
	tests := []struct {
		name  string
		model *voyage.EmbeddingModel
		call  ai.EmbedCall
		want  string
	}{
		{name: "image on text model", model: text, call: ai.EmbedCall{Inputs: []ai.Parts{{image}}}, want: "only text"},
		{name: "too many", model: text, call: ai.EmbedCall{Inputs: make([]ai.Parts, 1001)}, want: "exceed the limit"},
		{name: "count mismatch", model: text, call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}}, want: "got 1 embeddings for 2"},
		{name: "unknown input type", model: text, call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}, InputType: "x"}, want: "unknown input type"},
		{name: "audio on multimodal", model: multimodal, call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "audio/mpeg", Data: []byte("x")}}}}, want: "not supported"},
		{name: "empty file", model: multimodal, call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "image/png"}}}}, want: "neither URL nor data"},
		{name: "mixed url and data", model: multimodal, call: ai.EmbedCall{Inputs: []ai.Parts{{image, ai.File{MediaType: "image/png", URL: "https://example.com/a.png"}}}}, want: "cannot mix"},
		{name: "bad options", model: text, call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}, Provider: ai.ProviderData{"voyage": json.RawMessage(`{"truncation":"yes"}`)}}, want: "decode provider options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := srv.count()
			_, err := tt.model.Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Embed() error = %v, want containing %q", err, tt.want)
			}
			if tt.name != "count mismatch" && srv.count() != before {
				t.Errorf("Embed() sent a request despite the invalid input")
			}
		})
	}
}

func TestEmbedding_metadata(t *testing.T) {
	m := voyage.New().EmbeddingModel("voyage-3.5")
	if m.Provider() != "voyage" || m.ModelID() != "voyage-3.5" {
		t.Errorf("Provider(), ModelID() = %q, %q; want voyage, voyage-3.5", m.Provider(), m.ModelID())
	}
}
