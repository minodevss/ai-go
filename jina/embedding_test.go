package jina_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/jina"
)

func TestEmbedding_embed(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"model":"jina-embeddings-v3","object":"list","usage":{"total_tokens":11,"prompt_tokens":11},"data":[{"object":"embedding","index":1,"embedding":[0.3,0.4]},{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`))
	m := srv.provider().EmbeddingModel("jina-embeddings-v3")
	resp, err := m.Embed(t.Context(), &ai.EmbedCall{
		Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}, ai.Text{Text: "c"}}},
		InputType:  ai.InputDocument,
		Dimensions: 256,
		Provider:   ai.Providers(jina.EmbeddingOptions{Normalized: new(false), Truncate: new(true), LateChunking: new(true)}),
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff([][]float32{{0.1, 0.2}, {0.3, 0.4}}, resp.Embeddings); diff != "" {
		t.Errorf("embeddings mismatch (-want +got):\n%s", diff)
	}
	if resp.Usage.Input != 11 {
		t.Errorf("usage input = %d, want 11", resp.Usage.Input)
	}
	got := srv.last(t)
	if got.Path != "/v1/embeddings" {
		t.Errorf("path = %s, want /v1/embeddings", got.Path)
	}
	if auth := got.Header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", auth)
	}
	equalJSON(t, `{"model":"jina-embeddings-v3","input":["a","bc"],"task":"retrieval.passage","dimensions":256,"normalized":false,"truncate":true,"late_chunking":true}`, got.Body)
}

func TestEmbedding_task(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		inputType ai.InputType
		task      string
		wantBody  string
		wantWarn  bool
	}{
		{name: "query on v5", model: "jina-embeddings-v5-text-small", inputType: ai.InputQuery, wantBody: `{"model":"jina-embeddings-v5-text-small","input":["x"],"task":"retrieval.query"}`},
		{name: "task overrides input type", model: "jina-embeddings-v4", inputType: ai.InputQuery, task: "code.query", wantBody: `{"model":"jina-embeddings-v4","input":["x"],"task":"code.query"}`},
		{name: "no input type", model: "jina-embeddings-v3", wantBody: `{"model":"jina-embeddings-v3","input":["x"]}`},
		{name: "input type unmapped for code model", model: "jina-code-embeddings-1.5b", inputType: ai.InputDocument, wantBody: `{"model":"jina-code-embeddings-1.5b","input":["x"]}`, wantWarn: true},
		{name: "code model with explicit task", model: "jina-code-embeddings-1.5b", inputType: ai.InputDocument, task: "nl2code.passage", wantBody: `{"model":"jina-code-embeddings-1.5b","input":["x"],"task":"nl2code.passage"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}],"usage":{"total_tokens":1}}`))
			var provider ai.ProviderData
			if tt.task != "" {
				provider = ai.Providers(jina.EmbeddingOptions{Task: tt.task})
			}
			resp, err := srv.provider().EmbeddingModel(tt.model).Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "x"}}}, InputType: tt.inputType, Provider: provider})
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			equalJSON(t, tt.wantBody, srv.last(t).Body)
			if (len(resp.Warnings) > 0) != tt.wantWarn {
				t.Errorf("warnings = %v, want warning %v", resp.Warnings, tt.wantWarn)
			}
		})
	}
}

func TestEmbedding_oneRequestForAllInputs(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		var b strings.Builder
		b.WriteString(`{"data":[`)
		for i := range req.Input {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(`{"index":` + itoa(len(req.Input)-1-i) + `,"embedding":[` + itoa(i) + `]}`)
		}
		b.WriteString(`],"usage":{"total_tokens":` + itoa(len(req.Input)) + `}}`)
		jsonReply(b.String())(w, r)
	})
	m := srv.provider().EmbeddingModel("jina-embeddings-v3")
	if m.MaxBatch() != 0 {
		t.Fatalf("MaxBatch() = %d, want 0", m.MaxBatch())
	}
	texts := make([]string, 3000)
	for i := range texts {
		texts[i] = "t"
	}
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts, Concurrency: 4})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 3000 || srv.count() != 1 {
		t.Fatalf("got %d embeddings in %d requests, want 3000 in 1", len(res.Embeddings), srv.count())
	}
	if res.Embeddings[0][0] != 2999 || res.Embeddings[2999][0] != 0 {
		t.Errorf("embeddings were not ordered by index: first = %v, last = %v", res.Embeddings[0], res.Embeddings[2999])
	}
}

func TestEmbedding_multimodal(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[2]},{"index":2,"embedding":[3]}],"usage":{"total_tokens":40,"prompt_tokens":10,"image_tokens":30}}`))
	m := srv.provider().EmbeddingModel("jina-embeddings-v5-omni-small")
	_, err := m.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{
		{ai.Text{Text: "caption"}},
		{ai.File{MediaType: "image/png", Data: []byte("png")}},
		{ai.Text{Text: "a cat"}, ai.File{MediaType: "image/jpeg", URL: "https://example.com/cat.jpg"}, ai.File{MediaType: "audio/mpeg", Data: []byte("mp3")}},
	}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"jina-embeddings-v5-omni-small","input":["caption",{"image":"cG5n"},{"content":[{"text":"a cat"},{"image":"https://example.com/cat.jpg"},{"audio":"bXAz"}]}]}`, srv.last(t).Body)
}

func TestEmbedding_pdf(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}],"usage":{"total_tokens":5}}`))
	m := srv.provider().EmbeddingModel("jina-embeddings-v4")
	if _, err := m.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "application/pdf", URL: "https://example.com/a.pdf"}}}}); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"model":"jina-embeddings-v4","input":[{"pdf":"https://example.com/a.pdf"}]}`, srv.last(t).Body)
}

func TestEmbedding_errors(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"data":[{"index":0,"embedding":[1]}]}`))
	m := srv.provider().EmbeddingModel("jina-embeddings-v5-omni-small")
	pdf := ai.File{MediaType: "application/pdf", Data: []byte("x")}
	text := ai.Parts{ai.Text{Text: "a"}}
	tests := []struct {
		name string
		call ai.EmbedCall
		want string
	}{
		{name: "pdf with other inputs", call: ai.EmbedCall{Inputs: []ai.Parts{{pdf}, text}}, want: "on its own"},
		{name: "pdf in a group", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}, pdf}}}, want: "cannot be part"},
		{name: "unsupported media", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "application/zip", Data: []byte("x")}}}}, want: "not supported"},
		{name: "empty file", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.File{MediaType: "image/png"}}}}, want: "neither URL nor data"},
		{name: "empty input", call: ai.EmbedCall{Inputs: []ai.Parts{{}}}, want: "no parts"},
		{name: "unknown input type", call: ai.EmbedCall{Inputs: []ai.Parts{text}, InputType: "x"}, want: "unknown input type"},
		{name: "count mismatch", call: ai.EmbedCall{Inputs: []ai.Parts{text, text}}, want: "got 1 embeddings for 2"},
		{name: "bad options", call: ai.EmbedCall{Inputs: []ai.Parts{text}, Provider: ai.ProviderData{"jina": json.RawMessage(`{"task":1}`)}}, want: "decode provider options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Embed() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}
