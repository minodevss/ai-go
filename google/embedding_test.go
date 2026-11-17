package google_test

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

const studioEmbedReply = `{"embeddings":[{"values":[3,4]},{"values":[0,2]}],"usageMetadata":{"promptTokenCount":6}}`

func TestEmbeddingModel_taskTypes(t *testing.T) {
	tests := []struct {
		name  string
		model string
		in    ai.InputType
		dims  int
		want  string
	}{
		{
			name: "query", model: "gemini-embedding-001", in: ai.InputQuery,
			want: `{"requests":[{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"a"}]},"embedContentConfig":{"taskType":"RETRIEVAL_QUERY"}},{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"b"}]},"embedContentConfig":{"taskType":"RETRIEVAL_QUERY"}}]}`,
		},
		{
			name: "document with dimensions", model: "gemini-embedding-001", in: ai.InputDocument, dims: 768,
			want: `{"requests":[{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"a"}]},"embedContentConfig":{"taskType":"RETRIEVAL_DOCUMENT","outputDimensionality":768}},{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"b"}]},"embedContentConfig":{"taskType":"RETRIEVAL_DOCUMENT","outputDimensionality":768}}]}`,
		},
		{
			name: "unset type", model: "gemini-embedding-001",
			want: `{"requests":[{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"a"}]},"embedContentConfig":{}},{"model":"models/gemini-embedding-001","content":{"parts":[{"text":"b"}]},"embedContentConfig":{}}]}`,
		},
		{
			name: "embedding 2 query uses a prefix", model: "gemini-embedding-2", in: ai.InputQuery,
			want: `{"requests":[{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"task: search result | query: a"}]}},{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"task: search result | query: b"}]}}]}`,
		},
		{
			name: "embedding 2 document uses a prefix", model: "gemini-embedding-2", in: ai.InputDocument, dims: 768,
			want: `{"requests":[{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"title: none | text: a"}]},"embedContentConfig":{"outputDimensionality":768}},{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"title: none | text: b"}]},"embedContentConfig":{"outputDimensionality":768}}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(studioEmbedReply))
			_, err := f.provider().EmbeddingModel(tt.model).Embed(t.Context(), &ai.EmbedCall{
				Inputs:     []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}},
				InputType:  tt.in,
				Dimensions: tt.dims,
			})
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			got := f.last(t)
			if got.Path != "/v1beta/models/"+tt.model+":batchEmbedContents" || got.Header.Get("x-goog-api-key") != "test-key" {
				t.Errorf("request = %s with key %q", got.Path, got.Header.Get("x-goog-api-key"))
			}
			equalJSON(t, tt.want, got.Body)
		})
	}
}

func TestEmbeddingModel_resultAndNormalization(t *testing.T) {
	f := newFakeServer(t, jsonReply(studioEmbedReply))
	res, err := ai.Embed(t.Context(), f.provider().EmbeddingModel("gemini-embedding-001"), ai.EmbedRequest{Texts: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if math.Abs(float64(res.Embeddings[0][0])-0.6) > 1e-6 || math.Abs(float64(res.Embeddings[0][1])-0.8) > 1e-6 || res.Embeddings[1][1] != 1 {
		t.Errorf("embeddings = %v, want L2-normalized vectors", res.Embeddings)
	}
	if res.Usage.Input != 6 {
		t.Errorf("usage = %+v", res.Usage)
	}
	var warned bool
	for _, w := range res.Warnings {
		warned = warned || strings.Contains(w.Message, "normalized")
	}
	if !warned {
		t.Errorf("warnings = %v, want a normalization warning", res.Warnings)
	}

	f2 := newFakeServer(t, jsonReply(studioEmbedReply))
	full, err := ai.Embed(t.Context(), f2.provider().EmbeddingModel("gemini-embedding-001"), ai.EmbedRequest{Texts: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if full.Embeddings[0][0] != 3 {
		t.Errorf("full-size embeddings were changed: %v", full.Embeddings)
	}
}

func TestEmbeddingModel_batchLimit(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"embeddings":[]}`))
	model := f.provider().EmbeddingModel("gemini-embedding-001")
	if model.MaxBatch() != 100 {
		t.Fatalf("MaxBatch() = %d, want 100", model.MaxBatch())
	}
	inputs := make([]ai.Parts, 101)
	for i := range inputs {
		inputs[i] = ai.Parts{ai.Text{Text: "x"}}
	}
	if _, err := model.Embed(t.Context(), &ai.EmbedCall{Inputs: inputs}); err == nil || f.count() != 0 {
		t.Errorf("Embed() error = %v after %d requests, want a limit error and no request", err, f.count())
	}
}

func TestEmbeddingModel_splitsLargeRequests(t *testing.T) {
	f := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Requests []json.RawMessage `json:"requests"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Decode() error = %v", err)
		}
		items := make([]string, len(req.Requests))
		for i := range items {
			items[i] = `{"values":[1]}`
		}
		_, _ = io.WriteString(w, `{"embeddings":[`+strings.Join(items, ",")+`]}`)
	})
	texts := make([]string, 250)
	for i := range texts {
		texts[i] = "x"
	}
	res, err := ai.Embed(t.Context(), f.provider().EmbeddingModel("gemini-embedding-001"), ai.EmbedRequest{Texts: texts})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 250 || f.count() != 3 {
		t.Errorf("got %d embeddings in %d requests, want 250 in 3", len(res.Embeddings), f.count())
	}
}

func TestEmbeddingModel_multimodal(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"embeddings":[{"values":[1]}]}`))
	in := ai.Parts{ai.Text{Text: "caption"}, ai.File{MediaType: "image/png", Data: []byte{1, 2}}}
	if _, err := f.provider().EmbeddingModel("gemini-embedding-2").Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{in}, InputType: ai.InputDocument}); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	equalJSON(t, `{"requests":[{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"caption"},{"inlineData":{"mimeType":"image/png","data":"AQI="}}]}}]}`, f.last(t).Body)

	if _, err := f.provider().EmbeddingModel("gemini-embedding-001").Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{in}}); err == nil {
		t.Error("Embed() with an image on a text-only model error = nil, want an error")
	}
	if _, err := f.provider().EmbeddingModel("gemini-embedding-001").Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{}}}); err == nil {
		t.Error("Embed() with an empty input error = nil, want an error")
	}
}

func TestEmbeddingModel_countMismatch(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"embeddings":[{"values":[1]}]}`))
	_, err := f.provider().EmbeddingModel("gemini-embedding-001").Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}})
	if err == nil {
		t.Error("Embed() error = nil, want a count mismatch error")
	}
}

func TestEmbeddingModel_identity(t *testing.T) {
	m := google.New().EmbeddingModel("gemini-embedding-001")
	if m.Provider() != "google" || m.ModelID() != "gemini-embedding-001" {
		t.Errorf("identity = %q/%q", m.Provider(), m.ModelID())
	}
}
