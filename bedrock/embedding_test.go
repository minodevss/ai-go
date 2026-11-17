package bedrock_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
)

func TestEmbeddingModel_Embed(t *testing.T) {
	image := ai.File{MediaType: "image/png", Data: []byte("png")}
	tests := []struct {
		name        string
		model       string
		call        ai.EmbedCall
		reply       string
		wantPath    string
		wantBody    string
		wantVectors [][]float32
		wantTokens  int
		wantWarn    string
	}{
		{
			name:        "titan text v2",
			model:       "amazon.titan-embed-text-v2:0",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "hello"}}}, Dimensions: 512, Provider: ai.Providers(bedrock.EmbeddingOptions{Normalize: new(false)})},
			reply:       `{"embedding":[0.1,0.2],"inputTextTokenCount":3}`,
			wantPath:    "/model/amazon.titan-embed-text-v2%3A0/invoke",
			wantBody:    `{"inputText":"hello","dimensions":512,"normalize":false}`,
			wantVectors: [][]float32{{0.1, 0.2}},
			wantTokens:  3,
		},
		{
			name:        "titan text v2 with embeddings by type only",
			model:       "amazon.titan-embed-text-v2:0",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "hello"}}}},
			reply:       `{"inputTextTokenCount":2,"embeddingsByType":{"float":[0.5]}}`,
			wantPath:    "/model/amazon.titan-embed-text-v2%3A0/invoke",
			wantBody:    `{"inputText":"hello"}`,
			wantVectors: [][]float32{{0.5}},
			wantTokens:  2,
		},
		{
			name:        "titan text g1 ignores dimensions",
			model:       "amazon.titan-embed-text-v1",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "hello"}}}, Dimensions: 256},
			reply:       `{"embedding":[1],"inputTextTokenCount":1}`,
			wantPath:    "/model/amazon.titan-embed-text-v1/invoke",
			wantBody:    `{"inputText":"hello"}`,
			wantVectors: [][]float32{{1}},
			wantTokens:  1,
			wantWarn:    "dimensions",
		},
		{
			name:        "titan multimodal text and image",
			model:       "amazon.titan-embed-image-v1",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a cat"}, image}}, Dimensions: 384},
			reply:       `{"embedding":[0.3],"inputTextTokenCount":2}`,
			wantPath:    "/model/amazon.titan-embed-image-v1/invoke",
			wantBody:    `{"inputText":"a cat","inputImage":"cG5n","embeddingConfig":{"outputEmbeddingLength":384}}`,
			wantVectors: [][]float32{{0.3}},
			wantTokens:  2,
		},
		{
			name:        "cohere v3 texts",
			model:       "cohere.embed-english-v3",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}, InputType: ai.InputQuery, Provider: ai.Providers(bedrock.EmbeddingOptions{Truncate: "END"})},
			reply:       `{"id":"1","embeddings":[[0.1],[0.2]],"response_type":"embeddings_floats","texts":["a","b"]}`,
			wantPath:    "/model/cohere.embed-english-v3/invoke",
			wantBody:    `{"input_type":"search_query","texts":["a","b"],"truncate":"END"}`,
			wantVectors: [][]float32{{0.1}, {0.2}},
		},
		{
			name:        "cohere v3 image",
			model:       "cohere.embed-multilingual-v3",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{image}}},
			reply:       `{"embeddings":[[0.9]],"response_type":"embeddings_floats","images":[{"width":1,"height":1,"format":"png","bit_depth":8}]}`,
			wantPath:    "/model/cohere.embed-multilingual-v3/invoke",
			wantBody:    `{"input_type":"image","images":["data:image/png;base64,cG5n"]}`,
			wantVectors: [][]float32{{0.9}},
		},
		{
			name:        "cohere v4 texts with dimension and by-type response",
			model:       "us.cohere.embed-v4:0",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}, Dimensions: 1024},
			reply:       `{"id":"1","embeddings":{"float":[[0.4]]},"response_type":"embeddings_by_type"}`,
			wantPath:    "/model/us.cohere.embed-v4%3A0/invoke",
			wantBody:    `{"input_type":"search_document","texts":["a"],"output_dimension":1024}`,
			wantVectors: [][]float32{{0.4}},
		},
		{
			name:        "cohere v4 interleaved",
			model:       "cohere.embed-v4:0",
			call:        ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "caption"}, image}}},
			reply:       `{"embeddings":[[0.7]],"response_type":"embeddings_floats"}`,
			wantPath:    "/model/cohere.embed-v4%3A0/invoke",
			wantBody:    `{"input_type":"search_document","inputs":[{"content":[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}}]}]}`,
			wantVectors: [][]float32{{0.7}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(tt.reply))
			m := f.provider().EmbeddingModel(tt.model)
			got, err := m.Embed(t.Context(), &tt.call)
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			if f.last(t).Path != tt.wantPath {
				t.Errorf("path = %q, want %q", f.last(t).Path, tt.wantPath)
			}
			equalJSON(t, tt.wantBody, f.last(t).Body)
			if diff := cmp.Diff(tt.wantVectors, got.Embeddings); diff != "" {
				t.Errorf("Embed() vectors mismatch (-want +got):\n%s", diff)
			}
			if got.Usage.Input != tt.wantTokens {
				t.Errorf("Usage.Input = %d, want %d", got.Usage.Input, tt.wantTokens)
			}
			if tt.wantWarn != "" && (len(got.Warnings) != 1 || !strings.Contains(got.Warnings[0].Message, tt.wantWarn)) {
				t.Errorf("Warnings = %v, want one about %q", got.Warnings, tt.wantWarn)
			}
		})
	}
}

func TestEmbeddingModel_Embed_throughAiEmbed(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"embedding":[0.1,0.2],"inputTextTokenCount":4}`))
	m := f.provider().EmbeddingModel("amazon.titan-embed-text-v2:0")
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 3 || f.count() != 3 || res.Usage.Input != 12 {
		t.Errorf("Embed() = %d vectors, %d requests, %d tokens, want 3, 3 (Titan takes one input per call), 12", len(res.Embeddings), f.count(), res.Usage.Input)
	}
}

func TestEmbeddingModel_MaxBatch(t *testing.T) {
	p := bedrock.New()
	tests := []struct {
		model string
		want  int
	}{
		{model: "amazon.titan-embed-text-v2:0", want: 1},
		{model: "cohere.embed-english-v3", want: 96},
		{model: "cohere.embed-v4:0", want: 96},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := p.EmbeddingModel(tt.model).MaxBatch(); got != tt.want {
				t.Errorf("MaxBatch() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestEmbeddingModel_Embed_errors(t *testing.T) {
	image := ai.File{MediaType: "image/png", Data: []byte("png")}
	tests := []struct {
		name  string
		model string
		call  ai.EmbedCall
		reply string
		want  string
	}{
		{name: "unknown family", model: "meta.llama3", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}}, reply: `{}`, want: "unsupported embedding model"},
		{name: "titan text rejects images", model: "amazon.titan-embed-text-v2:0", call: ai.EmbedCall{Inputs: []ai.Parts{{image}}}, reply: `{}`, want: "text only"},
		{name: "empty input", model: "amazon.titan-embed-text-v2:0", call: ai.EmbedCall{Inputs: []ai.Parts{{}}}, reply: `{}`, want: "empty"},
		{name: "cohere v3 rejects mixed input", model: "cohere.embed-english-v3", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}, image}}}, reply: `{}`, want: "one image per call"},
		{name: "titan response without a vector", model: "amazon.titan-embed-image-v1", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}}, reply: `{"message":"Invalid image"}`, want: "Invalid image"},
		{name: "cohere count mismatch", model: "cohere.embed-english-v3", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}, {ai.Text{Text: "b"}}}}, reply: `{"embeddings":[[1]]}`, want: "1 vectors for 2 inputs"},
		{name: "cohere without float vectors", model: "cohere.embed-english-v3", call: ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}}, reply: `{"embeddings":{"int8":[[1]]}}`, want: "no float vectors"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(tt.reply))
			_, err := f.provider().EmbeddingModel(tt.model).Embed(t.Context(), &tt.call)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Embed() error = %v, want %q", err, tt.want)
			}
		})
	}
}
