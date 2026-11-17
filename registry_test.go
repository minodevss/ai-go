package ai_test

import (
	"errors"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type testProvider struct {
	models map[string]ai.LanguageModel
}

func (p testProvider) LanguageModel(id string) (ai.LanguageModel, error) {
	if m, ok := p.models[id]; ok {
		return m, nil
	}
	return nil, &ai.NoSuchModelError{Spec: id, Kind: "language"}
}

type fullProvider struct {
	testProvider
	embedding ai.EmbeddingModel
	rerank    ai.RerankModel
}

func (p fullProvider) EmbeddingModel(string) (ai.EmbeddingModel, error) { return p.embedding, nil }

func (p fullProvider) RerankModel(string) (ai.RerankModel, error) { return p.rerank, nil }

func TestRegistry_languageModel(t *testing.T) {
	opus := aitest.NewModel()
	reg := ai.Registry{
		Providers: map[string]ai.Provider{
			"anthropic": testProvider{models: map[string]ai.LanguageModel{"claude-opus-5-5": opus}},
		},
		Aliases: map[string]string{"smart": "anthropic:claude-opus-5-5", "broken": "nowhere:x"},
	}
	tests := []struct {
		name    string
		spec    string
		want    ai.LanguageModel
		wantErr bool
	}{
		{name: "provider:model", spec: "anthropic:claude-opus-5-5", want: opus},
		{name: "alias", spec: "smart", want: opus},
		{name: "unknown model", spec: "anthropic:nope", wantErr: true},
		{name: "unknown provider", spec: "openai:gpt", wantErr: true},
		{name: "no colon", spec: "claude", wantErr: true},
		{name: "empty id", spec: "anthropic:", wantErr: true},
		{name: "alias to unknown provider", spec: "broken", wantErr: true},
		{name: "empty", spec: "", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := reg.LanguageModel(tt.spec)
			if tt.wantErr {
				if _, ok := errors.AsType[*ai.NoSuchModelError](err); !ok {
					t.Fatalf("error = %v, want a NoSuchModelError", err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("LanguageModel(%q) = %v, %v; want %v", tt.spec, got, err, tt.want)
			}
		})
	}
}

func TestRegistry_modelIDMayContainColons(t *testing.T) {
	m := aitest.NewModel()
	reg := ai.Registry{Providers: map[string]ai.Provider{"ollama": testProvider{models: map[string]ai.LanguageModel{"llama3:8b": m}}}}
	if got, err := reg.LanguageModel("ollama:llama3:8b"); err != nil || got != ai.LanguageModel(m) {
		t.Errorf("LanguageModel() = %v, %v", got, err)
	}
}

func TestRegistry_otherKinds(t *testing.T) {
	emb, rr := aitest.NewEmbeddingModel(), aitest.NewRerankModel()
	reg := ai.Registry{Providers: map[string]ai.Provider{
		"full":  fullProvider{embedding: emb, rerank: rr},
		"plain": testProvider{},
	}}
	if got, err := reg.EmbeddingModel("full:e"); err != nil || got != ai.EmbeddingModel(emb) {
		t.Errorf("EmbeddingModel() = %v, %v", got, err)
	}
	if got, err := reg.RerankModel("full:r"); err != nil || got != ai.RerankModel(rr) {
		t.Errorf("RerankModel() = %v, %v", got, err)
	}
	if _, err := reg.EmbeddingModel("plain:e"); err == nil {
		t.Error("EmbeddingModel on a language-only provider error = nil")
	}
	if _, err := reg.RerankModel("plain:r"); err == nil {
		t.Error("RerankModel on a language-only provider error = nil")
	}
	if _, err := reg.EmbeddingModel("missing:e"); err == nil {
		t.Error("EmbeddingModel on an unknown provider error = nil")
	}
	if _, err := reg.RerankModel("missing:r"); err == nil {
		t.Error("RerankModel on an unknown provider error = nil")
	}
}

func TestRegistry_zeroValue(t *testing.T) {
	var reg ai.Registry
	if _, err := reg.LanguageModel("a:b"); err == nil {
		t.Error("zero Registry error = nil, want NoSuchModelError")
	}
}
