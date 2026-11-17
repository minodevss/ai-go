package jina

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
)

// EmbeddingOptions are Jina embedding settings passed with ai.Providers. Task overrides the task derived from the input type, and LateChunking needs all inputs in one request.
type EmbeddingOptions struct {
	Task         string `json:"task,omitzero"`
	Normalized   *bool  `json:"normalized,omitzero"`
	Truncate     *bool  `json:"truncate,omitzero"`
	LateChunking *bool  `json:"lateChunking,omitzero"`
}

func (EmbeddingOptions) ProviderName() string { return providerName }

// RerankOptions are Jina rerank settings passed with ai.Providers. MaxDocLength applies to the jina-reranker-v3 family only.
type RerankOptions struct {
	MaxDocLength int `json:"maxDocLength,omitzero"`
}

func (RerankOptions) ProviderName() string { return providerName }

func decodeOptions[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("jina: decode provider options: %w", err)
	}
	return o, nil
}
