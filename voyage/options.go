package voyage

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
)

// EmbeddingOptions are Voyage embedding settings passed with ai.Providers. Unset fields keep the API default.
type EmbeddingOptions struct {
	Truncation *bool `json:"truncation,omitzero"`
}

func (EmbeddingOptions) ProviderName() string { return providerName }

// RerankOptions are Voyage rerank settings passed with ai.Providers. Unset fields keep the API default.
type RerankOptions struct {
	Truncation *bool `json:"truncation,omitzero"`
}

func (RerankOptions) ProviderName() string { return providerName }

func optionsOf[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("voyage: decode provider options: %w", err)
	}
	return o, nil
}
