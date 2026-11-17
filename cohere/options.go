package cohere

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
)

const (
	ThinkingEnabled  = "enabled"
	ThinkingDisabled = "disabled"

	CitationsEnabled  = "ENABLED"
	CitationsDisabled = "DISABLED"
	CitationsFast     = "FAST"
	CitationsAccurate = "ACCURATE"
	CitationsOff      = "OFF"

	SafetyContextual = "CONTEXTUAL"
	SafetyStrict     = "STRICT"
	SafetyOff        = "OFF"

	TruncateNone  = "NONE"
	TruncateStart = "START"
	TruncateEnd   = "END"

	InputClassification = "classification"
	InputClustering     = "clustering"
	InputImage          = "image"
)

// Options are Cohere chat settings passed with ai.Providers. Unset fields keep the API default.
type Options struct {
	Documents    []Document `json:"documents,omitzero"`
	CitationMode string     `json:"citationMode,omitzero"`
	SafetyMode   string     `json:"safetyMode,omitzero"`
	Thinking     *Thinking  `json:"thinking,omitzero"`
	StrictTools  bool       `json:"strictTools,omitzero"`
	Priority     int        `json:"priority,omitzero"`
}

func (Options) ProviderName() string { return providerName }

// EmbeddingOptions are Cohere embedding settings passed with ai.Providers. InputType overrides the type derived from ai.EmbedRequest.InputType.
type EmbeddingOptions struct {
	InputType string `json:"inputType,omitzero"`
	MaxTokens int    `json:"maxTokens,omitzero"`
	Truncate  string `json:"truncate,omitzero"`
	Priority  int    `json:"priority,omitzero"`
}

func (EmbeddingOptions) ProviderName() string { return providerName }

// RerankOptions are Cohere rerank settings passed with ai.Providers.
type RerankOptions struct {
	MaxTokensPerDoc int `json:"maxTokensPerDoc,omitzero"`
	Priority        int `json:"priority,omitzero"`
}

func (RerankOptions) ProviderName() string { return providerName }

// Thinking overrides the thinking mode derived from ai.TextRequest.Reasoning. TokenBudget caps the thinking tokens when positive.
type Thinking struct {
	Type        string `json:"type"`
	TokenBudget int    `json:"tokenBudget,omitzero"`
}

// Document is a source the model can cite. An empty ID is replaced by the document's position in the request.
type Document struct {
	ID   string         `json:"id,omitzero"`
	Data map[string]any `json:"data"`
}

// ResponseData is stored in Response.Provider and in the assistant Message.Provider under "cohere". Citations belong to thinking and plan blocks, which have no text part.
type ResponseData struct {
	ID        string            `json:"id,omitzero"`
	Citations []json.RawMessage `json:"citations,omitzero"`
}

func optionsOf[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("cohere: decode provider options: %w", err)
	}
	return o, nil
}
