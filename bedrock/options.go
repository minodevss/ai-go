package bedrock

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const (
	LatencyStandard  = "standard"
	LatencyOptimized = "optimized"

	TierPriority = "priority"
	TierDefault  = "default"
	TierFlex     = "flex"
	TierReserved = "reserved"

	TraceEnabled     = "enabled"
	TraceDisabled    = "disabled"
	TraceEnabledFull = "enabled_full"

	ProcessingSync  = "sync"
	ProcessingAsync = "async"

	TTL5Minutes CacheTTL = "5m"
	TTL1Hour    CacheTTL = "1h"
)

type CacheTTL string

// Options are Converse request settings passed with ai.Providers. Unset fields keep the API default.
type Options struct {
	Guardrail                         *Guardrail        `json:"guardrail,omitzero"`
	AdditionalModelRequestFields      json.RawMessage   `json:"additionalModelRequestFields,omitzero"`
	AdditionalModelResponseFieldPaths []string          `json:"additionalModelResponseFieldPaths,omitzero"`
	PerformanceLatency                string            `json:"performanceLatency,omitzero"`
	ServiceTier                       string            `json:"serviceTier,omitzero"`
	RequestMetadata                   map[string]string `json:"requestMetadata,omitzero"`
	OutputViaTool                     bool              `json:"outputViaTool,omitzero"`
}

func (Options) ProviderName() string { return providerName }

// Guardrail applies a guardrail to the request. StreamProcessingMode applies to streaming calls only.
type Guardrail struct {
	Identifier           string `json:"identifier"`
	Version              string `json:"version"`
	Trace                string `json:"trace,omitzero"`
	StreamProcessingMode string `json:"streamProcessingMode,omitzero"`
}

// EmbeddingOptions are InvokeModel embedding settings passed with ai.Providers. Normalize applies to Titan text models, Truncate to Cohere models (NONE, START, END, LEFT or RIGHT).
type EmbeddingOptions struct {
	Normalize *bool  `json:"normalize,omitzero"`
	Truncate  string `json:"truncate,omitzero"`
}

func (EmbeddingOptions) ProviderName() string { return providerName }

// ResponseData is stored in Response.Provider and in the assistant Message.Provider under "bedrock".
type ResponseData struct {
	AdditionalModelResponseFields json.RawMessage `json:"additionalModelResponseFields,omitzero"`
	Metrics                       json.RawMessage `json:"metrics,omitzero"`
	Trace                         json.RawMessage `json:"trace,omitzero"`
	PerformanceConfig             json.RawMessage `json:"performanceConfig,omitzero"`
	ServiceTier                   json.RawMessage `json:"serviceTier,omitzero"`
}

// CachePoint marks a prompt cache checkpoint after the part, system text or tool that carries the returned data. An empty ttl means the model default of five minutes.
func CachePoint(ttl CacheTTL) ai.ProviderData {
	return pdata.Encode(providerName, partData{CachePoint: &cachePointData{TTL: ttl}})
}

// CachedText is system or user text followed by a cache checkpoint.
func CachedText(s string) ai.Text {
	return ai.Text{Text: s, Provider: CachePoint("")}
}

func CachedTextFor(s string, ttl CacheTTL) ai.Text {
	return ai.Text{Text: s, Provider: CachePoint(ttl)}
}

// SystemTool returns a model-provider tool that the Converse API calls by name, such as a built-in tool of the model.
func SystemTool(name string) ai.Tool {
	return ai.Tool{Name: name, Provider: pdata.Encode(providerName, partData{SystemTool: name})}
}

func optionsOf(d ai.ProviderData) (Options, error) {
	raw, ok := d[providerName]
	if !ok {
		return Options{}, nil
	}
	var o Options
	if err := json.Unmarshal(raw, &o); err != nil {
		return Options{}, fmt.Errorf("bedrock: decode provider options: %w", err)
	}
	return o, nil
}

func embeddingOptionsOf(d ai.ProviderData) (EmbeddingOptions, error) {
	raw, ok := d[providerName]
	if !ok {
		return EmbeddingOptions{}, nil
	}
	var o EmbeddingOptions
	if err := json.Unmarshal(raw, &o); err != nil {
		return EmbeddingOptions{}, fmt.Errorf("bedrock: decode provider options: %w", err)
	}
	return o, nil
}
