package google

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
)

// Options are Gemini request settings passed with ai.Providers under "google", also on Vertex.
// ThinkingBudget and ThinkingLevel override ai.TextRequest.Reasoning and exclude each other.
type Options struct {
	ThinkingBudget            *int              `json:"thinkingBudget,omitzero"`
	ThinkingLevel             string            `json:"thinkingLevel,omitzero"`
	IncludeThoughts           *bool             `json:"includeThoughts,omitzero"`
	SafetySettings            []SafetySetting   `json:"safetySettings,omitzero"`
	CachedContent             string            `json:"cachedContent,omitzero"`
	ResponseModalities        []string          `json:"responseModalities,omitzero"`
	Labels                    map[string]string `json:"labels,omitzero"`
	ServiceTier               string            `json:"serviceTier,omitzero"`
	ServerSideToolInvocations bool              `json:"serverSideToolInvocations,omitzero"`
	OpenAPISchemas            bool              `json:"openApiSchemas,omitzero"`
}

type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

func (Options) ProviderName() string { return providerName }

type hostedConfig struct {
	Hosted map[string]json.RawMessage `json:"hosted"`
}

func optionsOf(d ai.ProviderData) (Options, error) {
	raw, ok := d[providerName]
	if !ok {
		return Options{}, nil
	}
	var o Options
	if err := json.Unmarshal(raw, &o); err != nil {
		return Options{}, fmt.Errorf("google: decode provider options: %w", err)
	}
	return o, nil
}

// Search declares grounding with Google Search; grounding metadata lands in the response Provider data and as Text citations.
func Search() ai.Tool { return hostedTool("google_search", "googleSearch", struct{}{}) }

// URLContext lets the model read the URLs it finds in the prompt.
func URLContext() ai.Tool { return hostedTool("url_context", "urlContext", struct{}{}) }

// CodeExecution lets the model run Python; runs arrive as provider-executed ToolCall and ToolResult parts.
func CodeExecution() ai.Tool { return hostedTool("code_execution", "codeExecution", struct{}{}) }

// Maps declares grounding with Google Maps.
func Maps() ai.Tool { return hostedTool("google_maps", "googleMaps", struct{}{}) }

// FileSearch grounds answers in the named File Search stores, for example "fileSearchStores/docs-123".
func FileSearch(storeNames ...string) ai.Tool {
	return hostedTool("file_search", "fileSearch", map[string][]string{"fileSearchStoreNames": storeNames})
}

func hostedTool(name, field string, cfg any) ai.Tool {
	hosted := hostedConfig{Hosted: map[string]json.RawMessage{field: pdata.JSON(providerName, cfg)}}
	return ai.Tool{Name: name, Provider: pdata.Encode(providerName, hosted)}
}

func hostedToolOf(t ai.Tool) (map[string]json.RawMessage, bool) {
	raw, ok := t.Provider[providerName]
	if !ok {
		return nil, false
	}
	var cfg hostedConfig
	if err := json.Unmarshal(raw, &cfg); err != nil || len(cfg.Hosted) == 0 {
		return nil, false
	}
	return cfg.Hosted, true
}
