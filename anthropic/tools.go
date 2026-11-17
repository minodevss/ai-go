package anthropic

import (
	"cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const (
	webSearchVersion     = "web_search_20260318"
	webFetchVersion      = "web_fetch_20260318"
	codeExecutionVersion = "code_execution_20260521"
)

// WebSearchOptions configure Anthropic's web search tool. Version defaults to the newest tool version.
type WebSearchOptions struct {
	Version        string
	MaxUses        int
	AllowedDomains []string
	BlockedDomains []string
	UserLocation   *UserLocation
	AllowedCallers []string
}

type UserLocation struct {
	City     string `json:"city,omitzero"`
	Region   string `json:"region,omitzero"`
	Country  string `json:"country,omitzero"`
	Timezone string `json:"timezone,omitzero"`
}

// WebFetchOptions configure Anthropic's web fetch tool. Version defaults to the newest tool version.
type WebFetchOptions struct {
	Version          string
	MaxUses          int
	AllowedDomains   []string
	BlockedDomains   []string
	Citations        bool
	MaxContentTokens int
	AllowedCallers   []string
}

type hostedUserLocation struct {
	Type string `json:"type"`
	UserLocation
}

type hostedWebSearch struct {
	Type           string              `json:"type"`
	Name           string              `json:"name"`
	MaxUses        int                 `json:"max_uses,omitzero"`
	AllowedDomains []string            `json:"allowed_domains,omitzero"`
	BlockedDomains []string            `json:"blocked_domains,omitzero"`
	UserLocation   *hostedUserLocation `json:"user_location,omitzero"`
	AllowedCallers []string            `json:"allowed_callers,omitzero"`
}

type citationsConfig struct {
	Enabled bool `json:"enabled"`
}

type hostedWebFetch struct {
	Type             string           `json:"type"`
	Name             string           `json:"name"`
	MaxUses          int              `json:"max_uses,omitzero"`
	AllowedDomains   []string         `json:"allowed_domains,omitzero"`
	BlockedDomains   []string         `json:"blocked_domains,omitzero"`
	Citations        *citationsConfig `json:"citations,omitzero"`
	MaxContentTokens int              `json:"max_content_tokens,omitzero"`
	AllowedCallers   []string         `json:"allowed_callers,omitzero"`
}

// WebSearch declares Anthropic's web search. The API runs it; results arrive as provider-executed ToolCall and ToolResult parts.
func WebSearch(opts WebSearchOptions) ai.Tool {
	cfg := hostedWebSearch{
		Type:           cmp.Or(opts.Version, webSearchVersion),
		Name:           "web_search",
		MaxUses:        opts.MaxUses,
		AllowedDomains: opts.AllowedDomains,
		BlockedDomains: opts.BlockedDomains,
		AllowedCallers: opts.AllowedCallers,
	}
	if len(cfg.AllowedCallers) == 0 {
		cfg.AllowedCallers = []string{"direct"}
	}
	if opts.UserLocation != nil {
		cfg.UserLocation = &hostedUserLocation{Type: "approximate", UserLocation: *opts.UserLocation}
	}
	return hostedTool(cfg.Name, cfg)
}

func WebFetch(opts WebFetchOptions) ai.Tool {
	cfg := hostedWebFetch{
		Type:             cmp.Or(opts.Version, webFetchVersion),
		Name:             "web_fetch",
		MaxUses:          opts.MaxUses,
		AllowedDomains:   opts.AllowedDomains,
		BlockedDomains:   opts.BlockedDomains,
		MaxContentTokens: opts.MaxContentTokens,
		AllowedCallers:   opts.AllowedCallers,
	}
	if opts.Citations {
		cfg.Citations = &citationsConfig{Enabled: true}
	}
	return hostedTool(cfg.Name, cfg)
}

// CodeExecution declares Anthropic's sandboxed code execution. Its container id is reused automatically on later turns.
func CodeExecution() ai.Tool {
	return hostedTool("code_execution", map[string]string{"type": codeExecutionVersion, "name": "code_execution"})
}

func hostedTool(name string, cfg any) ai.Tool {
	return ai.Tool{Name: name, Provider: ai.ProviderData{providerName: pdata.JSON(providerName, cfg)}}
}
