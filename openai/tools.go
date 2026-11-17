package openai

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

// WebSearchOptions configure the hosted web search tool; zero fields are omitted.
type WebSearchOptions struct {
	SearchContextSize string        `json:"search_context_size,omitzero"`
	AllowedDomains    []string      `json:"-"`
	UserLocation      *UserLocation `json:"user_location,omitzero"`
}

type UserLocation struct {
	Type     string `json:"type"`
	Country  string `json:"country,omitzero"`
	City     string `json:"city,omitzero"`
	Region   string `json:"region,omitzero"`
	Timezone string `json:"timezone,omitzero"`
}

// MCPOptions configure a remote MCP server the model calls on OpenAI's side.
type MCPOptions struct {
	Label           string   `json:"server_label"`
	URL             string   `json:"server_url,omitzero"`
	ConnectorID     string   `json:"connector_id,omitzero"`
	Authorization   string   `json:"authorization,omitzero"`
	AllowedTools    []string `json:"allowed_tools,omitzero"`
	RequireApproval string   `json:"require_approval,omitzero"`
}

// WebSearch declares OpenAI hosted web search; results arrive as provider-executed tool parts.
func WebSearch(opts WebSearchOptions) ai.Tool {
	cfg := map[string]any{"type": "web_search"}
	if opts.SearchContextSize != "" {
		cfg["search_context_size"] = opts.SearchContextSize
	}
	if opts.UserLocation != nil {
		cfg["user_location"] = opts.UserLocation
	}
	if len(opts.AllowedDomains) > 0 {
		cfg["filters"] = map[string]any{"allowed_domains": opts.AllowedDomains}
	}
	return hostedTool("web_search", cfg)
}

func FileSearch(vectorStoreIDs ...string) ai.Tool {
	return hostedTool("file_search", map[string]any{"type": "file_search", "vector_store_ids": vectorStoreIDs})
}

func CodeInterpreter() ai.Tool {
	return hostedTool("code_interpreter", map[string]any{"type": "code_interpreter", "container": map[string]string{"type": "auto"}})
}

func ImageGeneration() ai.Tool {
	return hostedTool("image_generation", map[string]any{"type": "image_generation"})
}

func MCP(opts MCPOptions) ai.Tool {
	cfg := map[string]any{"type": "mcp"}
	raw := pdata.JSON(providerName, opts)
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err == nil {
		for k, v := range fields {
			cfg[k] = v
		}
	}
	return hostedTool("mcp."+opts.Label, cfg)
}

func hostedTool(name string, cfg map[string]any) ai.Tool {
	return ai.Tool{Name: name, Provider: pdata.Encode(providerName, cfg)}
}
