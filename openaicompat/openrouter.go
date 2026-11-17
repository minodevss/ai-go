package openaicompat

import (
	"encoding/json"
	"maps"
	"net/http"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const openRouterName = "openrouter"

// OpenRouterOptions are OpenRouter request settings passed with ai.Providers. Unset fields keep the API default.
type OpenRouterOptions struct {
	Provider  *OpenRouterRouting   `json:"provider,omitzero"`
	Models    []string             `json:"models,omitzero"`
	Reasoning *OpenRouterReasoning `json:"reasoning,omitzero"`
	Plugins   []OpenRouterPlugin   `json:"plugins,omitzero"`
	User      string               `json:"user,omitzero"`
}

// OpenRouterRouting maps to the provider routing object. Sort is price, throughput or latency.
type OpenRouterRouting struct {
	Order             []string         `json:"order,omitzero"`
	Only              []string         `json:"only,omitzero"`
	Ignore            []string         `json:"ignore,omitzero"`
	AllowFallbacks    *bool            `json:"allow_fallbacks,omitzero"`
	RequireParameters *bool            `json:"require_parameters,omitzero"`
	DataCollection    string           `json:"data_collection,omitzero"`
	ZDR               *bool            `json:"zdr,omitzero"`
	Quantizations     []string         `json:"quantizations,omitzero"`
	Sort              string           `json:"sort,omitzero"`
	MaxPrice          *OpenRouterPrice `json:"max_price,omitzero"`
}

// OpenRouterPrice caps prices in dollars per million tokens, or per request or image.
type OpenRouterPrice struct {
	Prompt     *float64 `json:"prompt,omitzero"`
	Completion *float64 `json:"completion,omitzero"`
	Request    *float64 `json:"request,omitzero"`
	Image      *float64 `json:"image,omitzero"`
}

// OpenRouterReasoning maps to the reasoning request object and replaces the call's reasoning effort. Set Effort or MaxTokens, not both.
type OpenRouterReasoning struct {
	Effort    string `json:"effort,omitzero"`
	MaxTokens int    `json:"max_tokens,omitzero"`
	Exclude   *bool  `json:"exclude,omitzero"`
	Enabled   *bool  `json:"enabled,omitzero"`
}

// OpenRouterPlugin enables a plugin such as "web". Settings are sent next to the id.
type OpenRouterPlugin struct {
	ID       string
	Settings map[string]any
}

func (OpenRouterOptions) ProviderName() string { return openRouterName }

func (p *OpenRouterPlugin) MarshalJSON() ([]byte, error) {
	fields := maps.Clone(p.Settings)
	if fields == nil {
		fields = map[string]any{}
	}
	fields["id"] = p.ID
	return json.Marshal(fields)
}

func (p *OpenRouterPlugin) UnmarshalJSON(data []byte) error {
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	id, ok := fields["id"].(string)
	if !ok {
		id = ""
	}
	delete(fields, "id")
	p.ID, p.Settings = id, fields
	return nil
}

// WithAttribution sets the HTTP-Referer and X-OpenRouter-Title headers that identify your app on OpenRouter.
func WithAttribution(referer, title string) Option {
	return func(o *options) {
		if o.headers == nil {
			o.headers = http.Header{}
		}
		o.headers.Set("HTTP-Referer", referer)
		o.headers.Set("X-OpenRouter-Title", title)
	}
}

// OpenRouter builds an OpenRouter provider and reads OPENROUTER_API_KEY.
// The reasoning_details array is stored verbatim on the Reasoning part and sent back unchanged, as OpenRouter requires for tool calling.
func OpenRouter(opts ...Option) *Provider {
	return build(preset{
		name:    openRouterName,
		baseURL: "https://openrouter.ai/api/v1",
		keyEnv:  "OPENROUTER_API_KEY",
		chat: chatcompletions.Dialect{
			TopK:            true,
			NoStreamOptions: true,
			ReasoningFields: []string{"reasoning", "reasoning_content"},
			Replay:          chatcompletions.Replay{Field: "reasoning"},
			Verbatim:        []string{"reasoning_details"},
			Request:         openRouterRequest,
			Response:        openRouterResponse,
		},
	}, opts)
}

func openRouterRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[OpenRouterOptions](openRouterName, c.Provider)
	if err != nil {
		return nil, err
	}
	if opts.Reasoning != nil {
		r.ReasoningEffort = ""
		put(r, "reasoning", opts.Reasoning)
	}
	put(r, "provider", opts.Provider)
	put(r, "models", opts.Models)
	put(r, "plugins", opts.Plugins)
	put(r, "user", opts.User)
	return nil, nil
}

func openRouterResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	data := map[string]json.RawMessage{}
	keep(top, data, "provider", "provider")
	keep(top, data, "model", "model")
	return chatcompletions.ResponseExtras{Data: data}
}
