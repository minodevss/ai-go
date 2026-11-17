package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const cerebrasName = "cerebras"

// CerebrasOptions are passed with ai.Providers. ReasoningFormat is parsed, raw, hidden or none; ClearThinking applies to qwen-3.8-27b and glm-5.3-flash only.
type CerebrasOptions struct {
	ReasoningFormat   string `json:"reasoningFormat,omitzero"`
	ClearThinking     *bool  `json:"clearThinking,omitzero"`
	ServiceTier       string `json:"serviceTier,omitzero"`
	PromptCacheKey    string `json:"promptCacheKey,omitzero"`
	ParallelToolCalls *bool  `json:"parallelToolCalls,omitzero"`
}

func (CerebrasOptions) ProviderName() string { return cerebrasName }

// Cerebras builds a Cerebras provider and reads CEREBRAS_API_KEY. The time_info timings of a response are kept as timeInfo in its provider data.
func Cerebras(opts ...Option) *Provider {
	return build(preset{
		name:    cerebrasName,
		baseURL: "https://api.cerebras.ai/v1",
		keyEnv:  "CEREBRAS_API_KEY",
		chat: chatcompletions.Dialect{
			NoStreamOptions: true,
			ToolStrict:      true,
			ReasoningFields: []string{"reasoning"},
			Request:         cerebrasRequest,
			Response:        cerebrasResponse,
		},
	}, opts)
}

func cerebrasRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[CerebrasOptions](cerebrasName, c.Provider)
	if err != nil {
		return nil, err
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "reasoning_format", opts.ReasoningFormat)
	put(r, "clear_thinking", opts.ClearThinking)
	put(r, "service_tier", opts.ServiceTier)
	put(r, "prompt_cache_key", opts.PromptCacheKey)
	return nil, nil
}

func cerebrasResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	data := map[string]json.RawMessage{}
	keep(top, data, "time_info", "timeInfo")
	return chatcompletions.ResponseExtras{Data: data}
}
