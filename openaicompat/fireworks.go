package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const fireworksName = "fireworks"

// FireworksOptions are Fireworks AI request settings passed with ai.Providers. Thinking and the call's reasoning effort exclude each other; Thinking wins.
type FireworksOptions struct {
	Thinking                      *FireworksThinking `json:"thinking,omitzero"`
	ReasoningHistory              string             `json:"reasoningHistory,omitzero"`
	MinP                          *float64           `json:"minP,omitzero"`
	RepetitionPenalty             *float64           `json:"repetitionPenalty,omitzero"`
	ParallelToolCalls             *bool              `json:"parallelToolCalls,omitzero"`
	PromptCacheKey                string             `json:"promptCacheKey,omitzero"`
	PromptCacheIsolationKey       string             `json:"promptCacheIsolationKey,omitzero"`
	ContextLengthExceededBehavior string             `json:"contextLengthExceededBehavior,omitzero"`
	PerfMetricsInResponse         *bool              `json:"perfMetricsInResponse,omitzero"`
}

// FireworksThinking maps to the thinking request object. Type is enabled, disabled or adaptive; BudgetTokens is at least 1024.
type FireworksThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitzero"`
}

func (FireworksOptions) ProviderName() string { return fireworksName }

// Fireworks builds a Fireworks AI provider and reads FIREWORKS_API_KEY. Model ids look like accounts/fireworks/models/name.
func Fireworks(opts ...Option) *Provider {
	return build(preset{
		name:    fireworksName,
		baseURL: "https://api.fireworks.ai/inference/v1",
		keyEnv:  "FIREWORKS_API_KEY",
		chat: chatcompletions.Dialect{
			MaxTokens:       chatcompletions.MaxTokens,
			TopK:            true,
			ReasoningFields: []string{"reasoning_content"},
			Request:         fireworksRequest,
			Response:        fireworksResponse,
		},
	}, opts)
}

func fireworksRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[FireworksOptions](fireworksName, c.Provider)
	if err != nil {
		return nil, err
	}
	if opts.Thinking != nil {
		r.ReasoningEffort = ""
		put(r, "thinking", opts.Thinking)
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "reasoning_history", opts.ReasoningHistory)
	put(r, "min_p", opts.MinP)
	put(r, "repetition_penalty", opts.RepetitionPenalty)
	put(r, "prompt_cache_key", opts.PromptCacheKey)
	put(r, "prompt_cache_isolation_key", opts.PromptCacheIsolationKey)
	put(r, "context_length_exceeded_behavior", opts.ContextLengthExceededBehavior)
	put(r, "perf_metrics_in_response", opts.PerfMetricsInResponse)
	return nil, nil
}

func fireworksResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	data := map[string]json.RawMessage{}
	keep(top, data, "perf_metrics", "perfMetrics")
	return chatcompletions.ResponseExtras{Data: data}
}
