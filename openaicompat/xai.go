package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const xaiName = "xai"

// XAIOptions are xAI request settings passed with ai.Providers. Unset fields keep the API default.
type XAIOptions struct {
	ParallelToolCalls *bool  `json:"parallelToolCalls,omitzero"`
	PromptCacheKey    string `json:"promptCacheKey,omitzero"`
	SafetyIdentifier  string `json:"safetyIdentifier,omitzero"`
	ServiceTier       string `json:"serviceTier,omitzero"`
	User              string `json:"user,omitzero"`
}

func (XAIOptions) ProviderName() string { return xaiName }

// XAI builds an xAI provider and reads XAI_API_KEY. Grok reasoning cannot be turned off, so ReasoningNone is dropped with a warning and ReasoningMinimal becomes low.
func XAI(opts ...Option) *Provider {
	return build(preset{
		name:    xaiName,
		baseURL: "https://api.x.ai/v1",
		keyEnv:  "XAI_API_KEY",
		chat: chatcompletions.Dialect{
			ReasoningFields: []string{"reasoning_content"},
			Request:         xaiRequest,
			Usage:           xaiUsage,
			Response:        xaiResponse,
		},
	}, opts)
}

func xaiRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[XAIOptions](xaiName, c.Provider)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	switch c.Reasoning {
	case ai.ReasoningNone:
		r.ReasoningEffort = ""
		warnings = append(warnings, ai.Warning{Message: "xai: reasoning cannot be disabled; reasoning effort none was ignored"})
	case ai.ReasoningMinimal:
		r.ReasoningEffort = "low"
	case ai.ReasoningMax:
		r.ReasoningEffort = "xhigh"
	default:
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "prompt_cache_key", opts.PromptCacheKey)
	put(r, "safety_identifier", opts.SafetyIdentifier)
	put(r, "service_tier", opts.ServiceTier)
	put(r, "user", opts.User)
	return warnings, nil
}

// xAI reports reasoning_tokens next to completion_tokens, not inside them: https://docs.x.ai/docs/guides/reasoning
func xaiUsage(raw json.RawMessage) ai.Usage {
	u, ok := decodeUsage(raw)
	if !ok {
		return standardUsage(raw)
	}
	reasoning := u.CompletionDetails.ReasoningTokens
	return ai.Usage{Input: u.PromptTokens, Output: u.CompletionTokens + reasoning, CacheRead: u.cacheRead(), Reasoning: reasoning, Raw: raw}
}

func xaiResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	var urls []string
	if err := json.Unmarshal(top["citations"], &urls); err != nil {
		return chatcompletions.ResponseExtras{}
	}
	return chatcompletions.ResponseExtras{Citations: urlCitations(urls, nil)}
}
