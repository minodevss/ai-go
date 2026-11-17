package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const togetherName = "together"

// TogetherOptions are Together AI request settings passed with ai.Providers. Unset fields keep the API default.
type TogetherOptions struct {
	Reasoning                     *TogetherReasoning `json:"reasoning,omitzero"`
	MinP                          *float64           `json:"minP,omitzero"`
	RepetitionPenalty             *float64           `json:"repetitionPenalty,omitzero"`
	SafetyModel                   string             `json:"safetyModel,omitzero"`
	ContextLengthExceededBehavior string             `json:"contextLengthExceededBehavior,omitzero"`
	ChatTemplateKwargs            map[string]any     `json:"chatTemplateKwargs,omitzero"`
}

// TogetherReasoning maps to the reasoning request object.
type TogetherReasoning struct {
	Enabled *bool `json:"enabled,omitzero"`
}

func (TogetherOptions) ProviderName() string { return togetherName }

// Together builds a Together AI provider and reads TOGETHER_API_KEY. The finish_reason eos counts as a normal stop.
func Together(opts ...Option) *Provider {
	return build(preset{
		name:    togetherName,
		baseURL: "https://api.together.ai/v1",
		keyEnv:  "TOGETHER_API_KEY",
		chat: chatcompletions.Dialect{
			MaxTokens:       chatcompletions.MaxTokens,
			NoStreamOptions: true,
			TopK:            true,
			Request:         togetherRequest,
			Usage:           togetherUsage,
			Response:        togetherResponse,
		},
	}, opts)
}

func togetherRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[TogetherOptions](togetherName, c.Provider)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	if r.ParallelToolCalls != nil {
		r.ParallelToolCalls = nil
		warnings = append(warnings, ignored(togetherName, "parallelToolCalls"))
	}
	put(r, "reasoning", opts.Reasoning)
	put(r, "min_p", opts.MinP)
	put(r, "repetition_penalty", opts.RepetitionPenalty)
	put(r, "safety_model", opts.SafetyModel)
	put(r, "context_length_exceeded_behavior", opts.ContextLengthExceededBehavior)
	put(r, "chat_template_kwargs", opts.ChatTemplateKwargs)
	return warnings, nil
}

func togetherUsage(raw json.RawMessage) ai.Usage {
	u, ok := decodeUsage(raw)
	if !ok {
		return standardUsage(raw)
	}
	return ai.Usage{Input: u.PromptTokens, Output: u.CompletionTokens, CacheRead: u.cacheRead(), Reasoning: u.CompletionDetails.ReasoningTokens, Raw: raw}
}

func togetherResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	var warnings []struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(top["warnings"], &warnings); err != nil {
		return chatcompletions.ResponseExtras{}
	}
	out := chatcompletions.ResponseExtras{}
	for _, w := range warnings {
		out.Warnings = append(out.Warnings, ai.Warning{Message: togetherName + ": " + w.Message})
	}
	return out
}
