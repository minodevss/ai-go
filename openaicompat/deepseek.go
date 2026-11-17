package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const deepseekName = "deepseek"

// DeepSeekThinkingEnabled and DeepSeekThinkingDisabled are the Thinking values of DeepSeekOptions. They override the call's reasoning effort.
const (
	DeepSeekThinkingEnabled  = "enabled"
	DeepSeekThinkingDisabled = "disabled"
)

// DeepSeekOptions are DeepSeek request settings passed with ai.Providers. Unset fields keep the API default.
type DeepSeekOptions struct {
	Thinking string `json:"thinking,omitzero"`
	UserID   string `json:"userId,omitzero"`
}

func (DeepSeekOptions) ProviderName() string { return deepseekName }

// DeepSeek builds a DeepSeek provider and reads DEEPSEEK_API_KEY. Structured output uses json_object with the schema in a system message.
// Reasoning is always replayed because the API rejects tool requests that lack an earlier reasoning_content.
func DeepSeek(opts ...Option) *Provider {
	return build(preset{
		name:    deepseekName,
		baseURL: "https://api.deepseek.com",
		keyEnv:  "DEEPSEEK_API_KEY",
		chat: chatcompletions.Dialect{
			MaxTokens:       chatcompletions.MaxTokens,
			Replay:          chatcompletions.Replay{Field: "reasoning_content"},
			ReasoningFields: []string{"reasoning_content"},
			Request:         deepseekRequest,
			Usage:           deepseekUsage,
		},
	}, opts)
}

func deepseekRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[DeepSeekOptions](deepseekName, c.Provider)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	if r.Seed != nil {
		r.Seed = nil
		warnings = append(warnings, ignored(deepseekName, "seed"))
	}
	if opts.Thinking != "" {
		r.ReasoningEffort = ""
		put(r, "thinking", map[string]string{"type": opts.Thinking})
	}
	put(r, "user_id", opts.UserID)
	if c.Output != nil {
		r.ResponseFormat = map[string]string{"type": "json_object"}
		prependSystem(r, "Reply with a single JSON object that matches this JSON Schema:\n"+string(c.Output.Schema))
		warnings = append(warnings, ai.Warning{Message: "deepseek: the API has no JSON Schema mode, so the schema is described in a system message and not enforced"})
	}
	return warnings, nil
}

func deepseekUsage(raw json.RawMessage) ai.Usage {
	u, ok := decodeUsage(raw)
	if !ok {
		return standardUsage(raw)
	}
	return ai.Usage{Input: u.PromptTokens, Output: u.CompletionTokens, CacheRead: u.cacheRead(), Reasoning: u.CompletionDetails.ReasoningTokens, Raw: raw}
}
