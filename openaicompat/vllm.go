package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const vllmName = "vllm"

// VLLMOptions are vLLM request settings passed with ai.Providers. Unset fields keep the server default.
type VLLMOptions struct {
	MinP                *float64        `json:"minP,omitzero"`
	RepetitionPenalty   *float64        `json:"repetitionPenalty,omitzero"`
	MinTokens           int             `json:"minTokens,omitzero"`
	StopTokenIDs        []int           `json:"stopTokenIds,omitzero"`
	IncludeReasoning    *bool           `json:"includeReasoning,omitzero"`
	ThinkingTokenBudget int             `json:"thinkingTokenBudget,omitzero"`
	ChatTemplateKwargs  map[string]any  `json:"chatTemplateKwargs,omitzero"`
	StructuredOutputs   *VLLMStructured `json:"structuredOutputs,omitzero"`
	SkipSpecialTokens   *bool           `json:"skipSpecialTokens,omitzero"`
	CacheSalt           string          `json:"cacheSalt,omitzero"`
	Priority            int             `json:"priority,omitzero"`
}

// VLLMStructured maps to the structured_outputs request object; set one field.
type VLLMStructured struct {
	JSON    json.RawMessage `json:"json,omitzero"`
	Regex   string          `json:"regex,omitzero"`
	Choice  []string        `json:"choice,omitzero"`
	Grammar string          `json:"grammar,omitzero"`
}

func (VLLMOptions) ProviderName() string { return vllmName }

// VLLM builds a provider for a vLLM server such as http://localhost:8000/v1 and reads the optional VLLM_API_KEY.
// Reasoning needs a server started with a reasoning parser.
func VLLM(baseURL string, opts ...Option) *Provider {
	return build(preset{
		name:    vllmName,
		baseURL: baseURL,
		keyEnv:  "VLLM_API_KEY",
		chat: chatcompletions.Dialect{
			TopK:            true,
			ReasoningFields: []string{"reasoning", "reasoning_content"},
			Request:         vllmRequest,
		},
	}, opts)
}

func vllmRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[VLLMOptions](vllmName, c.Provider)
	if err != nil {
		return nil, err
	}
	put(r, "min_p", opts.MinP)
	put(r, "repetition_penalty", opts.RepetitionPenalty)
	put(r, "min_tokens", opts.MinTokens)
	put(r, "stop_token_ids", opts.StopTokenIDs)
	put(r, "include_reasoning", opts.IncludeReasoning)
	put(r, "thinking_token_budget", opts.ThinkingTokenBudget)
	put(r, "chat_template_kwargs", opts.ChatTemplateKwargs)
	put(r, "structured_outputs", opts.StructuredOutputs)
	put(r, "skip_special_tokens", opts.SkipSpecialTokens)
	put(r, "cache_salt", opts.CacheSalt)
	put(r, "priority", opts.Priority)
	return nil, nil
}
