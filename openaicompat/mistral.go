package openaicompat

import (
	"crypto/sha256"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
	"github.com/minodevss/ai-go/internal/embeddings"
)

const (
	mistralName      = "mistral"
	mistralCallIDLen = 9
	alphanumeric     = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)

// MistralOptions are Mistral request settings passed with ai.Providers. Unset fields keep the API default.
type MistralOptions struct {
	SafePrompt        *bool    `json:"safePrompt,omitzero"`
	PromptMode        string   `json:"promptMode,omitzero"`
	ParallelToolCalls *bool    `json:"parallelToolCalls,omitzero"`
	MinTokens         int      `json:"minTokens,omitzero"`
	RepetitionPenalty *float64 `json:"repetitionPenalty,omitzero"`
	PromptCacheKey    string   `json:"promptCacheKey,omitzero"`
	ServiceTier       string   `json:"serviceTier,omitzero"`
}

func (MistralOptions) ProviderName() string { return mistralName }

// Mistral builds a Mistral provider and reads MISTRAL_API_KEY. Tool call ids are mapped to the 9 alphanumeric characters the API requires.
// Thinking chunks inside content are decoded to Reasoning parts and replayed as chunks.
func Mistral(opts ...Option) *Provider {
	return build(preset{
		name:    mistralName,
		baseURL: "https://api.mistral.ai/v1",
		keyEnv:  "MISTRAL_API_KEY",
		chat: chatcompletions.Dialect{
			MaxTokens:       chatcompletions.MaxTokens,
			NoStreamOptions: true,
			TopK:            true,
			ToolResultName:  true,
			ReasoningFields: []string{},
			Replay:          chatcompletions.Replay{Chunks: true},
			CallID:          mistralCallID,
			Request:         mistralRequest,
		},
		embedding: embeddings.Config{DimensionsField: "output_dimension"},
	}, opts)
}

func mistralRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[MistralOptions](mistralName, c.Provider)
	if err != nil {
		return nil, err
	}
	put(r, "random_seed", r.Seed)
	r.Seed = nil
	if r.ToolChoice == "required" {
		r.ToolChoice = "any"
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "safe_prompt", opts.SafePrompt)
	put(r, "prompt_mode", opts.PromptMode)
	put(r, "min_tokens", opts.MinTokens)
	put(r, "repetition_penalty", opts.RepetitionPenalty)
	put(r, "prompt_cache_key", opts.PromptCacheKey)
	put(r, "service_tier", opts.ServiceTier)
	return nil, nil
}

func mistralCallID(id string) string {
	if len(id) == mistralCallIDLen && onlyAlphanumeric(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	out := make([]byte, mistralCallIDLen)
	for i := range out {
		out[i] = alphanumeric[int(sum[i])%len(alphanumeric)]
	}
	return string(out)
}

func onlyAlphanumeric(s string) bool {
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			continue
		}
		return false
	}
	return true
}
