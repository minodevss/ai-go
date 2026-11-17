package openai

import (
	"context"
	"reflect"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

// ChatModel joins an assistant message text on history replay because a chat message cannot hold interleaved parts.
type ChatModel struct {
	model *chatcompletions.Model
}

func chatDialect() *chatcompletions.Dialect {
	return &chatcompletions.Dialect{
		Provider:   providerName,
		ToolStrict: true,
		Replay:     chatcompletions.Replay{Field: "reasoning_content"},
		Request:    chatRequest,
	}
}

func (m *ChatModel) Provider() string { return m.model.Provider() }

func (m *ChatModel) ModelID() string { return m.model.ModelID() }

func (m *ChatModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	return m.model.Generate(ctx, c)
}

func (m *ChatModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	return m.model.Stream(ctx, c)
}

func chatRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := optionsOf(c.Provider)
	if err != nil {
		return nil, err
	}
	r.ReasoningEffort = effort(c.Reasoning)
	r.ParallelToolCalls = opts.ParallelToolCalls
	r.Extra = map[string]any{}
	setExtra(r.Extra, "store", opts.Store)
	setExtra(r.Extra, "service_tier", opts.ServiceTier)
	setExtra(r.Extra, "prompt_cache_key", opts.PromptCacheKey)
	setExtra(r.Extra, "prompt_cache_retention", opts.PromptCacheRetention)
	setExtra(r.Extra, "user", opts.User)
	setExtra(r.Extra, "safety_identifier", opts.SafetyIdentifier)
	setExtra(r.Extra, "metadata", opts.Metadata)
	setExtra(r.Extra, "verbosity", opts.TextVerbosity)
	setExtra(r.Extra, "top_logprobs", opts.TopLogprobs)
	return nil, nil
}

func setExtra(extra map[string]any, key string, v any) {
	if !reflect.ValueOf(v).IsZero() {
		extra[key] = v
	}
}

var _ ai.LanguageModel = (*ChatModel)(nil)
