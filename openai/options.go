package openai

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

// Options are OpenAI request settings passed with ai.Providers. Unset fields keep the API default.
type Options struct {
	Store                *bool             `json:"store,omitzero"`
	PreviousResponseID   string            `json:"previousResponseId,omitzero"`
	Conversation         string            `json:"conversation,omitzero"`
	ReasoningSummary     string            `json:"reasoningSummary,omitzero"`
	ServiceTier          string            `json:"serviceTier,omitzero"`
	PromptCacheKey       string            `json:"promptCacheKey,omitzero"`
	PromptCacheRetention string            `json:"promptCacheRetention,omitzero"`
	ParallelToolCalls    *bool             `json:"parallelToolCalls,omitzero"`
	SafetyIdentifier     string            `json:"safetyIdentifier,omitzero"`
	User                 string            `json:"user,omitzero"`
	Metadata             map[string]string `json:"metadata,omitzero"`
	Truncation           string            `json:"truncation,omitzero"`
	MaxToolCalls         int               `json:"maxToolCalls,omitzero"`
	TextVerbosity        string            `json:"textVerbosity,omitzero"`
	TopLogprobs          int               `json:"topLogprobs,omitzero"`
	Include              []string          `json:"include,omitzero"`
}

func (Options) ProviderName() string { return providerName }

// ResponseData is stored in Response.Provider and in the assistant Message.Provider under "openai".
type ResponseData struct {
	ID          string `json:"id,omitzero"`
	ServiceTier string `json:"serviceTier,omitzero"`
}

func optionsOf(d ai.ProviderData) (Options, error) {
	raw, ok := d[providerName]
	if !ok {
		return Options{}, nil
	}
	var o Options
	if err := json.Unmarshal(raw, &o); err != nil {
		return Options{}, fmt.Errorf("openai: decode provider options: %w", err)
	}
	return o, nil
}

func responseProvider(data ResponseData) ai.ProviderData {
	if data == (ResponseData{}) {
		return nil
	}
	return pdata.Encode(providerName, data)
}

func effort(r ai.ReasoningEffort) string { return string(r) }
