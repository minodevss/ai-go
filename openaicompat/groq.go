package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const groqName = "groq"

// GroqOptions are Groq request settings passed with ai.Providers. ReasoningFormat (parsed, raw, hidden) and IncludeReasoning exclude each other, and gpt-oss models accept only IncludeReasoning.
type GroqOptions struct {
	ReasoningFormat       string      `json:"reasoningFormat,omitzero"`
	IncludeReasoning      *bool       `json:"includeReasoning,omitzero"`
	ServiceTier           string      `json:"serviceTier,omitzero"`
	SearchSettings        *GroqSearch `json:"searchSettings,omitzero"`
	CitationOptions       string      `json:"citationOptions,omitzero"`
	DisableToolValidation *bool       `json:"disableToolValidation,omitzero"`
	ParallelToolCalls     *bool       `json:"parallelToolCalls,omitzero"`
}

// GroqSearch limits the domains of Groq's built-in web search.
type GroqSearch struct {
	IncludeDomains []string `json:"include_domains,omitzero"`
	ExcludeDomains []string `json:"exclude_domains,omitzero"`
}

func (GroqOptions) ProviderName() string { return groqName }

// Groq builds a Groq provider and reads GROQ_API_KEY. Reasoning text needs ReasoningFormat parsed.
// Presence and frequency penalties are dropped with a warning.
func Groq(opts ...Option) *Provider {
	return build(preset{
		name:    groqName,
		baseURL: "https://api.groq.com/openai/v1",
		keyEnv:  "GROQ_API_KEY",
		chat: chatcompletions.Dialect{
			ReasoningFields: []string{"reasoning"},
			Request:         groqRequest,
			Response:        groqResponse,
		},
	}, opts)
}

func groqRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[GroqOptions](groqName, c.Provider)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	if r.PresencePenalty != nil {
		r.PresencePenalty = nil
		warnings = append(warnings, ignored(groqName, "presencePenalty"))
	}
	if r.FrequencyPenalty != nil {
		r.FrequencyPenalty = nil
		warnings = append(warnings, ignored(groqName, "frequencyPenalty"))
	}
	if c.Reasoning == ai.ReasoningMax {
		r.ReasoningEffort = "xhigh"
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "reasoning_format", opts.ReasoningFormat)
	put(r, "include_reasoning", opts.IncludeReasoning)
	put(r, "service_tier", opts.ServiceTier)
	put(r, "search_settings", opts.SearchSettings)
	put(r, "citation_options", opts.CitationOptions)
	put(r, "disable_tool_validation", opts.DisableToolValidation)
	return warnings, nil
}

func groqResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	data := map[string]json.RawMessage{}
	keep(top, data, "x_groq", "xGroq")
	keep(top, data, "usage_breakdown", "usageBreakdown")
	return chatcompletions.ResponseExtras{Data: data}
}
