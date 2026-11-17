package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
)

const perplexityName = "perplexity"

// PerplexityOptions are Perplexity Sonar search settings passed with ai.Providers. Dates use the API format 3/1/2025.
type PerplexityOptions struct {
	SearchMode             string               `json:"searchMode,omitzero"`
	SearchDomainFilter     []string             `json:"searchDomainFilter,omitzero"`
	SearchRecencyFilter    string               `json:"searchRecencyFilter,omitzero"`
	SearchAfterDate        string               `json:"searchAfterDate,omitzero"`
	SearchBeforeDate       string               `json:"searchBeforeDate,omitzero"`
	LastUpdatedAfter       string               `json:"lastUpdatedAfter,omitzero"`
	LastUpdatedBefore      string               `json:"lastUpdatedBefore,omitzero"`
	SearchLanguageFilter   []string             `json:"searchLanguageFilter,omitzero"`
	ReturnImages           *bool                `json:"returnImages,omitzero"`
	ReturnRelatedQuestions *bool                `json:"returnRelatedQuestions,omitzero"`
	DisableSearch          *bool                `json:"disableSearch,omitzero"`
	EnableSearchClassifier *bool                `json:"enableSearchClassifier,omitzero"`
	WebSearch              *PerplexityWebSearch `json:"webSearch,omitzero"`
}

// PerplexityWebSearch maps to the web_search_options request object.
type PerplexityWebSearch struct {
	SearchContextSize string              `json:"search_context_size,omitzero"`
	SearchType        string              `json:"search_type,omitzero"`
	UserLocation      *PerplexityLocation `json:"user_location,omitzero"`
}

type PerplexityLocation struct {
	Latitude  float64 `json:"latitude,omitzero"`
	Longitude float64 `json:"longitude,omitzero"`
	Country   string  `json:"country,omitzero"`
	City      string  `json:"city,omitzero"`
	Region    string  `json:"region,omitzero"`
}

func (PerplexityOptions) ProviderName() string { return perplexityName }

// Perplexity builds a Sonar provider and reads PERPLEXITY_API_KEY. Sources become url Citations, and searchResults, images and relatedQuestions stay in the response provider data.
// Sonar has no tool calling, so tools are ignored with a warning.
func Perplexity(opts ...Option) *Provider {
	return build(preset{
		name:    perplexityName,
		baseURL: "https://api.perplexity.ai",
		keyEnv:  "PERPLEXITY_API_KEY",
		chat: chatcompletions.Dialect{
			Path:            "/v1/sonar",
			MaxTokens:       chatcompletions.MaxTokens,
			NoStreamOptions: true,
			Request:         perplexityRequest,
			Usage:           perplexityUsage,
			Response:        perplexityResponse,
		},
	}, opts)
}

func perplexityRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[PerplexityOptions](perplexityName, c.Provider)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	if len(r.Tools) > 0 {
		r.Tools, r.ToolChoice, r.ParallelToolCalls = nil, nil, nil
		warnings = append(warnings, ignored(perplexityName, "tools"))
	}
	if r.Seed != nil {
		r.Seed = nil
		warnings = append(warnings, ignored(perplexityName, "seed"))
	}
	put(r, "search_mode", opts.SearchMode)
	put(r, "search_domain_filter", opts.SearchDomainFilter)
	put(r, "search_recency_filter", opts.SearchRecencyFilter)
	put(r, "search_after_date_filter", opts.SearchAfterDate)
	put(r, "search_before_date_filter", opts.SearchBeforeDate)
	put(r, "last_updated_after_filter", opts.LastUpdatedAfter)
	put(r, "last_updated_before_filter", opts.LastUpdatedBefore)
	put(r, "search_language_filter", opts.SearchLanguageFilter)
	put(r, "return_images", opts.ReturnImages)
	put(r, "return_related_questions", opts.ReturnRelatedQuestions)
	put(r, "disable_search", opts.DisableSearch)
	put(r, "enable_search_classifier", opts.EnableSearchClassifier)
	put(r, "web_search_options", opts.WebSearch)
	return warnings, nil
}

func perplexityUsage(raw json.RawMessage) ai.Usage {
	u, ok := decodeUsage(raw)
	if !ok {
		return standardUsage(raw)
	}
	return ai.Usage{Input: u.PromptTokens, Output: u.CompletionTokens, Raw: raw}
}

func perplexityResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	var urls []string
	if err := json.Unmarshal(top["citations"], &urls); err != nil {
		urls = nil
	}
	var results []struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	}
	if err := json.Unmarshal(top["search_results"], &results); err != nil {
		results = nil
	}
	titles := map[string]string{}
	for _, r := range results {
		titles[r.URL] = r.Title
	}
	if len(urls) == 0 {
		for _, r := range results {
			urls = append(urls, r.URL)
		}
	}
	data := map[string]json.RawMessage{}
	keep(top, data, "search_results", "searchResults")
	keep(top, data, "images", "images")
	keep(top, data, "related_questions", "relatedQuestions")
	return chatcompletions.ResponseExtras{Data: data, Citations: urlCitations(urls, titles)}
}
