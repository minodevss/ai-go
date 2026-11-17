package openaicompat

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
	"github.com/minodevss/ai-go/internal/openaiwire"
)

func put(r *chatcompletions.Request, key string, v any) {
	if reflect.ValueOf(v).IsZero() {
		return
	}
	if r.Extra == nil {
		r.Extra = map[string]any{}
	}
	r.Extra[key] = v
}

func ignored(provider, name string) ai.Warning {
	return ai.Warning{Message: fmt.Sprintf("%s: %s is not supported and was ignored", provider, name)}
}

func prependSystem(r *chatcompletions.Request, text string) {
	r.Messages = slices.Insert(r.Messages, 0, any(map[string]string{"role": "system", "content": text}))
}

func urlCitations(urls []string, titles map[string]string) []ai.Citation {
	out := make([]ai.Citation, 0, len(urls))
	for _, u := range urls {
		out = append(out, ai.Citation{Kind: "url", URL: u, Title: titles[u]})
	}
	return out
}

func keep(top map[string]json.RawMessage, into map[string]json.RawMessage, wire, name string) {
	if raw, ok := top[wire]; ok && string(raw) != "null" {
		into[name] = raw
	}
}

type tokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	PromptDetails    struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
	CacheHit int `json:"prompt_cache_hit_tokens"`
	Cached   int `json:"cached_tokens"`
}

func decodeUsage(raw json.RawMessage) (tokenUsage, bool) {
	var u tokenUsage
	if len(raw) == 0 || string(raw) == "null" || json.Unmarshal(raw, &u) != nil {
		return u, false
	}
	return u, true
}

func (u tokenUsage) cacheRead() int {
	return max(u.PromptDetails.CachedTokens, u.CacheHit, u.Cached)
}

func standardUsage(raw json.RawMessage) ai.Usage { return openaiwire.Usage(raw) }
