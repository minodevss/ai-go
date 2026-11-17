package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const maxCacheBreakpoints = 4

// Cache marks p as an ephemeral cache breakpoint with the API default TTL of five minutes. Parts the API cannot cache come back unchanged: empty text, reasoning, and types other than text, file, tool call, tool result and custom.
func Cache(p ai.Part) ai.Part { return CacheFor(p, "") }

// CacheFor is Cache with a TTL, either TTL5Minutes or TTL1Hour. A request may carry at most four breakpoints, counting Options.CacheControl, and 1h breakpoints must come before 5m ones.
func CacheFor(p ai.Part, ttl CacheTTL) ai.Part {
	cc := &CacheControl{TTL: ttl}
	switch v := p.(type) {
	case ai.Text:
		if v.Text != "" {
			v.Provider = withCache(v.Provider, cc)
		}
		return v
	case ai.File:
		v.Provider = withCache(v.Provider, cc)
		return v
	case ai.ToolCall:
		v.Provider = withCache(v.Provider, cc)
		return v
	case ai.ToolResult:
		v.Provider = withCache(v.Provider, cc)
		return v
	case ai.Custom:
		v.Provider = withCache(v.Provider, cc)
		return v
	default:
		return p
	}
}

// CacheTool marks the tool definition as an ephemeral cache breakpoint. Put it on the last tool to cache every tool before it.
func CacheTool(t ai.Tool) ai.Tool { return CacheToolFor(t, "") }

func CacheToolFor(t ai.Tool, ttl CacheTTL) ai.Tool {
	cc := &CacheControl{TTL: ttl}
	if raw := hostedConfig(t); raw != nil {
		t.Provider = ai.ProviderData{providerName: cachedBlock(raw, cc)}
		return t
	}
	t.Provider = withCache(t.Provider, cc)
	return t
}

func withCache(d ai.ProviderData, cc *CacheControl) ai.ProviderData {
	data := dataOf(d)
	data.CacheControl = cc
	return d.WithRaw(providerName, pdata.JSON(providerName, data))
}

func cachedBlock(raw json.RawMessage, cc *CacheControl) json.RawMessage {
	if cc == nil {
		return raw
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return raw
	}
	var kind string
	if err := json.Unmarshal(fields["type"], &kind); err != nil || uncacheable(kind) {
		return raw
	}
	fields["cache_control"] = pdata.JSON(providerName, wireCacheOf(cc))
	return pdata.JSON(providerName, fields)
}

func uncacheable(blockType string) bool {
	return blockType == "thinking" || blockType == "redacted_thinking"
}

func cacheValidation(req *messagesRequest) ([]ai.Warning, error) {
	var marks []*wireCache
	for _, t := range req.Tools {
		marks = appendMark(marks, cacheOf(t))
	}
	for _, s := range req.System {
		marks = appendMark(marks, s.CacheControl)
	}
	for _, m := range req.Messages {
		for _, c := range m.Content {
			marks = appendMark(marks, cacheOf(c))
		}
	}
	marks = appendMark(marks, req.CacheControl)
	if len(marks) > maxCacheBreakpoints {
		return nil, fmt.Errorf("anthropic: %d cache breakpoints exceed the limit of %d, which the API rejects with a 400", len(marks), maxCacheBreakpoints)
	}
	if hourAfterShort(marks) {
		return []ai.Warning{{Message: "anthropic: a 1h cache breakpoint follows a 5m one, but the API requires longer TTLs to come first, so the API may reject the request"}}, nil
	}
	return nil, nil
}

func appendMark(marks []*wireCache, c *wireCache) []*wireCache {
	if c == nil {
		return marks
	}
	return append(marks, c)
}

func hourAfterShort(marks []*wireCache) bool {
	short := false
	for _, m := range marks {
		switch m.TTL {
		case string(TTL1Hour):
			if short {
				return true
			}
		default:
			short = true
		}
	}
	return false
}

func cacheOf(v any) *wireCache {
	switch b := v.(type) {
	case textBlock:
		return b.CacheControl
	case imageBlock:
		return b.CacheControl
	case documentBlock:
		return b.CacheControl
	case toolUseBlock:
		return b.CacheControl
	case toolResultBlock:
		return b.CacheControl
	case customTool:
		return b.CacheControl
	case json.RawMessage:
		var probe struct {
			CacheControl *wireCache `json:"cache_control"`
		}
		if err := json.Unmarshal(b, &probe); err != nil {
			return nil
		}
		return probe.CacheControl
	default:
		return nil
	}
}
