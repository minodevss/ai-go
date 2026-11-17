package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

const (
	ThinkingAdaptive     = "adaptive"
	ThinkingEnabled      = "enabled"
	ThinkingDisabled     = "disabled"
	ThinkingBetweenTools = "between_tools"

	DisplaySummarized = "summarized"
	DisplayOmitted    = "omitted"

	TTL5Minutes CacheTTL = "5m"
	TTL1Hour    CacheTTL = "1h"
)

type CacheTTL string

// Options are Anthropic request settings passed with ai.Providers. Unset fields keep the API default.
type Options struct {
	Thinking               *Thinking     `json:"thinking,omitzero"`
	Effort                 string        `json:"effort,omitzero"`
	CacheControl           *CacheControl `json:"cacheControl,omitzero"`
	UserID                 string        `json:"userId,omitzero"`
	ServiceTier            string        `json:"serviceTier,omitzero"`
	DisableParallelToolUse bool          `json:"disableParallelToolUse,omitzero"`
	Container              string        `json:"container,omitzero"`
	MCPServers             []MCPServer   `json:"mcpServers,omitzero"`
	Betas                  []string      `json:"betas,omitzero"`
	OutputViaTool          bool          `json:"outputViaTool,omitzero"`
}

func (Options) ProviderName() string { return providerName }

// Thinking overrides the thinking mode derived from ai.TextRequest.Reasoning. BudgetTokens applies to ThinkingEnabled only.
type Thinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budgetTokens,omitzero"`
	Display      string `json:"display,omitzero"`
}

// CacheControl marks an ephemeral cache breakpoint. A zero TTL means the API default of five minutes.
type CacheControl struct {
	TTL CacheTTL `json:"ttl,omitzero"`
}

// MCPServer connects a remote MCP server that Anthropic calls on its side. AllowedTools and DisabledTools are mutually exclusive filters.
type MCPServer struct {
	Name               string   `json:"name"`
	URL                string   `json:"url"`
	AuthorizationToken string   `json:"authorizationToken,omitzero"`
	AllowedTools       []string `json:"allowedTools,omitzero"`
	DisabledTools      []string `json:"disabledTools,omitzero"`
}

// ResponseData is stored in Response.Provider and in the assistant Message.Provider under "anthropic".
type ResponseData struct {
	ID            string          `json:"id,omitzero"`
	Container     *Container      `json:"container,omitzero"`
	ServiceTier   string          `json:"serviceTier,omitzero"`
	StopSequence  string          `json:"stopSequence,omitzero"`
	StopDetails   json.RawMessage `json:"stopDetails,omitzero"`
	InferenceGeo  string          `json:"inferenceGeo,omitzero"`
	ServerToolUse json.RawMessage `json:"serverToolUse,omitzero"`
}

// Container identifies the code execution container of a turn. The next request reuses it until ExpiresAt.
type Container struct {
	ID        string `json:"id"`
	ExpiresAt string `json:"expiresAt,omitzero"`
}

func CachedText(s string) ai.Text {
	return ai.Text{Text: s, Provider: pdata.Encode(providerName, partData{CacheControl: &CacheControl{}})}
}

func CachedTextFor(s string, ttl CacheTTL) ai.Text {
	return ai.Text{Text: s, Provider: pdata.Encode(providerName, partData{CacheControl: &CacheControl{TTL: ttl}})}
}

// FileID references a file uploaded with the Files API instead of sending its bytes.
func FileID(id string) ai.ProviderData {
	return pdata.Encode(providerName, partData{FileID: id})
}

func optionsOf(d ai.ProviderData) (Options, error) {
	raw, ok := d[providerName]
	if !ok {
		return Options{}, nil
	}
	var o Options
	if err := json.Unmarshal(raw, &o); err != nil {
		return Options{}, fmt.Errorf("anthropic: decode provider options: %w", err)
	}
	return o, nil
}

func responseProvider(data ResponseData) ai.ProviderData {
	return pdata.Encode(providerName, data)
}

func containerOf(msgs []ai.Message, now time.Time) string {
	for _, m := range slices.Backward(msgs) {
		if m.Role != ai.RoleAssistant {
			continue
		}
		data, ok := ai.ProviderValue[ResponseData](m.Provider, providerName)
		if !ok || data.Container == nil || data.Container.ID == "" || data.Container.expired(now) {
			continue
		}
		return data.Container.ID
	}
	return ""
}

func (c Container) expired(now time.Time) bool {
	at, err := time.Parse(time.RFC3339, c.ExpiresAt)
	return err == nil && !at.After(now)
}
