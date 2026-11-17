package openaicompat

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
	"github.com/minodevss/ai-go/internal/embeddings"
)

const (
	azureName     = "azure"
	azureMaxBatch = 2048
)

// AzureOptions are Azure OpenAI request settings passed with ai.Providers. Unset fields keep the API default.
type AzureOptions struct {
	ParallelToolCalls    *bool                 `json:"parallelToolCalls,omitzero"`
	Store                *bool                 `json:"store,omitzero"`
	TextVerbosity        string                `json:"textVerbosity,omitzero"`
	PromptCacheKey       string                `json:"promptCacheKey,omitzero"`
	PromptCacheRetention string                `json:"promptCacheRetention,omitzero"`
	SafetyIdentifier     string                `json:"safetyIdentifier,omitzero"`
	User                 string                `json:"user,omitzero"`
	UserSecurityContext  *AzureSecurityContext `json:"userSecurityContext,omitzero"`
}

// AzureSecurityContext is passed to Microsoft Defender for Cloud with each request.
type AzureSecurityContext struct {
	ApplicationName string `json:"application_name,omitzero"`
	EndUserID       string `json:"end_user_id,omitzero"`
	EndUserTenantID string `json:"end_user_tenant_id,omitzero"`
	SourceIP        string `json:"source_ip,omitzero"`
}

func (AzureOptions) ProviderName() string { return azureName }

// Azure builds an Azure OpenAI provider for the named resource; the model id is the deployment name.
// It sends AZURE_OPENAI_API_KEY as the api-key header; WithTokenSource sends an Entra ID bearer token instead.
func Azure(resource string, opts ...Option) *Provider {
	return build(preset{
		name:         azureName,
		baseURL:      "https://" + resource + ".openai.azure.com/openai/v1/",
		keyEnv:       "AZURE_OPENAI_API_KEY",
		apiKeyHeader: "api-key",
		chat: chatcompletions.Dialect{
			ToolStrict: true,
			Request:    azureRequest,
			Response:   azureResponse,
		},
		embedding: embeddings.Config{MaxBatch: azureMaxBatch, Request: azureEmbedding},
	}, opts)
}

func azureRequest(c *ai.Call, r *chatcompletions.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[AzureOptions](azureName, c.Provider)
	if err != nil {
		return nil, err
	}
	if c.Reasoning == ai.ReasoningMax {
		r.ReasoningEffort = "xhigh"
	}
	r.ParallelToolCalls = opts.ParallelToolCalls
	put(r, "store", opts.Store)
	put(r, "verbosity", opts.TextVerbosity)
	put(r, "prompt_cache_key", opts.PromptCacheKey)
	put(r, "prompt_cache_retention", opts.PromptCacheRetention)
	put(r, "safety_identifier", opts.SafetyIdentifier)
	put(r, "user", opts.User)
	put(r, "user_security_context", opts.UserSecurityContext)
	return nil, nil
}

func azureEmbedding(c *ai.EmbedCall, r *embeddings.Request) ([]ai.Warning, error) {
	opts, err := chatcompletions.OptionsOf[AzureOptions](azureName, c.Provider)
	if err != nil {
		return nil, err
	}
	if opts.User != "" {
		r.Extra = map[string]any{"user": opts.User}
	}
	return nil, nil
}

func azureResponse(top map[string]json.RawMessage) chatcompletions.ResponseExtras {
	data := map[string]json.RawMessage{}
	keep(top, data, "prompt_filter_results", "promptFilterResults")
	return chatcompletions.ResponseExtras{Data: data}
}
