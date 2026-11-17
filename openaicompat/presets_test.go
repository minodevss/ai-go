package openaicompat_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openaicompat"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func recordingClient(got *http.Request) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		*got = *r
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(minimalChat))}, nil
	})}
}

func TestPresets_endpoint(t *testing.T) {
	tests := []struct {
		name       string
		build      func(c *http.Client) *openaicompat.Provider
		wantURL    string
		wantHeader string
		wantValue  string
	}{
		{name: "azure", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Azure("res", openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://res.openai.azure.com/openai/v1/chat/completions", wantHeader: "Api-Key", wantValue: "k"},
		{name: "mistral", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Mistral(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.mistral.ai/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "deepseek", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.DeepSeek(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.deepseek.com/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "perplexity", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Perplexity(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.perplexity.ai/v1/sonar", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "openrouter", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.OpenRouter(openaicompat.WithAPIKey("k"), openaicompat.WithAttribution("https://app.test", "App"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://openrouter.ai/api/v1/chat/completions", wantHeader: "Http-Referer", wantValue: "https://app.test"},
		{name: "xai", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.XAI(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.x.ai/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "groq", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Groq(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.groq.com/openai/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "together", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Together(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.together.ai/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "fireworks", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Fireworks(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.fireworks.ai/inference/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "cerebras", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.Cerebras(openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "https://api.cerebras.ai/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
		{name: "vllm", build: func(c *http.Client) *openaicompat.Provider {
			return openaicompat.VLLM("http://gpu:8000/v1", openaicompat.WithAPIKey("k"), openaicompat.WithHTTPClient(c))
		}, wantURL: "http://gpu:8000/v1/chat/completions", wantHeader: "Authorization", wantValue: "Bearer k"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got http.Request
			model := tt.build(recordingClient(&got)).Model("m")
			if _, err := model.Generate(t.Context(), &ai.Call{Messages: hi()}); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got.URL.String() != tt.wantURL {
				t.Errorf("URL = %s, want %s", got.URL, tt.wantURL)
			}
			if v := got.Header.Get(tt.wantHeader); v != tt.wantValue {
				t.Errorf("%s = %q, want %q", tt.wantHeader, v, tt.wantValue)
			}
			if model.Provider() != tt.name {
				t.Errorf("Provider() = %q, want %q", model.Provider(), tt.name)
			}
		})
	}
}

func TestPresets_keepNameOverWithName(t *testing.T) {
	m := openaicompat.Mistral(openaicompat.WithName("other")).Model("m")
	if m.Provider() != "mistral" {
		t.Errorf("Provider() = %q, want mistral", m.Provider())
	}
}

func TestAzure_tokenSourceReplacesKey(t *testing.T) {
	var got http.Request
	p := openaicompat.Azure("res", openaicompat.WithAPIKey("k"), openaicompat.WithTokenSource(func(context.Context) (string, error) { return "t", nil }), openaicompat.WithHTTPClient(recordingClient(&got)))
	if _, err := p.Model("dep").Generate(t.Context(), &ai.Call{Messages: hi()}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got.Header.Get("Authorization") != "Bearer t" || got.Header.Get("Api-Key") != "" {
		t.Errorf("headers = %v, want a bearer token and no api-key", got.Header)
	}
}

func TestPresets_request(t *testing.T) {
	weather := ai.Tool{Name: "f", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}
	tests := []struct {
		name  string
		model func(url string) *openaicompat.LanguageModel
		call  ai.Call
		want  string
	}{
		{
			name: "azure",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Azure("r", openaicompat.WithBaseURL(u)).Model("dep")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Reasoning: ai.ReasoningMax, Provider: ai.Providers(openaicompat.AzureOptions{
				Store: new(false), TextVerbosity: "low", UserSecurityContext: &openaicompat.AzureSecurityContext{ApplicationName: "app", EndUserID: "u1"},
			})},
			want: `{"model":"dep","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"reasoning_effort":"xhigh","store":false,"verbosity":"low","user_security_context":{"application_name":"app","end_user_id":"u1"}}`,
		},
		{
			name: "mistral",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Mistral(openaicompat.WithBaseURL(u)).Model("mistral-large-latest")
			},
			call: ai.Call{
				Messages: hi(), MaxOutputTokens: 9, Seed: new(5), TopK: 3, Tools: []ai.Tool{weather}, ToolChoice: ai.ToolRequired(),
				Provider: ai.Providers(openaicompat.MistralOptions{SafePrompt: new(true), PromptMode: "reasoning", ParallelToolCalls: new(false)}),
			},
			want: `{"model":"mistral-large-latest","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"random_seed":5,"top_k":3,"safe_prompt":true,"prompt_mode":"reasoning","parallel_tool_calls":false,
			"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}],"tool_choice":"any"}`,
		},
		{
			name: "deepseek",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.DeepSeek(openaicompat.WithBaseURL(u)).Model("deepseek-v4-pro")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Seed: new(1), Reasoning: ai.ReasoningHigh, Provider: ai.Providers(openaicompat.DeepSeekOptions{Thinking: openaicompat.DeepSeekThinkingEnabled})},
			want: `{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"thinking":{"type":"enabled"}}`,
		},
		{
			name: "deepseek reasoning effort",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.DeepSeek(openaicompat.WithBaseURL(u)).Model("deepseek-v4-pro")
			},
			call: ai.Call{Messages: hi(), Reasoning: ai.ReasoningMax},
			want: `{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"max"}`,
		},
		{
			name: "perplexity",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Perplexity(openaicompat.WithBaseURL(u)).Model("sonar-pro")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Tools: []ai.Tool{weather}, Provider: ai.Providers(openaicompat.PerplexityOptions{
				SearchMode: "academic", SearchDomainFilter: []string{"arxiv.org", "-pinterest.com"}, SearchRecencyFilter: "month", ReturnRelatedQuestions: new(true),
				WebSearch: &openaicompat.PerplexityWebSearch{SearchContextSize: "high", UserLocation: &openaicompat.PerplexityLocation{Country: "KR"}},
			})},
			want: `{"model":"sonar-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"search_mode":"academic","search_domain_filter":["arxiv.org","-pinterest.com"],"search_recency_filter":"month",
			"return_related_questions":true,"web_search_options":{"search_context_size":"high","user_location":{"country":"KR"}}}`,
		},
		{
			name: "openrouter",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.OpenRouter(openaicompat.WithBaseURL(u)).Model("anthropic/claude-sonnet-4.5")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, TopK: 40, Reasoning: ai.ReasoningHigh, Provider: ai.Providers(openaicompat.OpenRouterOptions{
				Provider: &openaicompat.OpenRouterRouting{
					Order: []string{"anthropic", "openai"}, AllowFallbacks: new(false), RequireParameters: new(true), DataCollection: "deny", Sort: "latency",
					MaxPrice: &openaicompat.OpenRouterPrice{Prompt: new(1.5)},
				},
				Models:  []string{"openai/gpt-5-mini"},
				Plugins: []openaicompat.OpenRouterPlugin{{ID: "web", Settings: map[string]any{"max_results": 3}}},
			})},
			want: `{"model":"anthropic/claude-sonnet-4.5","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"top_k":40,"reasoning_effort":"high",
			"provider":{"order":["anthropic","openai"],"allow_fallbacks":false,"require_parameters":true,"data_collection":"deny","sort":"latency","max_price":{"prompt":1.5}},
			"models":["openai/gpt-5-mini"],"plugins":[{"id":"web","max_results":3}]}`,
		},
		{
			name: "openrouter reasoning object",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.OpenRouter(openaicompat.WithBaseURL(u)).Model("m")
			},
			call: ai.Call{Messages: hi(), Reasoning: ai.ReasoningHigh, Provider: ai.Providers(openaicompat.OpenRouterOptions{Reasoning: &openaicompat.OpenRouterReasoning{MaxTokens: 2000, Exclude: new(true)}})},
			want: `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning":{"max_tokens":2000,"exclude":true}}`,
		},
		{
			name: "xai",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.XAI(openaicompat.WithBaseURL(u)).Model("grok-4.7")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Reasoning: ai.ReasoningMax, Provider: ai.Providers(openaicompat.XAIOptions{PromptCacheKey: "k", ParallelToolCalls: new(true)})},
			want: `{"model":"grok-4.7","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"reasoning_effort":"xhigh","parallel_tool_calls":true,"prompt_cache_key":"k"}`,
		},
		{
			name: "groq",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Groq(openaicompat.WithBaseURL(u)).Model("qwen/qwen3-32b")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, PresencePenalty: new(0.5), Reasoning: ai.ReasoningMedium, Provider: ai.Providers(openaicompat.GroqOptions{
				ReasoningFormat: "parsed", ServiceTier: "flex", SearchSettings: &openaicompat.GroqSearch{IncludeDomains: []string{"x.test"}},
			})},
			want: `{"model":"qwen/qwen3-32b","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"reasoning_effort":"medium","reasoning_format":"parsed","service_tier":"flex","search_settings":{"include_domains":["x.test"]}}`,
		},
		{
			name: "together",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Together(openaicompat.WithBaseURL(u)).Model("openai/gpt-oss-120b")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, TopK: 20, Provider: ai.Providers(openaicompat.TogetherOptions{
				Reasoning: &openaicompat.TogetherReasoning{Enabled: new(true)}, RepetitionPenalty: new(1.1), ChatTemplateKwargs: map[string]any{"clear_thinking": false},
			})},
			want: `{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"top_k":20,"reasoning":{"enabled":true},"repetition_penalty":1.1,"chat_template_kwargs":{"clear_thinking":false}}`,
		},
		{
			name: "fireworks",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Fireworks(openaicompat.WithBaseURL(u)).Model("accounts/fireworks/models/kimi-k2")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Reasoning: ai.ReasoningHigh, Provider: ai.Providers(openaicompat.FireworksOptions{
				Thinking: &openaicompat.FireworksThinking{Type: "enabled", BudgetTokens: 2048}, PerfMetricsInResponse: new(true),
			})},
			want: `{"model":"accounts/fireworks/models/kimi-k2","messages":[{"role":"user","content":"hi"}],"max_tokens":9,"thinking":{"type":"enabled","budget_tokens":2048},"perf_metrics_in_response":true}`,
		},
		{
			name: "cerebras",
			model: func(u string) *openaicompat.LanguageModel {
				return openaicompat.Cerebras(openaicompat.WithBaseURL(u)).Model("gpt-oss-120b")
			},
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, Tools: []ai.Tool{weather}, Reasoning: ai.ReasoningLow, Provider: ai.Providers(openaicompat.CerebrasOptions{ReasoningFormat: "parsed"})},
			want: `{"model":"gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"reasoning_effort":"low","reasoning_format":"parsed",
			"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"},"strict":false}}]}`,
		},
		{
			name:  "vllm",
			model: func(u string) *openaicompat.LanguageModel { return openaicompat.VLLM(u).Model("Qwen/Qwen3-8B") },
			call: ai.Call{Messages: hi(), MaxOutputTokens: 9, TopK: 5, Provider: ai.Providers(openaicompat.VLLMOptions{
				MinP: new(0.05), ChatTemplateKwargs: map[string]any{"enable_thinking": false}, StructuredOutputs: &openaicompat.VLLMStructured{Choice: []string{"a", "b"}},
			})},
			want: `{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9,"top_k":5,"min_p":0.05,"chat_template_kwargs":{"enable_thinking":false},"structured_outputs":{"choice":["a","b"]}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalChat))
			if _, err := tt.model(srv.URL).Generate(t.Context(), &tt.call); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			equalJSON(t, tt.want, srv.last(t).Body)
		})
	}
}
