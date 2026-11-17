// Package openaicompat adapts OpenAI-compatible Chat Completions and Embeddings endpoints, with presets for providers that document their own differences.
package openaicompat

import (
	"context"
	"net/http"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
	"github.com/minodevss/ai-go/internal/embeddings"
	"github.com/minodevss/ai-go/internal/transport"
)

const defaultName = "openaicompat"

type Option func(*options)

// TokenSource returns a current OAuth 2.0 access token. Implementations must be safe for concurrent use.
type TokenSource func(ctx context.Context) (string, error)

type options struct {
	name    string
	apiKey  string
	baseURL string
	token   TokenSource
	client  *http.Client
	limiter ai.Limiter
	headers http.Header
}

type preset struct {
	name         string
	baseURL      string
	keyEnv       string
	apiKeyHeader string
	chat         chatcompletions.Dialect
	embedding    embeddings.Config
}

// Provider is safe for concurrent use. Its models share one HTTP client and one rate limiter.
type Provider struct {
	client    *transport.Client
	chat      *chatcompletions.Dialect
	embedding *embeddings.Config
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithTokenSource authenticates every request with a bearer token from ts instead of the API key.
func WithTokenSource(ts TokenSource) Option {
	return func(o *options) { o.token = ts }
}

// WithBaseURL replaces the endpoint of a preset, for example to select a regional host.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.client = c }
}

// WithLimiter makes every request of every model kind wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithName sets the provider name, which is also the ProviderData key of its options. Presets keep their own fixed name.
func WithName(name string) Option {
	return func(o *options) { o.name = name }
}

// New builds a provider for any OpenAI-compatible endpoint. It sends max_tokens and assumes the standard Chat Completions fields only.
func New(baseURL string, opts ...Option) *Provider {
	return build(preset{
		name:    defaultName,
		baseURL: baseURL,
		chat:    chatcompletions.Dialect{MaxTokens: chatcompletions.MaxTokens, TopK: true},
	}, opts)
}

func build(ps preset, opts []Option) *Provider {
	o := options{name: ps.name, baseURL: ps.baseURL}
	if ps.keyEnv != "" {
		o.apiKey = os.Getenv(ps.keyEnv)
	}
	for _, opt := range opts {
		opt(&o)
	}
	if ps.name != defaultName {
		o.name = ps.name
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	switch {
	case o.token != nil:
	case o.apiKey != "" && ps.apiKeyHeader != "":
		headers.Set(ps.apiKeyHeader, o.apiKey)
	case o.apiKey != "":
		headers.Set("Authorization", "Bearer "+o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	chat, embedding := ps.chat, ps.embedding
	chat.Provider, embedding.Provider = o.name, o.name
	return &Provider{
		client: transport.New(transport.Config{
			Provider: o.name,
			BaseURL:  o.baseURL,
			Headers:  headers,
			Client:   withToken(o.client, o.token),
			Limiter:  o.limiter,
			Errors:   parseError,
		}),
		chat:      &chat,
		embedding: &embedding,
	}
}

type LanguageModel struct {
	model *chatcompletions.Model
}

type EmbeddingModel struct {
	model *embeddings.Model
}

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{model: chatcompletions.New(p.client, id, p.chat)}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{model: embeddings.New(p.client, id, p.embedding)}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

func (m *LanguageModel) Provider() string { return m.model.Provider() }

func (m *LanguageModel) ModelID() string { return m.model.ModelID() }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	return m.model.Generate(ctx, c)
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	return m.model.Stream(ctx, c)
}

func (m *EmbeddingModel) Provider() string { return m.model.Provider() }

func (m *EmbeddingModel) ModelID() string { return m.model.ModelID() }

func (m *EmbeddingModel) MaxBatch() int { return m.model.MaxBatch() }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	return m.model.Embed(ctx, c)
}

var (
	_ ai.Provider       = (*Provider)(nil)
	_ ai.LanguageModel  = (*LanguageModel)(nil)
	_ ai.EmbeddingModel = (*EmbeddingModel)(nil)
)
