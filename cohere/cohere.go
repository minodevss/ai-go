// Package cohere adapts the Cohere Chat, Embed and Rerank v2 APIs to package ai.
package cohere

import (
	"net/http"
	"os"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "cohere"
	defaultBaseURL = "https://api.cohere.com"
	clientName     = "ai-go"
)

type Option func(*options)

type options struct {
	apiKey  string
	baseURL string
	client  *http.Client
	limiter ai.Limiter
	headers http.Header
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client *transport.Client
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.client = c }
}

// WithLimiter makes every request of every model kind of the provider wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// New reads CO_API_KEY and CO_API_URL as defaults. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{
		apiKey:  os.Getenv("CO_API_KEY"),
		baseURL: os.Getenv("CO_API_URL"),
	}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultBaseURL
	}
	headers := http.Header{"User-Agent": {"ai-go"}, "X-Client-Name": {clientName}}
	if o.apiKey != "" {
		headers.Set("Authorization", "Bearer "+o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = slices.Clone(v)
	}
	return &Provider{client: transport.New(transport.Config{
		Provider: providerName,
		BaseURL:  o.baseURL,
		Headers:  headers,
		Client:   o.client,
		Limiter:  o.limiter,
	})}
}

// Model returns a chat model from a provider configured from the environment.
func Model(id string) *LanguageModel { return New().Model(id) }

// NewEmbeddingModel returns an embedding model from a provider configured from the environment.
func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

// NewRerankModel returns a rerank model from a provider configured from the environment.
func NewRerankModel(id string) *RerankModel { return New().RerankModel(id) }

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{client: p.client, id: id}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{client: p.client, id: id}
}

func (p *Provider) RerankModel(id string) *RerankModel {
	return &RerankModel{client: p.client, id: id}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

var _ ai.Provider = (*Provider)(nil)
