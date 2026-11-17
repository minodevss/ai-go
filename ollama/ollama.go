// Package ollama adapts the native Ollama chat and embed APIs to package ai.
package ollama

import (
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "ollama"
	defaultBaseURL = "http://localhost:11434"
	defaultPort    = "11434"
	httpsPort      = "443"

	// CloudBaseURL is the endpoint of Ollama's hosted models: https://docs.ollama.com/cloud
	CloudBaseURL = "https://ollama.com"
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

// WithAPIKey sends the key as a bearer token, which only Ollama's cloud endpoint requires.
func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithBaseURL accepts a full URL or a bare host such as "10.0.0.5:11434", like the OLLAMA_HOST variable.
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

// New reads OLLAMA_HOST and OLLAMA_API_KEY as defaults. Without either, requests go to http://localhost:11434 unauthenticated.
func New(opts ...Option) *Provider {
	o := options{
		apiKey:  os.Getenv("OLLAMA_API_KEY"),
		baseURL: os.Getenv("OLLAMA_HOST"),
	}
	for _, opt := range opts {
		opt(&o)
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("Authorization", "Bearer "+o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = slices.Clone(v)
	}
	return &Provider{client: transport.New(transport.Config{
		Provider: providerName,
		BaseURL:  normalizeHost(o.baseURL),
		Headers:  headers,
		Client:   o.client,
		Limiter:  o.limiter,
	})}
}

// Model returns a chat model from a provider configured from the environment.
func Model(id string) *LanguageModel { return New().Model(id) }

// NewEmbeddingModel returns an embedding model from a provider configured from the environment.
func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{client: p.client, id: id}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{client: p.client, id: id}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

func normalizeHost(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return defaultBaseURL
	}
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" {
		return raw
	}
	if u.Port() == "" {
		port := defaultPort
		if u.Scheme == "https" {
			port = httpsPort
		}
		u.Host = joinHostPort(u.Hostname(), port)
	}
	return strings.TrimRight(u.String(), "/")
}

var _ ai.Provider = (*Provider)(nil)
