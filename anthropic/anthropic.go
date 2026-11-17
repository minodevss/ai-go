// Package anthropic adapts the Anthropic Messages API to package ai.
package anthropic

import (
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "anthropic"
	defaultBaseURL = "https://api.anthropic.com"
	apiVersion     = "2026-10-21"
	betaHeader     = "anthropic-beta"
)

type Option func(*options)

type options struct {
	apiKey  string
	baseURL string
	client  *http.Client
	limiter ai.Limiter
	headers http.Header
	betas   []string
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client *transport.Client
	betas  []string
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

// WithLimiter makes every request of the provider wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

// WithHeaders sets headers on every request. Setting anthropic-version here replaces the default API version.
func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithBeta adds beta feature names to the anthropic-beta header of every request.
func WithBeta(names ...string) Option {
	return func(o *options) { o.betas = append(o.betas, names...) }
}

// New reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL as defaults. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{
		apiKey:  os.Getenv("ANTHROPIC_API_KEY"),
		baseURL: os.Getenv("ANTHROPIC_BASE_URL"),
	}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultBaseURL
	}
	headers := http.Header{"User-Agent": {"ai-go"}, "Anthropic-Version": {apiVersion}}
	if o.apiKey != "" {
		headers.Set("X-Api-Key", o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = slices.Clone(v)
	}
	betas := splitBetas(append(o.betas, headers.Values(betaHeader)...))
	headers.Del(betaHeader)
	return &Provider{
		client: transport.New(transport.Config{
			Provider: providerName,
			BaseURL:  o.baseURL,
			Headers:  headers,
			Client:   o.client,
			Limiter:  o.limiter,
			Errors:   parseError,
			Classify: contextTooLong,
		}),
		betas: betas,
	}
}

// Model returns a model from a provider configured from the environment.
func Model(id string) *LanguageModel { return New().Model(id) }

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{client: p.client, id: id, betas: p.betas}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

func splitBetas(values []string) []string {
	var out []string
	for _, v := range values {
		for name := range strings.SplitSeq(v, ",") {
			name = strings.TrimSpace(name)
			if name != "" && !slices.Contains(out, name) {
				out = append(out, name)
			}
		}
	}
	return out
}

var _ ai.Provider = (*Provider)(nil)
