// Package jina adapts the Jina embeddings and rerank APIs to package ai.
package jina

import (
	"encoding/json"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName            = "jina"
	defaultBaseURL          = "https://api.jina.ai"
	codeInsufficientBalance = "AUTHZ_INSUFFICIENT_BALANCE"
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

// New reads JINA_API_KEY as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("JINA_API_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultBaseURL
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
		BaseURL:  o.baseURL,
		Headers:  headers,
		Client:   o.client,
		Limiter:  o.limiter,
		Errors:   parseError,
	})}
}

// NewEmbeddingModel returns a model from a provider configured from the environment.
func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

// NewRerankModel returns a model from a provider configured from the environment.
func NewRerankModel(id string) *RerankModel { return New().RerankModel(id) }

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{client: p.client, id: id}
}

func (p *Provider) RerankModel(id string) *RerankModel {
	return &RerankModel{client: p.client, id: id}
}

func parseError(body []byte) (code, message string) {
	var env struct {
		Detail json.RawMessage `json:"detail"`
		Code   string          `json:"code"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return "", ""
	}
	code = env.Code
	if strings.EqualFold(code, codeInsufficientBalance) {
		code = "billing_" + code
	}
	var text string
	if err := json.Unmarshal(env.Detail, &text); err == nil {
		return code, text
	}
	return code, string(env.Detail)
}
