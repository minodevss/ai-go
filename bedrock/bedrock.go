// Package bedrock adapts the Amazon Bedrock Converse and InvokeModel APIs to package ai.
package bedrock

import (
	"net/http"
	"os"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/sigv4"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName = "bedrock"
	envRegion    = "AWS_REGION"
	envAPIKey    = "AWS_BEARER_TOKEN_BEDROCK"
)

type Option func(*options)

type options struct {
	region      string
	baseURL     string
	credentials Credentials
	apiKey      string
	client      *http.Client
	limiter     ai.Limiter
	headers     http.Header
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client *transport.Client
}

// WithRegion sets the AWS region that picks the default endpoint and the signing scope.
func WithRegion(region string) Option {
	return func(o *options) { o.region = region }
}

// WithCredentials signs requests with SigV4 keys from c and replaces any API key.
func WithCredentials(c Credentials) Option {
	return func(o *options) {
		o.credentials = c
		o.apiKey = ""
	}
}

// WithAPIKey authenticates with an Amazon Bedrock API key, sent as a bearer token, and replaces any credentials.
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.apiKey = key
		o.credentials = nil
	}
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.client = c }
}

// WithLimiter makes every request of the provider wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

// WithHeaders sets headers on every request.
func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithBaseURL replaces the default https://bedrock-runtime.<region>.amazonaws.com, for VPC endpoints, other partitions and proxies. The region still scopes SigV4 signing.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// New reads AWS_REGION and AWS_BEARER_TOKEN_BEDROCK as defaults. Without an API key it signs requests with credentials from WithCredentials, or from AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN. Missing credentials or region are reported by the first request.
func New(opts ...Option) *Provider {
	o := options{region: os.Getenv(envRegion), apiKey: os.Getenv(envAPIKey)}
	for _, opt := range opts {
		opt(&o)
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	credentials := o.credentials
	switch {
	case o.apiKey != "":
		headers.Set("Authorization", "Bearer "+o.apiKey)
		credentials = nil
	case credentials == nil:
		credentials = EnvCredentials()
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = slices.Clone(v)
	}
	baseURL := o.baseURL
	var setupErr error
	if baseURL == "" {
		baseURL = "https://bedrock-runtime." + o.region + ".amazonaws.com"
		if o.region == "" {
			setupErr = ErrNoRegion
		}
	}
	return &Provider{client: transport.New(transport.Config{
		Provider: providerName,
		BaseURL:  baseURL,
		Headers:  headers,
		Client:   newHTTPClient(o.client, credentials, o.region, setupErr),
		Limiter:  o.limiter,
		Errors:   parseError,
		Classify: contextTooLong,
	})}
}

// Model returns a Converse model from a provider configured from the environment.
func Model(id string) *LanguageModel { return New().Model(id) }

// NewEmbeddingModel returns an embedding model from a provider configured from the environment.
func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

// Model accepts a model ID, an inference profile ID or an ARN.
func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{client: p.client, id: id}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{client: p.client, id: id, family: embeddingFamilyOf(id)}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

func modelPath(id, operation string) string {
	return "/model/" + sigv4.URIEncode(id) + "/" + operation
}

var _ ai.Provider = (*Provider)(nil)
