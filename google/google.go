// Package google adapts the Gemini API (AI Studio) and Vertex AI generateContent and embedding endpoints to package ai.
package google

import (
	"cmp"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName       = "google"
	vertexProviderName = "google-vertex"
	defaultBaseURL     = "https://generativelanguage.googleapis.com/v1beta"
	globalLocation     = "global"
)

type Option func(*options)

type options struct {
	apiKey   string
	baseURL  string
	client   *http.Client
	limiter  ai.Limiter
	headers  http.Header
	vertex   bool
	project  string
	location string
	tokens   TokenSource
}

type backend struct {
	client *transport.Client
	vertex bool
	name   string
}

// Provider is safe for concurrent use and builds models that share one HTTP client and one rate limiter.
// Request options always use the key "google"; replay data uses "google" on AI Studio and "google-vertex" on Vertex.
type Provider struct {
	backend *backend
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithBaseURL replaces the endpoint root. In Vertex mode it must include /projects/{project}/locations/{location}.
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

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithVertex switches the provider to Vertex AI. Empty arguments fall back to GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION and "global".
func WithVertex(project, location string) Option {
	return func(o *options) {
		o.vertex = true
		o.project = project
		o.location = location
	}
}

// WithTokenSource sets the bearer token source of a Vertex provider. The default is DefaultTokenSource.
func WithTokenSource(ts TokenSource) Option {
	return func(o *options) { o.tokens = ts }
}

// New reads GEMINI_API_KEY, then GOOGLE_API_KEY, and GOOGLE_GEMINI_BASE_URL as defaults.
func New(opts ...Option) *Provider {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	cfg := transport.Config{Limiter: o.limiter, Errors: parseError}
	if o.vertex {
		cfg.Provider = vertexProviderName
		cfg.BaseURL = o.baseURL
		if cfg.BaseURL == "" {
			cfg.BaseURL = vertexBaseURL(cmp.Or(o.project, os.Getenv("GOOGLE_CLOUD_PROJECT")), cmp.Or(o.location, os.Getenv("GOOGLE_CLOUD_LOCATION"), globalLocation))
		}
		cfg.Client = bearerClient(o.client, o.tokens)
	} else {
		cfg.Provider = providerName
		cfg.BaseURL = cmp.Or(o.baseURL, os.Getenv("GOOGLE_GEMINI_BASE_URL"), defaultBaseURL)
		cfg.Client = o.client
		if key := cmp.Or(o.apiKey, os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY")); key != "" {
			headers.Set("x-goog-api-key", key)
		}
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	cfg.Headers = headers
	return &Provider{backend: &backend{client: transport.New(cfg), vertex: o.vertex, name: cfg.Provider}}
}

// Model returns a language model from a provider configured from the environment.
func Model(id string) *LanguageModel { return New().Model(id) }

func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

func NewImageModel(id string) *ImageModel { return New().ImageModel(id) }

func NewVideoModel(id string) *VideoModel { return New().VideoModel(id) }

func NewSpeechModel(id string) *SpeechModel { return New().SpeechModel(id) }

func NewRealtimeModel(id string) *RealtimeModel { return New().RealtimeModel(id) }

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{backend: p.backend, id: id}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{backend: p.backend, id: id}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

func (b *backend) modelPath(id string) string {
	id = url.PathEscape(strings.TrimPrefix(id, "models/"))
	if b.vertex {
		return "/publishers/google/models/" + id
	}
	return "/models/" + id
}

func vertexBaseURL(project, location string) string {
	host := "aiplatform.googleapis.com"
	if location != globalLocation {
		host = location + "-" + host
	}
	return "https://" + host + "/v1/projects/" + project + "/locations/" + location
}

var _ ai.Provider = (*Provider)(nil)
