// Package openai adapts the OpenAI Responses, Chat Completions and Embeddings APIs to package ai.
package openai

import (
	"net/http"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/chatcompletions"
	"github.com/minodevss/ai-go/internal/embeddings"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "openai"
	defaultBaseURL = "https://api.openai.com/v1"
)

type Option func(*options)

type options struct {
	apiKey       string
	baseURL      string
	client       *http.Client
	limiter      ai.Limiter
	headers      http.Header
	organization string
	project      string
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

// WithLimiter makes every request of every model kind wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

func WithOrganization(id string) Option {
	return func(o *options) { o.organization = id }
}

func WithProject(id string) Option {
	return func(o *options) { o.project = id }
}

// New reads OPENAI_API_KEY and OPENAI_BASE_URL as defaults.
func New(opts ...Option) *Provider {
	o := options{
		apiKey:  os.Getenv("OPENAI_API_KEY"),
		baseURL: os.Getenv("OPENAI_BASE_URL"),
	}
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
	if o.organization != "" {
		headers.Set("OpenAI-Organization", o.organization)
	}
	if o.project != "" {
		headers.Set("OpenAI-Project", o.project)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	return &Provider{client: transport.New(transport.Config{
		Provider: providerName,
		BaseURL:  o.baseURL,
		Headers:  headers,
		Client:   o.client,
		Limiter:  o.limiter,
	})}
}

func Model(id string) *LanguageModel { return New().Model(id) }

func NewChatModel(id string) *ChatModel { return New().ChatModel(id) }

func NewEmbeddingModel(id string) *EmbeddingModel { return New().EmbeddingModel(id) }

func NewImageModel(id string) *ImageModel { return New().ImageModel(id) }

func NewVideoModel(id string) *VideoModel { return New().VideoModel(id) }

func NewSpeechModel(id string) *SpeechModel { return New().SpeechModel(id) }

func NewTranscriptionModel(id string) *TranscriptionModel { return New().TranscriptionModel(id) }

func NewRealtimeModel(id string) *RealtimeModel { return New().RealtimeModel(id) }

func (p *Provider) Model(id string) *LanguageModel {
	return &LanguageModel{client: p.client, id: id}
}

func (p *Provider) ChatModel(id string) *ChatModel {
	return &ChatModel{model: chatcompletions.New(p.client, id, chatDialect())}
}

func (p *Provider) EmbeddingModel(id string) *EmbeddingModel {
	return &EmbeddingModel{model: embeddings.New(p.client, id, embeddingConfig())}
}

func (p *Provider) LanguageModel(id string) (ai.LanguageModel, error) { return p.Model(id), nil }

var _ ai.Provider = (*Provider)(nil)
