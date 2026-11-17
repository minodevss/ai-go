// Package luma adapts the Luma Dream Machine API to the image and video model kinds of package ai.
package luma

import (
	"context"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "luma"
	defaultBaseURL = "https://api.lumalabs.ai/dream-machine/v1"
)

type Option func(*options)

type options struct {
	apiKey       string
	baseURL      string
	client       *http.Client
	limiter      ai.Limiter
	headers      http.Header
	pollInterval time.Duration
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client  *transport.Client
	dl      mediaio.Downloader
	backoff mediaio.Backoff
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

// WithLimiter makes every request of every model of the provider, including result downloads, wait on l first.
func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithPollInterval sets the first wait between polls of an image generation. It grows by half each poll up to ten times as long, and defaults to one second.
func WithPollInterval(d time.Duration) Option {
	return func(o *options) { o.pollInterval = d }
}

// New reads LUMAAI_API_KEY as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("LUMAAI_API_KEY")}
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
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	client := transport.New(transport.Config{
		Provider: providerName,
		BaseURL:  o.baseURL,
		Headers:  headers,
		Client:   o.client,
		Limiter:  o.limiter,
		Errors:   mediaio.ParseError,
	})
	return &Provider{
		client:  client,
		dl:      mediaio.Downloader{Provider: providerName, Client: client},
		backoff: mediaio.Backoff{Initial: o.pollInterval, Max: 10 * o.pollInterval},
	}
}

// DeleteGeneration removes a generation and its assets, which stay on Luma until deleted.
func (p *Provider) DeleteGeneration(ctx context.Context, id string) error {
	resp, err := p.client.Do(ctx, transport.Request{Method: http.MethodDelete, Path: "/generations/" + url.PathEscape(id)})
	if err != nil {
		return mediaio.FixError(err)
	}
	return resp.Body.Close()
}

// GenerationID references an earlier completed generation, to extend it or interpolate to it, when set as the Provider of an input ai.File of a video call.
func GenerationID(id string) ai.ProviderData {
	return pdata.Encode(providerName, generationRef{ID: id})
}

type generationRef struct {
	ID string `json:"id"`
}
