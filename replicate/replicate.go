// Package replicate adapts the Replicate predictions API to the image and video model kinds of package ai.
package replicate

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "replicate"
	defaultBaseURL = "https://api.replicate.com/v1"
	cancelTimeout  = 10 * time.Second
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

// WithPollInterval sets the first wait between polls of an image prediction. It grows by half each poll up to ten times as long, and defaults to one second.
func WithPollInterval(d time.Duration) Option {
	return func(o *options) { o.pollInterval = d }
}

// New reads REPLICATE_API_TOKEN as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("REPLICATE_API_TOKEN")}
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

// CancelPrediction cancels a running prediction, for example the one named by an ai.JobError.
func (p *Provider) CancelPrediction(ctx context.Context, id string) error {
	_, err := p.client.JSON(ctx, transport.Request{Path: "/predictions/" + url.PathEscape(id) + "/cancel"}, &prediction{})
	return mediaio.FixError(err)
}

func (p *Provider) cancelAfterEnd(ctx context.Context, id string, err error) error {
	if ctx.Err() == nil {
		return err
	}
	cancelCtx, stop := context.WithTimeout(context.WithoutCancel(ctx), cancelTimeout)
	defer stop()
	if cancelErr := p.CancelPrediction(cancelCtx, id); cancelErr != nil {
		return fmt.Errorf("%w; %w", err, cancelErr)
	}
	return err
}
