// Package fal adapts the fal.ai queue and synchronous inference APIs to package ai.
package fal

import (
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName       = "fal"
	defaultQueueURL    = "https://queue.fal.run"
	defaultSyncURL     = "https://fal.run"
	defaultPollInitial = time.Second
	maxImagePoll       = 10 * time.Second
)

type Option func(*options)

type options struct {
	apiKey       string
	baseURL      string
	syncURL      string
	client       *http.Client
	limiter      ai.Limiter
	headers      http.Header
	pollInterval time.Duration
	unrestricted bool
}

// Provider is safe for concurrent use. Models share one HTTP client and one rate limiter.
type Provider struct {
	queue        *transport.Client
	sync         *transport.Client
	queueURL     string
	pollInterval time.Duration
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithBaseURL sets the queue host. Queue status, result and cancel URLs returned by fal must live on it.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// WithSyncBaseURL sets the host used by ImageOptions.Sync.
func WithSyncBaseURL(url string) Option {
	return func(o *options) { o.syncURL = url }
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.client = c }
}

func WithLimiter(l ai.Limiter) Option {
	return func(o *options) { o.limiter = l }
}

func WithHeaders(h http.Header) Option {
	return func(o *options) { o.headers = h.Clone() }
}

// WithUnrestrictedDownloads lets result downloads use http and private addresses through the configured HTTP client, for tests and local development.
func WithUnrestrictedDownloads() Option {
	return func(o *options) { o.unrestricted = true }
}

// WithPollInterval sets the first wait between status polls of an image job (default 1s). It grows by half each poll up to 10s.
func WithPollInterval(d time.Duration) Option {
	return func(o *options) { o.pollInterval = d }
}

// New reads FAL_KEY as the default API key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("FAL_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultQueueURL
	}
	if o.syncURL == "" {
		o.syncURL = defaultSyncURL
	}
	if o.pollInterval <= 0 {
		o.pollInterval = defaultPollInitial
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("Authorization", "Key "+o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	cfg := transport.Config{Provider: providerName, Headers: headers, Client: o.client, Limiter: o.limiter, Errors: parseError, UnrestrictedDownloads: o.unrestricted}
	cfg.BaseURL = o.baseURL
	queue := transport.New(cfg)
	cfg.BaseURL = o.syncURL
	return &Provider{queue: queue, sync: transport.New(cfg), queueURL: strings.TrimRight(o.baseURL, "/"), pollInterval: o.pollInterval}
}

func (p *Provider) downloader() mediaio.Downloader {
	return mediaio.Downloader{Provider: providerName, Client: p.queue}
}
