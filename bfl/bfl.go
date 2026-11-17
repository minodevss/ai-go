// Package bfl adapts the Black Forest Labs FLUX API to the image model kind of package ai.
package bfl

import (
	"context"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName = "bfl"
	maxPollBody  = 1 << 20
)

// Region selects the endpoint family. The global endpoint routes across all clusters; EU and US keep requests in that region.
type Region string

const (
	RegionGlobal Region = ""
	RegionEU     Region = "eu"
	RegionUS     Region = "us"
)

type Option func(*options)

type options struct {
	apiKey       string
	baseURL      string
	region       Region
	client       *http.Client
	limiter      ai.Limiter
	headers      http.Header
	pollInterval time.Duration
	unrestricted bool
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client    *transport.Client
	dl        mediaio.Downloader
	backoff   mediaio.Backoff
	apiHost   string
	apiScheme string
	baseURL   string
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithBaseURL replaces the API host, for example for a proxy. It takes precedence over WithRegion.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

func WithRegion(r Region) Option {
	return func(o *options) { o.region = r }
}

func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.client = c }
}

// WithLimiter makes every request of every model of the provider, including polls and result downloads, wait on l first.
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

// WithPollInterval sets the first wait between polls. It grows by half each poll up to ten times as long, and defaults to one second.
func WithPollInterval(d time.Duration) Option {
	return func(o *options) { o.pollInterval = d }
}

// New reads BFL_API_KEY as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("BFL_API_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = hostFor(o.region)
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("X-Key", o.apiKey)
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

		UnrestrictedDownloads: o.unrestricted,
	})
	apiHost, apiScheme := "", ""
	if u, err := url.Parse(o.baseURL); err == nil {
		apiHost, apiScheme = u.Host, u.Scheme
	}
	return &Provider{
		client:    client,
		dl:        mediaio.Downloader{Provider: providerName, Client: client},
		backoff:   mediaio.Backoff{Initial: o.pollInterval, Max: 10 * o.pollInterval},
		apiHost:   apiHost,
		apiScheme: apiScheme,
		baseURL:   o.baseURL,
	}
}

func hostFor(r Region) string {
	switch r {
	case RegionEU:
		return "https://api.eu.bfl.ai"
	case RegionUS:
		return "https://api.us.bfl.ai"
	default:
		return "https://api.bfl.ai"
	}
}

func (p *Provider) trustedPollingURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || u.User != nil || (u.Scheme != "https" && u.Scheme != p.apiScheme) {
		return fmt.Errorf("bfl: polling url %q is not an https url", raw)
	}
	host := u.Hostname()
	if u.Scheme == p.apiScheme && u.Host == p.apiHost || u.Scheme == "https" && (host == "bfl.ai" || strings.HasSuffix(host, ".bfl.ai")) {
		return nil
	}
	return fmt.Errorf("bfl: refusing to send the API key to polling host %q", host)
}

func (p *Provider) getURL(ctx context.Context, rawURL string, out any) error {
	if err := p.client.Wait(ctx); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, http.NoBody)
	if err != nil {
		return fmt.Errorf("bfl: build poll request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	maps.Copy(req.Header, p.client.Headers())
	resp, err := p.client.HTTPClient().Do(req)
	if err != nil {
		return fmt.Errorf("bfl: send poll request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxPollBody))
	if err != nil {
		return fmt.Errorf("bfl: read poll response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return mediaio.FixError(p.client.APIError(resp, raw))
	}
	return decode(raw, out)
}
