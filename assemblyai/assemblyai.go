// Package assemblyai adapts the AssemblyAI transcription and Universal-Streaming APIs to package ai.
package assemblyai

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName = "assemblyai"

	globalAPI    = "https://api.assemblyai.com"
	euAPI        = "https://api.eu.assemblyai.com"
	globalStream = "wss://streaming.assemblyai.com"
	usStream     = "wss://streaming.us.assemblyai.com"
	euStream     = "wss://streaming.eu.assemblyai.com"
	globalToken  = "https://token.assemblyai.com"
	usToken      = "https://token.us.assemblyai.com"
	euToken      = "https://token.eu.assemblyai.com"

	minTokenTTL     = time.Second
	maxTokenTTL     = 600 * time.Second
	minTokenSession = 60 * time.Second
	maxTokenSession = 10800 * time.Second
)

// Region selects the AssemblyAI data residency. The zero value is the global endpoint, which routes streaming and token requests to the nearest region.
type Region string

const (
	RegionGlobal Region = ""
	RegionUS     Region = "us"
	RegionEU     Region = "eu"
)

type Option func(*options)

type options struct {
	apiKey    string
	baseURL   string
	streamURL string
	tokenURL  string
	region    Region
	client    *http.Client
	limiter   ai.Limiter
	headers   http.Header
}

// Provider is safe for concurrent use. It builds models that share one HTTP client and one rate limiter.
type Provider struct {
	client      *transport.Client
	tokenClient *transport.Client
	streamURL   string
}

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

// WithBaseURL sets the REST API host used for uploads and transcripts.
func WithBaseURL(url string) Option {
	return func(o *options) { o.baseURL = url }
}

// WithStreamingURL sets the host of the Universal-Streaming WebSocket; http and https schemes are dialed as ws and wss.
func WithStreamingURL(url string) Option {
	return func(o *options) { o.streamURL = url }
}

// WithTokenURL sets the host that mints temporary streaming tokens.
func WithTokenURL(url string) Option {
	return func(o *options) { o.tokenURL = url }
}

// WithRegion picks the default hosts of a data residency region; explicit URL options win.
func WithRegion(r Region) Option {
	return func(o *options) { o.region = r }
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

// New reads ASSEMBLYAI_API_KEY as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("ASSEMBLYAI_API_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	hosts := hostsOf(o.region)
	if o.baseURL == "" {
		o.baseURL = hosts.api
	}
	if o.streamURL == "" {
		o.streamURL = hosts.stream
	}
	if o.tokenURL == "" {
		o.tokenURL = hosts.token
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("Authorization", o.apiKey)
	}
	for k, v := range o.headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	cfg := transport.Config{Provider: providerName, Headers: headers, Client: o.client, Limiter: o.limiter}
	cfg.BaseURL = o.baseURL
	api := transport.New(cfg)
	cfg.BaseURL = o.tokenURL
	return &Provider{client: api, tokenClient: transport.New(cfg), streamURL: o.streamURL}
}

type hosts struct{ api, stream, token string }

func hostsOf(r Region) hosts {
	switch r {
	case RegionEU:
		return hosts{api: euAPI, stream: euStream, token: euToken}
	case RegionUS:
		return hosts{api: globalAPI, stream: usStream, token: usToken}
	default:
		return hosts{api: globalAPI, stream: globalStream, token: globalToken}
	}
}

// TokenRequest.ExpiresIn is how long the token can be redeemed (1s to 10m); MaxSession caps the session length (1m to 3h, zero means the 3h default).
type TokenRequest struct {
	ExpiresIn  time.Duration
	MaxSession time.Duration
}

// StreamingToken is passed by a browser as the token query parameter of the streaming WebSocket. It is not single use: until it expires it can open several sessions.
type StreamingToken struct {
	Value     string
	ExpiresAt time.Time
}

type tokenBody struct {
	Product            string `json:"product"`
	ExpiresInSeconds   int    `json:"expires_in_seconds"`
	MaxSessionDuration int    `json:"max_session_duration_seconds,omitzero"`
}

type tokenResponse struct {
	Token            string `json:"token"`
	ExpiresInSeconds int    `json:"expires_in_seconds"`
}

// MintStreamingToken creates a temporary token with POST /v1/token so a browser can open a streaming session without the API key.
func (p *Provider) MintStreamingToken(ctx context.Context, r TokenRequest) (*StreamingToken, error) {
	if r.ExpiresIn < minTokenTTL || r.ExpiresIn > maxTokenTTL {
		return nil, fmt.Errorf("assemblyai: token expiry %s is outside %s to %s", r.ExpiresIn, minTokenTTL, maxTokenTTL)
	}
	if r.MaxSession != 0 && (r.MaxSession < minTokenSession || r.MaxSession > maxTokenSession) {
		return nil, fmt.Errorf("assemblyai: token session limit %s is outside %s to %s", r.MaxSession, minTokenSession, maxTokenSession)
	}
	body := tokenBody{Product: "streaming", ExpiresInSeconds: int(r.ExpiresIn / time.Second), MaxSessionDuration: int(r.MaxSession / time.Second)}
	var out tokenResponse
	if _, err := p.tokenClient.JSON(ctx, transport.Request{Path: "/v1/token", Body: body}, &out); err != nil {
		return nil, err
	}
	return &StreamingToken{Value: out.Token, ExpiresAt: time.Now().Add(time.Duration(out.ExpiresInSeconds) * time.Second)}, nil
}
