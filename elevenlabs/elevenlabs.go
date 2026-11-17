// Package elevenlabs adapts the ElevenLabs text-to-speech and Scribe speech-to-text APIs to package ai.
package elevenlabs

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "elevenlabs"
	defaultBaseURL = "https://api.elevenlabs.io"
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

// WithBaseURL replaces https://api.elevenlabs.io, for example with a regional host such as https://api.eu.residency.elevenlabs.io.
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

// New reads ELEVENLABS_API_KEY as the default key. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("ELEVENLABS_API_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultBaseURL
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("Xi-Api-Key", o.apiKey)
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
		Errors:   parseError,
	})}
}

func parseError(body []byte) (code, message string) {
	var env struct {
		Detail json.RawMessage `json:"detail"`
	}
	if err := json.Unmarshal(body, &env); err != nil || len(env.Detail) == 0 {
		return transport.ParseError(body)
	}
	var obj struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(env.Detail, &obj); err == nil {
		return obj.Status, obj.Message
	}
	var text string
	if err := json.Unmarshal(env.Detail, &text); err == nil {
		return "", text
	}
	var issues []struct {
		Loc  []any  `json:"loc"`
		Msg  string `json:"msg"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(env.Detail, &issues); err != nil || len(issues) == 0 {
		return "", ""
	}
	msgs := make([]string, 0, len(issues))
	for _, i := range issues {
		msgs = append(msgs, i.Msg)
	}
	return issues[0].Type, strings.Join(msgs, "; ")
}
