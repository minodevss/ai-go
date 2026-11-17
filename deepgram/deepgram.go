// Package deepgram adapts the Deepgram speech-to-text and Aura text-to-speech APIs to package ai.
package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	providerName   = "deepgram"
	defaultBaseURL = "https://api.deepgram.com/v1"
	wsReadLimit    = 16 << 20
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

// WithBaseURL replaces the API root, for example "https://api.eu.deepgram.com/v1" for the EU endpoint.
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

// New reads DEEPGRAM_API_KEY and DEEPGRAM_BASE_URL as defaults. A missing key is reported by the API as an authentication error.
func New(opts ...Option) *Provider {
	o := options{
		apiKey:  os.Getenv("DEEPGRAM_API_KEY"),
		baseURL: os.Getenv("DEEPGRAM_BASE_URL"),
	}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = defaultBaseURL
	}
	headers := http.Header{"User-Agent": {"ai-go"}}
	if o.apiKey != "" {
		headers.Set("Authorization", "Token "+o.apiKey)
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
	var e struct {
		ErrCode  string `json:"err_code"`
		ErrMsg   string `json:"err_msg"`
		Category string `json:"category"`
		Message  string `json:"message"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		return "", strings.TrimSpace(transport.Truncate(string(body)))
	}
	code = e.ErrCode
	if code == "" {
		code = e.Category
	}
	message = e.ErrMsg
	if message == "" {
		message = e.Message
	}
	return code, transport.Truncate(message)
}

func dial(ctx context.Context, client *transport.Client, path string, query url.Values, extra http.Header) (*websocket.Conn, http.Header, error) {
	if err := client.Wait(ctx); err != nil {
		return nil, nil, err
	}
	target, err := url.Parse(client.URL(path))
	if err != nil {
		return nil, nil, fmt.Errorf("deepgram: build websocket url: %w", err)
	}
	switch target.Scheme {
	case "https":
		target.Scheme = "wss"
	case "http":
		target.Scheme = "ws"
	}
	target.RawQuery = query.Encode()
	header := client.Headers()
	for k, v := range extra {
		header[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	ws, resp, err := websocket.Dial(ctx, target.String(), &websocket.DialOptions{HTTPClient: client.HTTPClient(), HTTPHeader: header})
	if err != nil {
		if resp != nil {
			return nil, nil, client.HandshakeError(resp)
		}
		return nil, nil, fmt.Errorf("deepgram: connect: %w", err)
	}
	ws.SetReadLimit(wsReadLimit)
	return ws, resp.Header, nil
}

func (p *Provider) TranscriptionModel(id string) *TranscriptionModel {
	return &TranscriptionModel{client: p.client, id: id}
}

func NewTranscriptionModel(id string) *TranscriptionModel { return New().TranscriptionModel(id) }

func (p *Provider) SpeechModel(id string) *SpeechModel {
	return &SpeechModel{client: p.client, id: id}
}

func NewSpeechModel(id string) *SpeechModel { return New().SpeechModel(id) }
