// Package transport is the shared HTTP layer of provider adapters: JSON requests, limiter hook and API error mapping.
package transport

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/retry"
)

const (
	maxErrorBody     = 1 << 20
	maxMessageChars  = 300
	maxHandshakeBody = 1 << 16
	maxRedirects     = 10

	DefaultMaxResponseBytes = 64 << 20
)

var ErrCrossHostRedirect = errors.New("transport: refusing redirect to a different host")

type Config struct {
	Provider string
	BaseURL  string
	Headers  http.Header
	Client   *http.Client
	Limiter  ai.Limiter
	Errors   ErrorParser
	Classify ContextClassifier

	MaxResponseBytes      int64
	UnrestrictedDownloads bool
}

type ErrorParser func(body []byte) (code, message string)

type ContextClassifier func(status int, code, message string) bool

type Client struct {
	cfg Config
}

type Request struct {
	Method      string
	Path        string
	Body        any
	Header      http.Header
	Stream      bool
	Raw         io.Reader
	ContentType string
}

func New(cfg Config) *Client {
	cfg.Client = sameHostRedirects(cfg.Client)
	if cfg.MaxResponseBytes <= 0 {
		cfg.MaxResponseBytes = DefaultMaxResponseBytes
	}
	if cfg.Errors == nil {
		cfg.Errors = ParseError
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	cfg.UnrestrictedDownloads = cfg.UnrestrictedDownloads || strings.HasPrefix(cfg.BaseURL, "http://")
	return &Client{cfg: cfg}
}

func (c *Client) Provider() string { return c.cfg.Provider }

func (c *Client) Do(ctx context.Context, r Request) (*http.Response, error) {
	if c.cfg.Limiter != nil {
		if err := c.cfg.Limiter.Wait(ctx); err != nil {
			return nil, fmt.Errorf("%s: wait for rate limiter: %w", c.cfg.Provider, err)
		}
	}
	method := r.Method
	if method == "" {
		method = http.MethodPost
	}
	var body io.Reader
	if r.Raw != nil {
		body = r.Raw
	} else if r.Body != nil {
		raw, err := EncodeJSON(r.Body)
		if err != nil {
			return nil, fmt.Errorf("%s: encode request: %w", c.cfg.Provider, err)
		}
		body = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.cfg.BaseURL+r.Path, body)
	if err != nil {
		return nil, fmt.Errorf("%s: build request: %w", c.cfg.Provider, err)
	}
	switch {
	case r.Raw != nil:
		req.Header.Set("Content-Type", r.ContentType)
	case r.Body != nil:
		req.Header.Set("Content-Type", "application/json")
	}
	if r.Stream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	setHeaders(req.Header, c.cfg.Headers)
	setHeaders(req.Header, r.Header)
	resp, err := c.cfg.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: send request: %w", c.cfg.Provider, err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	if err != nil {
		raw = nil
	}
	return nil, c.APIError(resp, raw)
}

func (c *Client) JSON(ctx context.Context, r Request, out any) (http.Header, error) {
	resp, err := c.Do(ctx, r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, c.cfg.MaxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%s: read response: %w", c.cfg.Provider, err)
	}
	if int64(len(raw)) > c.cfg.MaxResponseBytes {
		return nil, fmt.Errorf("%s: response exceeds the %d byte limit", c.cfg.Provider, c.cfg.MaxResponseBytes)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return nil, fmt.Errorf("%s: decode response: %w", c.cfg.Provider, err)
	}
	return resp.Header, nil
}

func (c *Client) APIError(resp *http.Response, body []byte) *ai.APIError {
	code, message := c.cfg.Errors(body)
	return &ai.APIError{
		Provider:       c.cfg.Provider,
		StatusCode:     resp.StatusCode,
		Code:           code,
		Message:        message,
		RequestID:      requestID(resp.Header),
		RetryAfter:     retry.RetryAfter(resp.Header, time.Now()),
		Body:           string(body),
		ShouldRetry:    shouldRetry(resp.Header),
		ContextTooLong: c.cfg.Classify != nil && c.cfg.Classify(resp.StatusCode, code, message),
	}
}

func shouldRetry(h http.Header) *bool {
	switch strings.ToLower(strings.TrimSpace(h.Get("x-should-retry"))) {
	case "true":
		return new(true)
	case "false":
		return new(false)
	default:
		return nil
	}
}

func EncodeJSON(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(b.Bytes(), "\n"), nil
}

func ParseError(body []byte) (code, message string) {
	var env struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  string          `json:"detail"`
		Code    string          `json:"code"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return "", ""
	}
	var obj struct {
		Code    json.RawMessage `json:"code"`
		Type    string          `json:"type"`
		Message string          `json:"message"`
	}
	if err := json.Unmarshal(env.Error, &obj); err == nil {
		code = scalar(obj.Code)
		if code == "" {
			code = obj.Type
		}
		return code, Truncate(obj.Message)
	}
	var text string
	if err := json.Unmarshal(env.Error, &text); err == nil && text != "" {
		return env.Code, Truncate(text)
	}
	return env.Code, Truncate(cmp.Or(env.Message, env.Detail))
}

func ErrorFromEvent(provider string, code, message string) *ai.APIError {
	return &ai.APIError{Provider: provider, StatusCode: statusForCode(code), Code: code, Message: Truncate(message)}
}

func statusForCode(code string) int {
	switch code {
	case "rate_limit_exceeded", "insufficient_quota":
		return http.StatusTooManyRequests
	case "server_error", "internal_error":
		return http.StatusInternalServerError
	case "invalid_api_key", "authentication_error":
		return http.StatusUnauthorized
	case "context_length_exceeded", "invalid_prompt", "invalid_request_error":
		return http.StatusBadRequest
	default:
		return 0
	}
}

func setHeaders(dst, src http.Header) {
	for k, v := range src {
		dst[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
}

func requestID(h http.Header) string {
	for _, k := range []string{"x-request-id", "request-id", "x-amzn-requestid", "x-amzn-request-id"} {
		if v := h.Get(k); v != "" {
			return v
		}
	}
	return ""
}

func scalar(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var n json.Number
	if json.Unmarshal(raw, &n) == nil {
		return n.String()
	}
	return ""
}

func Truncate(s string) string {
	if len(s) <= maxMessageChars {
		return s
	}
	return strings.ToValidUTF8(s[:maxMessageChars], "") + "..."
}

func (c *Client) HandshakeError(resp *http.Response) error {
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxHandshakeBody))
	if err != nil {
		body = nil
	}
	return c.APIError(resp, body)
}

func (c *Client) Wait(ctx context.Context) error {
	if c.cfg.Limiter == nil {
		return nil
	}
	if err := c.cfg.Limiter.Wait(ctx); err != nil {
		return fmt.Errorf("%s: wait for rate limiter: %w", c.cfg.Provider, err)
	}
	return nil
}

func (c *Client) UnrestrictedDownloads() bool { return c.cfg.UnrestrictedDownloads }

func (c *Client) Headers() http.Header { return c.cfg.Headers.Clone() }

func (c *Client) HTTPClient() *http.Client { return c.cfg.Client }

func (c *Client) URL(path string) string { return c.cfg.BaseURL + path }

func sameHostRedirects(base *http.Client) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	guarded := *base
	inner := base.CheckRedirect
	guarded.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if req.URL.Host != via[0].URL.Host || (via[0].URL.Scheme == "https" && req.URL.Scheme != "https") {
			return ErrCrossHostRedirect
		}
		if inner != nil {
			return inner(req, via)
		}
		if len(via) >= maxRedirects {
			return fmt.Errorf("transport: stopped after %d redirects", maxRedirects)
		}
		return nil
	}
	return &guarded
}
