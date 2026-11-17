package transport_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

type limiterFunc func(context.Context) error

func (f limiterFunc) Wait(ctx context.Context) error { return f(ctx) }

func TestClient_Do(t *testing.T) {
	var gotHeader http.Header
	var gotBody, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Clone()
		gotPath = r.URL.Path
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		gotBody = string(b)
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()
	waited := 0
	c := transport.New(transport.Config{
		Provider: "p",
		BaseURL:  srv.URL + "/v1/",
		Headers:  http.Header{"Authorization": {"Bearer a"}, "X-Both": {"provider"}},
		Limiter:  limiterFunc(func(context.Context) error { waited++; return nil }),
	})
	resp, err := c.Do(t.Context(), transport.Request{Path: "/x", Body: map[string]string{"k": "<&>"}, Header: http.Header{"x-both": {"call"}}, Stream: true})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	if gotPath != "/v1/x" || gotBody != `{"k":"<&>"}` || waited != 1 {
		t.Errorf("path %q body %q waited %d, want /v1/x, unescaped body, 1", gotPath, gotBody, waited)
	}
	want := map[string]string{"Authorization": "Bearer a", "X-Both": "call", "Accept": "text/event-stream", "Content-Type": "application/json"}
	for k, v := range want {
		if gotHeader.Get(k) != v {
			t.Errorf("header %s = %q, want %q", k, gotHeader.Get(k), v)
		}
	}
}

func TestClient_Do_limiterError(t *testing.T) {
	errStop := errors.New("stop")
	c := transport.New(transport.Config{Provider: "p", BaseURL: "http://invalid.test", Limiter: limiterFunc(func(context.Context) error { return errStop })})
	if _, err := c.Do(t.Context(), transport.Request{Path: "/x"}); !errors.Is(err, errStop) {
		t.Errorf("Do() error = %v, want %v", err, errStop)
	}
}

func TestClient_Do_networkError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	c := transport.New(transport.Config{Provider: "p", BaseURL: url})
	_, err := c.Do(t.Context(), transport.Request{Path: "/x"})
	if err == nil || !strings.HasPrefix(err.Error(), "p: send request") {
		t.Errorf("Do() error = %v, want p: send request", err)
	}
}

func TestClient_JSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Seen", "1")
		_, _ = io.WriteString(w, `{"a":3}`)
	}))
	defer srv.Close()
	c := transport.New(transport.Config{Provider: "p", BaseURL: srv.URL})
	var out struct{ A int }
	header, err := c.JSON(t.Context(), transport.Request{Path: "/"}, &out)
	if err != nil || out.A != 3 || header.Get("X-Seen") != "1" {
		t.Errorf("JSON() = %+v, %v, %v; want A=3 with header", out, header, err)
	}
}

func TestClient_JSON_invalid(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "nope") }))
	defer srv.Close()
	c := transport.New(transport.Config{Provider: "p", BaseURL: srv.URL})
	var out struct{}
	if _, err := c.JSON(t.Context(), transport.Request{Path: "/"}, &out); err == nil || !strings.HasPrefix(err.Error(), "p: decode response") {
		t.Errorf("JSON() error = %v, want p: decode response", err)
	}
}

func TestParseError(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantCode    string
		wantMessage string
	}{
		{name: "openai", body: `{"error":{"message":"m","type":"t","code":"c"}}`, wantCode: "c", wantMessage: "m"},
		{name: "type fallback", body: `{"error":{"message":"m","type":"t","code":null}}`, wantCode: "t", wantMessage: "m"},
		{name: "numeric code", body: `{"error":{"message":"m","code":429}}`, wantCode: "429", wantMessage: "m"},
		{name: "string error", body: `{"error":"bad key","code":"x"}`, wantCode: "x", wantMessage: "bad key"},
		{name: "top-level message", body: `{"message":"m","code":"c"}`, wantCode: "c", wantMessage: "m"},
		{name: "detail", body: `{"detail":"d"}`, wantMessage: "d"},
		{name: "not json", body: `<html>`},
		{name: "empty", body: ``},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, message := transport.ParseError([]byte(tt.body))
			if code != tt.wantCode || message != tt.wantMessage {
				t.Errorf("ParseError(%q) = %q, %q, want %q, %q", tt.body, code, message, tt.wantCode, tt.wantMessage)
			}
		})
	}
}

func TestParseError_truncates(t *testing.T) {
	_, message := transport.ParseError([]byte(`{"error":{"message":"` + strings.Repeat("é", 400) + `"}}`))
	if len(message) > 310 || !strings.HasSuffix(message, "...") {
		t.Errorf("message length %d, want truncated with ellipsis", len(message))
	}
}

func TestErrorFromEvent(t *testing.T) {
	tests := []struct {
		code   string
		status int
		is     error
	}{
		{code: "rate_limit_exceeded", status: http.StatusTooManyRequests, is: ai.ErrRateLimit},
		{code: "insufficient_quota", status: http.StatusTooManyRequests, is: ai.ErrQuota},
		{code: "invalid_api_key", status: http.StatusUnauthorized, is: ai.ErrAuth},
		{code: "context_length_exceeded", status: http.StatusBadRequest, is: ai.ErrContextTooLong},
		{code: "server_error", status: http.StatusInternalServerError},
		{code: "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			err := transport.ErrorFromEvent("p", tt.code, "m")
			if err.StatusCode != tt.status {
				t.Errorf("StatusCode = %d, want %d", err.StatusCode, tt.status)
			}
			if tt.is != nil && !errors.Is(err, tt.is) {
				t.Errorf("errors.Is(err, %v) = false", tt.is)
			}
		})
	}
}

func TestClient_APIError_retryAfterDate(t *testing.T) {
	resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": {time.Now().Add(10 * time.Second).UTC().Format(http.TimeFormat)}}}
	c := transport.New(transport.Config{Provider: "p"})
	err := c.APIError(resp, []byte(`{}`))
	if err.RetryAfter <= 0 || err.RetryAfter > 11*time.Second {
		t.Errorf("RetryAfter = %v, want about 10s", err.RetryAfter)
	}
}

func TestEncodeJSON(t *testing.T) {
	got, err := transport.EncodeJSON(map[string]string{"a": "<b>&"})
	if err != nil {
		t.Fatalf("EncodeJSON() error = %v", err)
	}
	if diff := cmp.Diff(`{"a":"<b>&"}`, string(got)); diff != "" {
		t.Errorf("EncodeJSON() mismatch (-want +got):\n%s", diff)
	}
}

func TestClient_APIError_shouldRetryHeader(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		header     string
		wantSet    bool
		wantRetry  bool
		wantAPIRty bool
	}{
		{name: "true overrides a 400", status: http.StatusBadRequest, header: "true", wantSet: true, wantRetry: true, wantAPIRty: true},
		{name: "false overrides a 500", status: http.StatusInternalServerError, header: "false", wantSet: true},
		{name: "absent keeps the status default", status: http.StatusInternalServerError, wantAPIRty: true},
		{name: "unknown value is ignored", status: http.StatusBadRequest, header: "maybe"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{StatusCode: tt.status, Header: http.Header{}}
			if tt.header != "" {
				resp.Header.Set("X-Should-Retry", tt.header)
			}
			err := transport.New(transport.Config{Provider: "p"}).APIError(resp, []byte(`{}`))
			if (err.ShouldRetry != nil) != tt.wantSet || tt.wantSet && *err.ShouldRetry != tt.wantRetry {
				t.Errorf("ShouldRetry = %v, want set=%v value=%v", err.ShouldRetry, tt.wantSet, tt.wantRetry)
			}
			if got := err.Retryable(); got != tt.wantAPIRty {
				t.Errorf("Retryable() = %v, want %v", got, tt.wantAPIRty)
			}
		})
	}
}

func TestClient_APIError_contextClassifier(t *testing.T) {
	c := transport.New(transport.Config{Provider: "p", Classify: func(_ int, _, message string) bool { return message == "too big" }})
	resp := &http.Response{StatusCode: http.StatusBadRequest, Header: http.Header{}}
	if err := c.APIError(resp, []byte(`{"message":"too big"}`)); !errors.Is(err, ai.ErrContextTooLong) {
		t.Errorf("errors.Is(err, ErrContextTooLong) = false for a classified message")
	}
	if err := c.APIError(resp, []byte(`{"message":"other"}`)); errors.Is(err, ai.ErrContextTooLong) {
		t.Errorf("errors.Is(err, ErrContextTooLong) = true for an unclassified message")
	}
}

func redirectTarget(t *testing.T) (*httptest.Server, <-chan http.Header) {
	t.Helper()
	got := make(chan http.Header, 1)
	evil := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got <- r.Header.Clone()
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(evil.Close)
	return evil, got
}

func TestClient_JSON_refusesCrossHostRedirect(t *testing.T) {
	headers := http.Header{"X-Api-Key": {"SECRET"}, "Authorization": {"Bearer SECRET"}}
	tests := []struct {
		name   string
		client func(*http.Client) *http.Client
	}{
		{name: "default client", client: func(*http.Client) *http.Client { return nil }},
		{name: "caller client", client: func(c *http.Client) *http.Client { return c }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			evil, got := redirectTarget(t)
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, strings.Replace(evil.URL, "127.0.0.1", "localhost", 1)+"/steal", http.StatusTemporaryRedirect)
			}))
			defer api.Close()
			caller := &http.Client{}
			c := transport.New(transport.Config{Provider: "x", BaseURL: api.URL, Headers: headers, Client: tt.client(caller)})
			var out map[string]any
			_, err := c.JSON(t.Context(), transport.Request{Path: "/v1", Body: map[string]any{"a": 1}}, &out)
			if !errors.Is(err, transport.ErrCrossHostRedirect) {
				t.Errorf("JSON() error = %v, want ErrCrossHostRedirect", err)
			}
			select {
			case h := <-got:
				t.Errorf("redirect target received headers %v", h)
			default:
			}
			if caller.CheckRedirect != nil {
				t.Error("caller's client was mutated")
			}
		})
	}
}

func TestClient_JSON_followsSameHostRedirect(t *testing.T) {
	var mux http.ServeMux
	mux.HandleFunc("/a", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/b", http.StatusTemporaryRedirect) })
	mux.HandleFunc("/b", func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, `{"ok":true}`) })
	srv := httptest.NewServer(&mux)
	defer srv.Close()
	c := transport.New(transport.Config{Provider: "x", BaseURL: srv.URL})
	var out map[string]bool
	if _, err := c.JSON(t.Context(), transport.Request{Path: "/a", Method: http.MethodGet}, &out); err != nil || !out["ok"] {
		t.Errorf("JSON() = %v, %v, want ok", out, err)
	}
}

func TestClient_JSON_responseLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"a":"`+strings.Repeat("x", 100)+`"}`)
	}))
	defer srv.Close()
	var out map[string]string
	small := transport.New(transport.Config{Provider: "x", BaseURL: srv.URL, MaxResponseBytes: 50})
	if _, err := small.JSON(t.Context(), transport.Request{Path: "/"}, &out); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Errorf("JSON() error = %v, want limit error", err)
	}
	roomy := transport.New(transport.Config{Provider: "x", BaseURL: srv.URL, MaxResponseBytes: 1000})
	if _, err := roomy.JSON(t.Context(), transport.Request{Path: "/"}, &out); err != nil {
		t.Errorf("JSON() error = %v", err)
	}
}

func TestClient_HandshakeError(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		header     http.Header
		body       string
		wantStatus int
		wantBody   string
	}{
		{name: "json error body", status: http.StatusUnauthorized, header: http.Header{"X-Request-Id": {"req_1"}}, body: `{"error":{"message":"bad key"}}`, wantStatus: http.StatusUnauthorized, wantBody: `{"error":{"message":"bad key"}}`},
		{name: "oversized body is cut", status: http.StatusBadGateway, body: strings.Repeat("x", 1<<20), wantStatus: http.StatusBadGateway},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := transport.New(transport.Config{Provider: "p"})
			closed := false
			resp := &http.Response{StatusCode: tt.status, Header: tt.header, Body: &closeSpy{Reader: strings.NewReader(tt.body), closed: &closed}}
			apiErr, ok := errors.AsType[*ai.APIError](c.HandshakeError(resp))
			if !ok {
				t.Fatal("HandshakeError() is not an *ai.APIError")
			}
			if apiErr.StatusCode != tt.wantStatus || apiErr.Provider != "p" {
				t.Errorf("HandshakeError() = status %d provider %q, want %d p", apiErr.StatusCode, apiErr.Provider, tt.wantStatus)
			}
			if tt.wantBody != "" && apiErr.Body != tt.wantBody {
				t.Errorf("HandshakeError() body = %q, want %q", apiErr.Body, tt.wantBody)
			}
			if len(apiErr.Body) >= len(tt.body) && len(tt.body) > 1<<16 {
				t.Errorf("HandshakeError() body has %d bytes, want it cut", len(apiErr.Body))
			}
			if !closed {
				t.Error("HandshakeError() left the response body open")
			}
		})
	}
}

type closeSpy struct {
	io.Reader
	closed *bool
}

func (c *closeSpy) Close() error {
	*c.closed = true
	return nil
}
