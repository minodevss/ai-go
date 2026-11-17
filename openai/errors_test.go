package openai_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
)

func TestResponsesModel_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		header        map[string]string
		body          string
		wantIs        error
		wantNotIs     []error
		wantRetryable bool
		wantCode      string
		wantRetry     time.Duration
	}{
		{
			name:      "rate limit with retry-after-ms",
			status:    http.StatusTooManyRequests,
			header:    map[string]string{"retry-after-ms": "1500", "x-request-id": "req_1"},
			body:      `{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}`,
			wantIs:    ai.ErrRateLimit,
			wantNotIs: []error{ai.ErrQuota, ai.ErrAuth},

			wantRetryable: true,
			wantCode:      "rate_limit_exceeded",
			wantRetry:     1500 * time.Millisecond,
		},
		{
			name:      "retry-after seconds capped",
			status:    http.StatusTooManyRequests,
			header:    map[string]string{"retry-after": "600"},
			body:      `{"error":{"message":"slow down","code":"rate_limit_exceeded"}}`,
			wantIs:    ai.ErrRateLimit,
			wantCode:  "rate_limit_exceeded",
			wantRetry: time.Minute,

			wantRetryable: true,
		},
		{
			name:      "quota is not retryable",
			status:    http.StatusTooManyRequests,
			body:      `{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}`,
			wantIs:    ai.ErrQuota,
			wantNotIs: []error{ai.ErrRateLimit},
			wantCode:  "insufficient_quota",
		},
		{
			name:     "quota from type when code is null",
			status:   http.StatusTooManyRequests,
			body:     `{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":null}}`,
			wantIs:   ai.ErrQuota,
			wantCode: "insufficient_quota",
		},
		{
			name:     "auth",
			status:   http.StatusUnauthorized,
			body:     `{"error":{"message":"Incorrect API key provided: sk-secret","type":"invalid_request_error","code":"invalid_api_key"}}`,
			wantIs:   ai.ErrAuth,
			wantCode: "invalid_api_key",
		},
		{
			name:     "context too long",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"This model's maximum context length is 128000 tokens","type":"invalid_request_error","code":"context_length_exceeded"}}`,
			wantIs:   ai.ErrContextTooLong,
			wantCode: "context_length_exceeded",
		},
		{
			name:          "server error",
			status:        http.StatusInternalServerError,
			body:          `{"error":{"message":"The server had an error","type":"server_error","code":null}}`,
			wantRetryable: true,
			wantCode:      "server_error",
		},
		{
			name:          "overloaded 529",
			status:        529,
			body:          `{"error":{"message":"overloaded","type":"overloaded_error"}}`,
			wantRetryable: true,
			wantCode:      "overloaded_error",
		},
		{
			name:          "conflict is retryable",
			status:        http.StatusConflict,
			body:          `{"error":{"message":"busy"}}`,
			wantRetryable: true,
		},
		{
			name:          "request timeout is retryable",
			status:        http.StatusRequestTimeout,
			body:          `{"error":{"message":"slow"}}`,
			wantRetryable: true,
		},
		{
			name:   "bad request is not retryable",
			status: http.StatusBadRequest,
			body:   `{"error":{"message":"bad","type":"invalid_request_error","param":"x","code":"invalid_value"}}`,

			wantCode: "invalid_value",
		},
		{
			name:          "html body",
			status:        http.StatusBadGateway,
			body:          `<html>` + strings.Repeat("bad gateway ", 100) + `</html>`,
			wantRetryable: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				for k, v := range tt.header {
					w.Header().Set(k, v)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Generate() error = %v, want *ai.APIError", err)
			}
			if apiErr.Provider != "openai" || apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode {
				t.Errorf("error = %+v, want provider openai status %d code %q", apiErr, tt.status, tt.wantCode)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			for _, not := range tt.wantNotIs {
				if errors.Is(err, not) {
					t.Errorf("errors.Is(err, %v) = true", not)
				}
			}
			if apiErr.Retryable() != tt.wantRetryable {
				t.Errorf("Retryable() = %v, want %v", apiErr.Retryable(), tt.wantRetryable)
			}
			if apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("RetryAfter = %v, want %v", apiErr.RetryAfter, tt.wantRetry)
			}
			if strings.Contains(err.Error(), "<html>") || len(apiErr.Message) > 310 {
				t.Errorf("Error() leaks or does not truncate the body: %q", err.Error())
			}
			if apiErr.Body != tt.body {
				t.Errorf("Body was not kept")
			}
		})
	}
}

func TestResponsesModel_requestID(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-request-id", "req_abc")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad"}}`))
	})
	_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.RequestID != "req_abc" {
		t.Errorf("error = %v, want request id req_abc", err)
	}
}

func TestGenerateText_retriesRateLimit(t *testing.T) {
	calls := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","message":"x"}}`))
			return
		}
		jsonReply(minimalResponse)(w, r)
	})
	res, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Text != "hi" || srv.count() != 2 {
		t.Errorf("text = %q after %d requests, want hi after 2", res.Text, srv.count())
	}
}

func TestGenerateText_doesNotRetryQuota(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":"insufficient_quota","message":"x"}}`))
	})
	_, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if !errors.Is(err, ai.ErrQuota) || srv.count() != 1 {
		t.Errorf("GenerateText() error = %v after %d requests, want quota error after 1", err, srv.count())
	}
}
