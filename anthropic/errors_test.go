package anthropic_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
)

func TestLanguageModel_apiErrors(t *testing.T) {
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
			name:          "overloaded 529",
			status:        529,
			body:          `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_1"}`,
			wantRetryable: true,
			wantCode:      "overloaded_error",
		},
		{
			name:          "rate limit with retry-after",
			status:        http.StatusTooManyRequests,
			header:        map[string]string{"retry-after": "7"},
			body:          `{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your per-minute rate limit"}}`,
			wantIs:        ai.ErrRateLimit,
			wantNotIs:     []error{ai.ErrQuota, ai.ErrAuth},
			wantRetryable: true,
			wantCode:      "rate_limit_error",
			wantRetry:     7 * time.Second,
		},
		{
			name:      "retry-after capped",
			status:    http.StatusTooManyRequests,
			header:    map[string]string{"retry-after": "600"},
			body:      `{"type":"error","error":{"type":"rate_limit_error","message":"x"}}`,
			wantIs:    ai.ErrRateLimit,
			wantCode:  "rate_limit_error",
			wantRetry: time.Minute,

			wantRetryable: true,
		},
		{
			name:      "spend cap is quota and not retryable",
			status:    http.StatusTooManyRequests,
			body:      `{"type":"error","error":{"type":"rate_limit_error","message":"You have reached your specified API usage limits.","details":{"error_code":"enforced_spend_limit_reached"}}}`,
			wantIs:    ai.ErrQuota,
			wantNotIs: []error{ai.ErrRateLimit},
			wantCode:  "billing_enforced_spend_limit_reached",
		},
		{
			name:     "billing",
			status:   http.StatusPaymentRequired,
			body:     `{"type":"error","error":{"type":"billing_error","message":"Your credit balance is too low"}}`,
			wantIs:   ai.ErrQuota,
			wantCode: "billing_error",
		},
		{
			name:     "authentication",
			status:   http.StatusUnauthorized,
			body:     `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`,
			wantIs:   ai.ErrAuth,
			wantCode: "authentication_error",
		},
		{
			name:     "permission",
			status:   http.StatusForbidden,
			body:     `{"type":"error","error":{"type":"permission_error","message":"no access"}}`,
			wantIs:   ai.ErrAuth,
			wantCode: "permission_error",
		},
		{
			name:     "prompt too long",
			status:   http.StatusBadRequest,
			body:     `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}`,
			wantIs:   ai.ErrContextTooLong,
			wantCode: "invalid_request_error",
		},
		{
			name:      "request too large",
			status:    http.StatusRequestEntityTooLarge,
			body:      `{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}`,
			wantNotIs: []error{ai.ErrContextTooLong},
			wantCode:  "request_too_large",
		},
		{
			name:          "server error",
			status:        http.StatusInternalServerError,
			body:          `{"type":"error","error":{"type":"api_error","message":"An unexpected error has occurred"}}`,
			wantRetryable: true,
			wantCode:      "api_error",
		},
		{
			name:          "timeout",
			status:        http.StatusGatewayTimeout,
			body:          `{"type":"error","error":{"type":"timeout_error","message":"timed out"}}`,
			wantRetryable: true,
			wantCode:      "timeout_error",
		},
		{
			name:     "bad request",
			status:   http.StatusBadRequest,
			body:     `{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: required"}}`,
			wantCode: "invalid_request_error",
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
			if apiErr.Provider != "anthropic" || apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode {
				t.Errorf("error = %+v, want provider anthropic status %d code %q", apiErr, tt.status, tt.wantCode)
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

func TestLanguageModel_requestID(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("request-id", "req_abc")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad"},"request_id":"req_abc"}`))
	})
	_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.RequestID != "req_abc" {
		t.Errorf("error = %v, want request id req_abc", err)
	}
}

func TestGenerateText_retriesOverloaded(t *testing.T) {
	calls := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(529)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
			return
		}
		jsonReply(minimalMessage)(w, r)
	})
	res, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Text != "hi" || srv.count() != 2 {
		t.Errorf("text = %q after %d requests, want hi after 2", res.Text, srv.count())
	}
}

func TestGenerateText_doesNotRetrySpendCap(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"cap","details":{"error_code":"enforced_spend_limit_reached"}}}`))
	})
	_, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if !errors.Is(err, ai.ErrQuota) || srv.count() != 1 {
		t.Errorf("GenerateText() error = %v after %d requests, want quota error after 1", err, srv.count())
	}
}
