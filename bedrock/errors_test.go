package bedrock_test

import (
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
)

func TestLanguageModel_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		errorType     string
		header        map[string]string
		body          string
		wantCode      string
		wantMessage   string
		wantIs        error
		wantNotIs     []error
		wantRetryable bool
		wantRetry     time.Duration
	}{
		{
			name:          "throttling",
			status:        http.StatusTooManyRequests,
			errorType:     "ThrottlingException:http://internal.amazon.com/coral/com.amazon.bedrock/",
			header:        map[string]string{"retry-after": "3"},
			body:          `{"message":"Too many requests, please wait before trying again."}`,
			wantCode:      "ThrottlingException",
			wantMessage:   "Too many requests, please wait before trying again.",
			wantIs:        ai.ErrRateLimit,
			wantNotIs:     []error{ai.ErrQuota, ai.ErrAuth},
			wantRetryable: true,
			wantRetry:     3 * time.Second,
		},
		{
			name:          "model not ready",
			status:        http.StatusTooManyRequests,
			errorType:     "ModelNotReadyException",
			body:          `{"message":"Model is warming up"}`,
			wantCode:      "ModelNotReadyException",
			wantIs:        ai.ErrRateLimit,
			wantRetryable: true,
		},
		{
			name:          "service unavailable",
			status:        http.StatusServiceUnavailable,
			errorType:     "ServiceUnavailableException",
			body:          `{"message":"Service unavailable"}`,
			wantCode:      "ServiceUnavailableException",
			wantRetryable: true,
		},
		{
			name:          "internal server",
			status:        http.StatusInternalServerError,
			errorType:     "InternalServerException",
			body:          `{"message":"boom"}`,
			wantCode:      "InternalServerException",
			wantRetryable: true,
		},
		{
			name:          "model timeout",
			status:        http.StatusRequestTimeout,
			errorType:     "ModelTimeoutException",
			body:          `{"message":"timed out"}`,
			wantCode:      "ModelTimeoutException",
			wantRetryable: true,
		},
		{
			name:      "overloaded 529",
			status:    529,
			errorType: "overloaded_error",
			body:      `{"message":"Overloaded"}`,
			wantCode:  "overloaded_error",

			wantRetryable: true,
		},
		{
			name:          "model error from an upstream 5xx is retried",
			status:        http.StatusFailedDependency,
			errorType:     "ModelErrorException",
			body:          `{"message":"model failed","originalStatusCode":500,"resourceName":"m"}`,
			wantCode:      "ModelErrorException",
			wantMessage:   "model failed",
			wantRetryable: true,
		},
		{
			name:      "model error from an upstream 4xx is not retried",
			status:    http.StatusFailedDependency,
			errorType: "ModelErrorException",
			body:      `{"message":"model rejected","originalStatusCode":400,"resourceName":"m"}`,
			wantCode:  "ModelErrorException",
		},
		{
			name:      "input too long",
			status:    http.StatusBadRequest,
			errorType: "ValidationException",
			body:      `{"message":"Input is too long for requested model."}`,
			wantCode:  "ValidationException",
			wantIs:    ai.ErrContextTooLong,
		},
		{
			name:      "access denied",
			status:    http.StatusForbidden,
			errorType: "AccessDeniedException",
			body:      `{"message":"You don't have access to the model"}`,
			wantCode:  "AccessDeniedException",
			wantIs:    ai.ErrAuth,
			wantNotIs: []error{ai.ErrRateLimit},
		},
		{
			name:      "invalid client token",
			status:    http.StatusForbidden,
			errorType: "InvalidClientTokenId",
			body:      `{"message":"The security token included in the request is invalid."}`,
			wantCode:  "InvalidClientTokenId",
			wantIs:    ai.ErrAuth,
		},
		{
			name:      "validation",
			status:    http.StatusBadRequest,
			errorType: "ValidationException",
			body:      `{"message":"The provided model identifier is invalid."}`,
			wantCode:  "ValidationException",
			wantNotIs: []error{ai.ErrAuth, ai.ErrRateLimit},
		},
		{
			name:      "service quota exceeded is quota",
			status:    http.StatusBadRequest,
			errorType: "ServiceQuotaExceededException",
			body:      `{"message":"quota"}`,
			wantCode:  "ServiceQuotaExceededException",
			wantIs:    ai.ErrQuota,
		},
		{
			name:      "not found",
			status:    http.StatusNotFound,
			errorType: "ResourceNotFoundException",
			body:      `{"message":"Could not find the model"}`,
			wantCode:  "ResourceNotFoundException",
		},
		{
			name:        "capitalized Message and namespace prefix",
			status:      http.StatusBadRequest,
			errorType:   "com.amazon.coral.validate#ValidationException",
			body:        `{"Message":"bad input"}`,
			wantCode:    "ValidationException",
			wantMessage: "bad input",
		},
		{
			name:        "type in the body only",
			status:      http.StatusBadRequest,
			body:        `{"__type":"ValidationException","message":"bad"}`,
			wantCode:    "ValidationException",
			wantMessage: "bad",
		},
		{
			name:      "not json",
			status:    http.StatusBadGateway,
			errorType: "",
			body:      `<html>bad gateway</html>`,
			wantCode:  "",

			wantRetryable: true,
		},
		{
			name:        "not json with an error type",
			status:      http.StatusBadGateway,
			errorType:   "InternalFailure",
			body:        `upstream failed`,
			wantCode:    "InternalFailure",
			wantMessage: "upstream failed",

			wantRetryable: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
				for k, v := range tt.header {
					w.Header().Set(k, v)
				}
				errorReply(tt.status, tt.errorType, tt.body)(w, r)
			})
			_, err := generate(t, f, noRetries(hello()))
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("GenerateText() error = %v, want *ai.APIError", err)
			}
			if apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode || apiErr.Message != tt.wantMessage && tt.wantMessage != "" {
				t.Errorf("APIError = status %d code %q message %q, want %d %q %q", apiErr.StatusCode, apiErr.Code, apiErr.Message, tt.status, tt.wantCode, tt.wantMessage)
			}
			if apiErr.Provider != "bedrock" || apiErr.RequestID != "req-1" {
				t.Errorf("APIError provider %q request id %q, want bedrock req-1", apiErr.Provider, apiErr.RequestID)
			}
			if apiErr.Retryable() != tt.wantRetryable {
				t.Errorf("Retryable() = %v, want %v", apiErr.Retryable(), tt.wantRetryable)
			}
			if apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("RetryAfter = %v, want %v", apiErr.RetryAfter, tt.wantRetry)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			for _, not := range tt.wantNotIs {
				if errors.Is(err, not) {
					t.Errorf("errors.Is(err, %v) = true, want false", not)
				}
			}
		})
	}
}

func TestLanguageModel_errorBodyIsKeptOutOfTheMessage(t *testing.T) {
	f := newFakeServer(t, errorReply(http.StatusBadRequest, "ValidationException", `{"message":"bad","secret":"do-not-leak"}`))
	_, err := generate(t, f, noRetries(hello()))
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok {
		t.Fatalf("GenerateText() error = %v, want *ai.APIError", err)
	}
	if want := "ai: bedrock: status 400 (ValidationException): bad"; apiErr.Error() != want {
		t.Errorf("Error() = %q, want %q", apiErr.Error(), want)
	}
}
