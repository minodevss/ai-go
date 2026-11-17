package fal_test

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/minodevss/ai-go"
)

func TestImageModel_errorMapping(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		fixture   string
		wantCode  string
		wantRetry bool
		wantIs    error
	}{
		{name: "content policy", status: http.StatusUnprocessableEntity, fixture: "error_content_policy.json", wantCode: "content_policy_violation"},
		{name: "auth", status: http.StatusUnauthorized, fixture: "error_auth.json", wantIs: ai.ErrAuth},
		{name: "timeout", status: http.StatusGatewayTimeout, fixture: "error_timeout.json", wantCode: "request_timeout", wantRetry: true},
		{name: "rate limit", status: http.StatusTooManyRequests, fixture: "error_auth.json", wantIs: ai.ErrRateLimit, wantRetry: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", tt.fixture))
			if err != nil {
				t.Fatal(err)
			}
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request, _ *fakeServer) { reply(tt.status, string(raw))(w) })
			_, err = srv.provider().ImageModel("m").GenerateImage(t.Context(), &ai.ImageCall{Prompt: "p"})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok || apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode || apiErr.Retryable() != tt.wantRetry || apiErr.Message == "" {
				t.Fatalf("GenerateImage() error = %#v, want status %d code %q retryable %v", err, tt.status, tt.wantCode, tt.wantRetry)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
		})
	}
}
