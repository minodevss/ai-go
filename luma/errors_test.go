package luma_test

import (
	"errors"
	"net/http"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/luma"
)

func TestProvider_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		fixture       string
		wantIs        error
		wantNotIs     error
		wantRetryable bool
		wantMsg       string
	}{
		{name: "invalid key", status: http.StatusUnauthorized, fixture: "error_key.json", wantIs: ai.ErrAuth, wantNotIs: ai.ErrQuota, wantMsg: "Invalid API key"},
		{name: "rate limited", status: http.StatusTooManyRequests, fixture: "error_key.json", wantIs: ai.ErrRateLimit, wantNotIs: ai.ErrQuota, wantRetryable: true},
		{name: "no credits", status: http.StatusPaymentRequired, fixture: "error_key.json", wantIs: ai.ErrQuota, wantNotIs: ai.ErrRateLimit},
		{name: "bad prompt", status: http.StatusBadRequest, fixture: "error_prompt.json", wantNotIs: ai.ErrAuth, wantMsg: "too short"},
		{name: "server error", status: http.StatusInternalServerError, fixture: "error_key.json", wantRetryable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.status, fixture(t, tt.fixture)))
			_, err := srv.provider().VideoModel("ray-2").StartVideo(t.Context(), &ai.VideoCall{Prompt: "p"})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("StartVideo() error = %v, want *ai.APIError", err)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			if tt.wantNotIs != nil && errors.Is(err, tt.wantNotIs) {
				t.Errorf("errors.Is(err, %v) = true", tt.wantNotIs)
			}
			if apiErr.Retryable() != tt.wantRetryable || (tt.wantMsg != "" && apiErr.Message == "") {
				t.Errorf("APIError = %+v, want retryable=%v", apiErr, tt.wantRetryable)
			}
		})
	}
}

func TestProvider_DeleteGeneration(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	if err := srv.provider().DeleteGeneration(t.Context(), "abc"); err != nil {
		t.Fatalf("DeleteGeneration() error = %v", err)
	}
	if got := srv.all()[0]; got.Method != http.MethodDelete || got.Path != "/v1/generations/abc" {
		t.Errorf("request = %s %s, want DELETE /v1/generations/abc", got.Method, got.Path)
	}
	failing := newFakeServer(t, jsonReply(http.StatusNotFound, `{"detail":"Generation not found"}`))
	if err := failing.provider().DeleteGeneration(t.Context(), "abc"); err == nil {
		t.Error("DeleteGeneration() of a missing generation succeeded")
	}
}

func TestNew_headersAndEnvironmentKey(t *testing.T) {
	t.Setenv("LUMAAI_API_KEY", "env-key")
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	p := luma.New(luma.WithBaseURL(srv.URL), luma.WithHTTPClient(srv.Client()), luma.WithHeaders(http.Header{"X-Team": {"a"}}))
	if err := p.DeleteGeneration(t.Context(), "x"); err != nil {
		t.Fatalf("DeleteGeneration() error = %v", err)
	}
	h := srv.all()[0].Header
	if h.Get("Authorization") != "Bearer env-key" || h.Get("X-Team") != "a" {
		t.Errorf("headers = %v, want the environment key and the custom header", h)
	}
}
