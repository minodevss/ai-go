package assemblyai_test

import (
	"net/http"
	"testing"
	"time"

	"github.com/minodevss/ai-go/assemblyai"
)

func TestProvider_MintStreamingToken(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, http.StatusOK, fixture(t, "token.json")) })
	before := time.Now()
	tok, err := srv.provider().MintStreamingToken(t.Context(), assemblyai.TokenRequest{ExpiresIn: time.Minute, MaxSession: 10 * time.Minute})
	if err != nil {
		t.Fatalf("MintStreamingToken() error = %v", err)
	}
	if tok.Value != "temp-token-123" || tok.ExpiresAt.Before(before.Add(59*time.Second)) || tok.ExpiresAt.After(time.Now().Add(61*time.Second)) {
		t.Errorf("token = %+v, want the value expiring in about a minute", tok)
	}
	got := srv.last(t)
	if got.Path != "/v1/token" || got.Header.Get("Authorization") != "test-key" {
		t.Errorf("request = %s auth %q, want /v1/token with the raw key", got.Path, got.Header.Get("Authorization"))
	}
	equalJSON(t, `{"product":"streaming","expires_in_seconds":60,"max_session_duration_seconds":600}`, got.Body)
}

func TestProvider_MintStreamingToken_validation(t *testing.T) {
	tests := []struct {
		name string
		req  assemblyai.TokenRequest
	}{
		{name: "no expiry", req: assemblyai.TokenRequest{}},
		{name: "expiry too long", req: assemblyai.TokenRequest{ExpiresIn: 11 * time.Minute}},
		{name: "session too short", req: assemblyai.TokenRequest{ExpiresIn: time.Minute, MaxSession: 30 * time.Second}},
		{name: "session too long", req: assemblyai.TokenRequest{ExpiresIn: time.Minute, MaxSession: 4 * time.Hour}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := assemblyai.New().MintStreamingToken(t.Context(), tt.req); err == nil {
				t.Error("MintStreamingToken() succeeded, want a validation error")
			}
		})
	}
}

func TestProvider_MintStreamingToken_apiError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusBadRequest, `{"error":"invalid parameters","code":"invalid_request"}`)
	})
	_, err := srv.provider().MintStreamingToken(t.Context(), assemblyai.TokenRequest{ExpiresIn: time.Minute})
	if err == nil {
		t.Fatal("MintStreamingToken() succeeded, want an API error")
	}
}
