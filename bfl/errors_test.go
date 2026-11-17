package bfl_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bfl"
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
		{name: "too many active tasks", status: http.StatusTooManyRequests, fixture: "error_rate_limit.json", wantIs: ai.ErrRateLimit, wantNotIs: ai.ErrQuota, wantRetryable: true, wantMsg: "active tasks"},
		{name: "insufficient credits", status: http.StatusPaymentRequired, fixture: "error_credits.json", wantIs: ai.ErrQuota, wantNotIs: ai.ErrRateLimit, wantMsg: "Insufficient credits"},
		{name: "validation list", status: http.StatusUnprocessableEntity, fixture: "error_validation.json", wantNotIs: ai.ErrAuth, wantMsg: "multiple of 32"},
		{name: "forbidden key", status: http.StatusForbidden, fixture: "error_rate_limit.json", wantIs: ai.ErrAuth},
		{name: "unavailable", status: http.StatusServiceUnavailable, fixture: "error_rate_limit.json", wantRetryable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.status, fixture(t, tt.fixture)))
			_, err := srv.provider().ImageModel("flux-dev").GenerateImage(t.Context(), &ai.ImageCall{Prompt: "p"})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("GenerateImage() error = %v, want *ai.APIError", err)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			if tt.wantNotIs != nil && errors.Is(err, tt.wantNotIs) {
				t.Errorf("errors.Is(err, %v) = true", tt.wantNotIs)
			}
			if apiErr.Retryable() != tt.wantRetryable || !strings.Contains(apiErr.Message, tt.wantMsg) {
				t.Errorf("APIError = %+v, want retryable=%v and message containing %q", apiErr, tt.wantRetryable, tt.wantMsg)
			}
		})
	}
}

func TestNew_regions(t *testing.T) {
	tests := []struct {
		name     string
		region   bfl.Region
		wantHost string
	}{
		{name: "global", region: bfl.RegionGlobal, wantHost: "api.bfl.ai"},
		{name: "eu", region: bfl.RegionEU, wantHost: "api.eu.bfl.ai"},
		{name: "us", region: bfl.RegionUS, wantHost: "api.us.bfl.ai"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt := &hostRecorder{}
			p := bfl.New(bfl.WithAPIKey("k"), bfl.WithRegion(tt.region), bfl.WithHTTPClient(&http.Client{Transport: rt}), bfl.WithHeaders(http.Header{"X-Team": {"a"}}))
			_, err := p.ImageModel("flux-dev").GenerateImage(t.Context(), &ai.ImageCall{Prompt: "p"})
			if err == nil || rt.host != tt.wantHost || rt.key != "k" || rt.team != "a" {
				t.Errorf("host = %q key = %q team = %q (err %v), want %q with the key and header", rt.host, rt.key, rt.team, err, tt.wantHost)
			}
		})
	}
}

type hostRecorder struct{ host, key, team string }

func (h *hostRecorder) RoundTrip(r *http.Request) (*http.Response, error) {
	h.host, h.key, h.team = r.URL.Host, r.Header.Get("X-Key"), r.Header.Get("X-Team")
	return nil, errors.New("bfl: stop")
}

func TestNew_environmentKey(t *testing.T) {
	t.Setenv("BFL_API_KEY", "env-key")
	rt := &hostRecorder{}
	p := bfl.New(bfl.WithHTTPClient(&http.Client{Transport: rt}))
	if _, err := p.ImageModel("flux-dev").GenerateImage(t.Context(), &ai.ImageCall{Prompt: "p"}); err == nil || rt.key != "env-key" {
		t.Errorf("x-key = %q (err %v), want the environment key", rt.key, err)
	}
}
