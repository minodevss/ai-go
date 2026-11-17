package replicate_test

import (
	"errors"
	"net/http"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/replicate"
)

func TestProvider_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		fixture       string
		wantIs        error
		wantNotIs     error
		wantRetryable bool
		wantCode      string
	}{
		{name: "throttled", status: http.StatusTooManyRequests, fixture: "error_throttled.json", wantIs: ai.ErrRateLimit, wantNotIs: ai.ErrQuota, wantRetryable: true},
		{name: "insufficient credit", status: http.StatusPaymentRequired, fixture: "error_payment.json", wantIs: ai.ErrQuota, wantNotIs: ai.ErrRateLimit, wantCode: "billing_payment_required"},
		{name: "unauthenticated", status: http.StatusUnauthorized, fixture: "error_invalid.json", wantIs: ai.ErrAuth, wantNotIs: ai.ErrQuota, wantCode: "Invalid version or not permitted"},
		{name: "server error", status: http.StatusBadGateway, fixture: "error_throttled.json", wantIs: nil, wantRetryable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.status, fixture(t, tt.fixture)))
			_, err := srv.provider().ImageModel("o/m").GenerateImage(t.Context(), &ai.ImageCall{Prompt: "p"})
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
			if apiErr.Retryable() != tt.wantRetryable || apiErr.Message == "" || (tt.wantCode != "" && apiErr.Code != tt.wantCode) {
				t.Errorf("APIError = %+v, want retryable=%v code=%q and a message", apiErr, tt.wantRetryable, tt.wantCode)
			}
		})
	}
}

func TestProvider_CancelPrediction(t *testing.T) {
	srv := newFakeServer(t, jsonReply(http.StatusOK, fixture(t, "prediction_canceled.json")))
	if err := srv.provider().CancelPrediction(t.Context(), "abc"); err != nil {
		t.Fatalf("CancelPrediction() error = %v", err)
	}
	got := srv.all()[0]
	if got.Method != http.MethodPost || got.Path != "/v1/predictions/abc/cancel" {
		t.Errorf("request = %s %s, want POST /v1/predictions/abc/cancel", got.Method, got.Path)
	}
}

func TestNew_headers(t *testing.T) {
	srv := newFakeServer(t, jsonReply(http.StatusOK, fixture(t, "prediction_canceled.json")))
	p := replicate.New(replicate.WithAPIKey("k"), replicate.WithBaseURL(srv.URL+"/v1"), replicate.WithHTTPClient(srv.Client()), replicate.WithHeaders(http.Header{"X-Team": {"a"}}))
	if err := p.CancelPrediction(t.Context(), "x"); err != nil {
		t.Fatalf("CancelPrediction() error = %v", err)
	}
	h := srv.all()[0].Header
	if h.Get("Authorization") != "Bearer k" || h.Get("X-Team") != "a" {
		t.Errorf("headers = %v, want bearer auth and the custom header", h)
	}
}

func TestNew_environmentKey(t *testing.T) {
	t.Setenv("REPLICATE_API_TOKEN", "env-token")
	srv := newFakeServer(t, jsonReply(http.StatusOK, fixture(t, "prediction_canceled.json")))
	p := replicate.New(replicate.WithBaseURL(srv.URL+"/v1"), replicate.WithHTTPClient(srv.Client()))
	if err := p.CancelPrediction(t.Context(), "x"); err != nil {
		t.Fatalf("CancelPrediction() error = %v", err)
	}
	if got := srv.all()[0].Header.Get("Authorization"); got != "Bearer env-token" {
		t.Errorf("Authorization = %q, want the environment token", got)
	}
}
