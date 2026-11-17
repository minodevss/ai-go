package google

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestBearerClient_tokenStaysOnOriginalHost(t *testing.T) {
	var other atomic.Value
	elsewhere := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		other.Store(r.Header.Get("Authorization"))
	}))
	t.Cleanup(elsewhere.Close)
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/start" {
			http.Redirect(w, r, elsewhere.URL+"/next", http.StatusFound)
			return
		}
		t.Errorf("unexpected origin path %q", r.URL.Path)
	}))
	t.Cleanup(origin.Close)
	client := bearerClient(nil, func(context.Context) (string, error) { return "tok", nil })
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, origin.URL+"/start", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	_ = resp.Body.Close()
	if got, ok := other.Load().(string); !ok || got != "" {
		t.Errorf("redirect target Authorization = %q (reached %v), want none", got, ok)
	}
}
