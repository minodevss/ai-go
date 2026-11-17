package bfl_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bfl"
)

type countingLimiter struct{ waits atomic.Int32 }

func (l *countingLimiter) Wait(context.Context) error {
	l.waits.Add(1)
	return nil
}

func TestWithLimiter_waitsBeforeEveryRequest(t *testing.T) {
	limiter := &countingLimiter{}
	var requests atomic.Int32
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := requests.Add(1)
		if got := limiter.waits.Load(); got < n {
			t.Errorf("request %d %s saw %d limiter waits, want at least %d", n, r.URL.Path, got, n)
		}
		switch {
		case r.Method == http.MethodPost:
			fmt.Fprintf(w, `{"id":"task-1","polling_url":"%s/v1/get_result?id=task-1"}`, srv.URL)
		case strings.HasSuffix(r.URL.Path, "/get_result"):
			fmt.Fprintf(w, `{"id":"task-1","status":"Ready","result":{"sample":"%s/sample.png"}}`, srv.URL)
		default:
			w.Header().Set("Content-Type", "image/png")
			fmt.Fprint(w, "png-bytes")
		}
	}))
	defer srv.Close()
	p := bfl.New(bfl.WithAPIKey("k"), bfl.WithBaseURL(srv.URL), bfl.WithPollInterval(time.Millisecond), bfl.WithLimiter(limiter))

	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("flux-pro-1.1"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := requests.Load(); got < 3 {
		t.Errorf("server saw %d requests, want at least 3", got)
	}
}
