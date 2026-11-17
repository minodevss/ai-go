package fal_test

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
	"github.com/minodevss/ai-go/fal"
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
		job := srv.URL + "/fal-ai/flux/requests/req-1"
		switch {
		case r.Method == http.MethodPost:
			fmt.Fprintf(w, `{"request_id":"req-1","response_url":"%[1]s","status_url":"%[1]s/status","cancel_url":"%[1]s/cancel"}`, job)
		case strings.HasSuffix(r.URL.Path, "/status"):
			fmt.Fprint(w, `{"status":"COMPLETED"}`)
		case strings.HasPrefix(r.URL.Path, "/files/"):
			w.Header().Set("Content-Type", "image/png")
			fmt.Fprint(w, "png-bytes")
		default:
			fmt.Fprintf(w, `{"images":[{"url":"%s/files/i.png","content_type":"image/png"}]}`, srv.URL)
		}
	}))
	defer srv.Close()
	p := fal.New(fal.WithAPIKey("k"), fal.WithBaseURL(srv.URL), fal.WithPollInterval(time.Millisecond), fal.WithLimiter(limiter))

	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("fal-ai/flux"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := requests.Load(); got < 4 {
		t.Errorf("server saw %d requests, want at least 4", got)
	}
}

func TestWithHeaders_sentOnEveryRequest(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request, srv *fakeServer) {
		queueScript{statuses: []string{"status_completed.json"}, result: "result_image.json"}.handler(t)(w, r, srv)
	})
	p := fal.New(
		fal.WithAPIKey("test-key"),
		fal.WithBaseURL(srv.URL),
		fal.WithPollInterval(time.Millisecond),
		fal.WithHTTPClient(srv.client()),
		fal.WithHeaders(http.Header{"x-team": {"blue"}}),
	)

	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	for _, r := range srv.all() {
		if strings.HasPrefix(r.Path, "/files/") {
			continue
		}
		if got := r.Header.Get("X-Team"); got != "blue" {
			t.Errorf("%s %s X-Team = %q, want blue", r.Method, r.Path, got)
		}
	}
}
