package luma_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/luma"
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
			fmt.Fprint(w, `{"id":"img-1","state":"queued"}`)
		case path.Base(r.URL.Path) == "img-1":
			fmt.Fprintf(w, `{"id":"img-1","state":"completed","assets":{"image":"%s/a.jpg"}}`, srv.URL)
		case strings.HasSuffix(r.URL.Path, ".jpg"):
			w.Header().Set("Content-Type", "image/jpeg")
			fmt.Fprint(w, "jpeg-bytes")
		}
	}))
	defer srv.Close()
	p := luma.New(luma.WithAPIKey("k"), luma.WithBaseURL(srv.URL), luma.WithPollInterval(time.Millisecond), luma.WithLimiter(limiter))

	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("photon-1"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := requests.Load(); got < 3 {
		t.Errorf("server saw %d requests, want at least 3", got)
	}
}
