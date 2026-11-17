package replicate_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/replicate"
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
		if r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1/") {
			fmt.Fprintf(w, `{"id":"p1","status":"succeeded","output":["%s/out.webp"]}`, srv.URL)
			return
		}
		w.Header().Set("Content-Type", "image/webp")
		fmt.Fprint(w, "webp-bytes")
	}))
	defer srv.Close()
	p := replicate.New(replicate.WithAPIKey("k"), replicate.WithBaseURL(srv.URL+"/v1"), replicate.WithLimiter(limiter))

	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("black-forest-labs/flux-schnell"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := requests.Load(); got < 2 {
		t.Errorf("server saw %d requests, want at least 2", got)
	}
}
