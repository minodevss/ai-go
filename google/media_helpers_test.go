package google_test

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

const bubbleURL = "http://bubble.test"

type bubbleServer struct {
	mu       sync.Mutex
	requests []captured
	handle   func(http.ResponseWriter, *http.Request)
}

func newBubbleServer(handle func(http.ResponseWriter, *http.Request)) *bubbleServer {
	return &bubbleServer{handle: handle}
}

func (b *bubbleServer) client() *http.Client { return &http.Client{Transport: b} }

func (b *bubbleServer) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		var err error
		if body, err = io.ReadAll(req.Body); err != nil {
			return nil, err
		}
		req.Body = io.NopCloser(bytes.NewReader(body))
	}
	b.mu.Lock()
	b.requests = append(b.requests, captured{Path: req.URL.Path, Query: req.URL.RawQuery, Header: req.Header.Clone(), Body: body})
	b.mu.Unlock()
	rec := httptest.NewRecorder()
	b.handle(rec, req)
	return rec.Result(), nil
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "media", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}
