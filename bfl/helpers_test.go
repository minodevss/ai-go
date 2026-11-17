package bfl_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/bfl"
)

type captured struct {
	Method string
	Path   string
	Header http.Header
	Body   []byte
	At     time.Time
}

type fakeServer struct {
	URL      string
	handle   func(w http.ResponseWriter, r *http.Request)
	mu       sync.Mutex
	requests []captured
	t        *testing.T
}

func newFakeServer(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) *fakeServer {
	t.Helper()
	return &fakeServer{URL: "http://api.test", handle: handle, t: t}
}

func (f *fakeServer) RoundTrip(r *http.Request) (*http.Response, error) {
	var body []byte
	if r.Body != nil {
		var err error
		if body, err = io.ReadAll(r.Body); err != nil {
			return nil, err
		}
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	f.mu.Lock()
	f.requests = append(f.requests, captured{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: body, At: time.Now()})
	f.mu.Unlock()
	rec := httptest.NewRecorder()
	f.handle(rec, r)
	return rec.Result(), nil
}

func (f *fakeServer) Client() *http.Client { return &http.Client{Transport: f} }

func (f *fakeServer) provider(opts ...bfl.Option) *bfl.Provider {
	base := []bfl.Option{bfl.WithAPIKey("test-key"), bfl.WithBaseURL(f.URL), bfl.WithHTTPClient(f.Client())}
	return bfl.New(append(base, opts...)...)
}

func (f *fakeServer) all() []captured {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]captured(nil), f.requests...)
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}

func fixtureAt(t *testing.T, name, host string) string {
	t.Helper()
	return strings.ReplaceAll(fixture(t, name), "REPLACE", host)
}

func jsonReply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func equalJSON(t *testing.T, want string, got []byte) {
	t.Helper()
	var w, g any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want is not JSON: %v", err)
	}
	if err := json.Unmarshal(bytes.TrimSpace(got), &g); err != nil {
		t.Fatalf("got is not JSON: %v\n%s", err, got)
	}
	if diff := cmp.Diff(w, g); diff != "" {
		t.Errorf("JSON mismatch (-want +got):\n%s", diff)
	}
}
