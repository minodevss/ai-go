package assemblyai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/assemblyai"
)

type captured struct {
	Method string
	Path   string
	Query  string
	Header http.Header
	Body   []byte
}

type fakeServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []captured
}

func newFakeServer(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) *fakeServer {
	t.Helper()
	f := &fakeServer{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		f.mu.Lock()
		f.requests = append(f.requests, captured{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Header: r.Header.Clone(), Body: body})
		f.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeServer) provider(opts ...assemblyai.Option) *assemblyai.Provider {
	return assemblyai.New(append([]assemblyai.Option{assemblyai.WithAPIKey("test-key"), assemblyai.WithBaseURL(f.URL), assemblyai.WithTokenURL(f.URL), assemblyai.WithStreamingURL(f.URL)}, opts...)...)
}

func (f *fakeServer) all() []captured {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]captured(nil), f.requests...)
}

func (f *fakeServer) last(t *testing.T) captured {
	t.Helper()
	all := f.all()
	if len(all) == 0 {
		t.Fatal("no request was captured")
	}
	return all[len(all)-1]
}

func fixture(tb testing.TB, name string) string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}

func writeJSON(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, body)
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

type wsServer struct {
	*httptest.Server
	mu     sync.Mutex
	query  string
	header http.Header
}

func newWSServer(t *testing.T, script func(ctx context.Context, c *websocket.Conn)) *wsServer {
	t.Helper()
	s := &wsServer{}
	var handlers sync.WaitGroup
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers.Add(1)
		defer handlers.Done()
		s.mu.Lock()
		s.query, s.header = r.URL.Path+"?"+r.URL.RawQuery, r.Header.Clone()
		s.mu.Unlock()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		script(t.Context(), c)
	}))
	t.Cleanup(func() {
		s.Close()
		handlers.Wait()
	})
	return s
}

func (s *wsServer) provider() *assemblyai.Provider {
	return assemblyai.New(assemblyai.WithAPIKey("test-key"), assemblyai.WithStreamingURL(s.URL))
}

func (s *wsServer) seen() (string, http.Header) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.query, s.header
}
