package elevenlabs_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/elevenlabs"
)

type captured struct {
	Path   string
	Query  url.Values
	Header http.Header
	Body   []byte
}

type fakeServer struct {
	*httptest.Server
	mu       sync.Mutex
	requests []captured
}

func newFakeServer(t *testing.T, handle http.HandlerFunc) *fakeServer {
	t.Helper()
	f := &fakeServer{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		f.mu.Lock()
		f.requests = append(f.requests, captured{Path: r.URL.Path, Query: r.URL.Query(), Header: r.Header.Clone(), Body: body})
		f.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeServer) provider(opts ...elevenlabs.Option) *elevenlabs.Provider {
	return elevenlabs.New(append([]elevenlabs.Option{elevenlabs.WithAPIKey("test-key"), elevenlabs.WithBaseURL(f.URL)}, opts...)...)
}

func (f *fakeServer) last(t *testing.T) captured {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		t.Fatal("no request was captured")
	}
	return f.requests[len(f.requests)-1]
}

func fixture(tb testing.TB, name string) string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}

func fixtureLines(tb testing.TB, name string) []string {
	tb.Helper()
	return strings.Split(strings.TrimSpace(fixture(tb, name)), "\n")
}

func reply(status int, contentType, body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
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
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("got is not JSON: %v\n%s", err, got)
	}
	if diff := cmp.Diff(w, g); diff != "" {
		t.Errorf("JSON mismatch (-want +got):\n%s", diff)
	}
}

func formFields(t *testing.T, c captured) (map[string][]string, []byte) {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(c.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		t.Fatalf("content type = %q (%v), want multipart/form-data", c.Header.Get("Content-Type"), err)
	}
	r := multipart.NewReader(bytes.NewReader(c.Body), params["boundary"])
	fields := map[string][]string{}
	var file []byte
	for {
		p, err := r.NextPart()
		if err != nil {
			return fields, file
		}
		data, err := io.ReadAll(p)
		if err != nil {
			t.Fatalf("read part: %v", err)
		}
		if p.FileName() != "" {
			file = data
			continue
		}
		fields[p.FormName()] = append(fields[p.FormName()], string(data))
	}
}

type wsServer struct {
	*httptest.Server
	mu      sync.Mutex
	url     *url.URL
	headers http.Header
}

func newWSServer(t *testing.T, script func(ctx context.Context, c *websocket.Conn)) *wsServer {
	t.Helper()
	s := &wsServer{}
	var handlers sync.WaitGroup
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers.Add(1)
		defer handlers.Done()
		s.mu.Lock()
		s.url, s.headers = r.URL, r.Header.Clone()
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

func (s *wsServer) provider() *elevenlabs.Provider {
	return elevenlabs.New(elevenlabs.WithAPIKey("test-key"), elevenlabs.WithBaseURL(s.URL))
}

func (s *wsServer) seen() (*url.URL, http.Header) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.url, s.headers
}

func readJSON(ctx context.Context, t *testing.T, c *websocket.Conn) map[string]any {
	t.Helper()
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Errorf("server read: %v", err)
		return map[string]any{}
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Errorf("client sent invalid JSON: %v", err)
	}
	return m
}
