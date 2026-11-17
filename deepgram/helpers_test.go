package deepgram_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
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

	"github.com/minodevss/ai-go/deepgram"
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

func newFakeServer(t *testing.T, handle func(w http.ResponseWriter, r *http.Request)) *fakeServer {
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

func (f *fakeServer) provider(opts ...deepgram.Option) *deepgram.Provider {
	return deepgram.New(append([]deepgram.Option{deepgram.WithAPIKey("test-key"), deepgram.WithBaseURL(f.URL + "/v1")}, opts...)...)
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

func (f *fakeServer) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func jsonReply(body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}

func fixtureLines(t *testing.T, name string) []string {
	t.Helper()
	return strings.Split(strings.TrimSpace(fixture(t, name)), "\n")
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

func (s *wsServer) provider() *deepgram.Provider {
	return deepgram.New(deepgram.WithAPIKey("test-key"), deepgram.WithBaseURL(s.URL+"/v1"))
}

func (s *wsServer) seen() (*url.URL, http.Header) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.url, s.headers
}

type frame struct {
	Kind websocket.MessageType
	Data []byte
}

func readFrames(ctx context.Context, c *websocket.Conn) <-chan frame {
	ch := make(chan frame, 64)
	go func() {
		defer close(ch)
		for {
			kind, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			ch <- frame{Kind: kind, Data: data}
		}
	}()
	return ch
}

func writeText(ctx context.Context, t *testing.T, c *websocket.Conn, messages ...string) {
	t.Helper()
	for _, m := range messages {
		if err := c.Write(ctx, websocket.MessageText, []byte(m)); err != nil {
			t.Errorf("server write: %v", err)
			return
		}
	}
}
