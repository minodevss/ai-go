package fal_test

import (
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

	"github.com/minodevss/ai-go/fal"
)

type captured struct {
	Method string
	Path   string
	Header http.Header
	Body   []byte
}

type fakeServer struct {
	URL      string
	handle   func(w http.ResponseWriter, r *http.Request, srv *fakeServer)
	t        *testing.T
	mu       sync.Mutex
	requests []captured
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newFakeServer(t *testing.T, handle func(w http.ResponseWriter, r *http.Request, srv *fakeServer)) *fakeServer {
	t.Helper()
	return &fakeServer{URL: "http://fal.test", handle: handle, t: t}
}

func (f *fakeServer) client() *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body []byte
		if r.Body != nil {
			var err error
			if body, err = io.ReadAll(r.Body); err != nil {
				f.t.Errorf("read request body: %v", err)
			}
		}
		f.mu.Lock()
		f.requests = append(f.requests, captured{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: body})
		f.mu.Unlock()
		rec := httptest.NewRecorder()
		f.handle(rec, r, f)
		return rec.Result(), nil
	})}
}

func (f *fakeServer) provider() *fal.Provider {
	return fal.New(fal.WithAPIKey("test-key"), fal.WithBaseURL(f.URL), fal.WithSyncBaseURL(f.URL+"/sync"), fal.WithPollInterval(time.Second), fal.WithHTTPClient(f.client()))
}

func (f *fakeServer) all() []captured {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]captured(nil), f.requests...)
}

func (f *fakeServer) fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return strings.NewReplacer(
		"https://queue.fal.run", f.URL,
		"IMAGE_URL", f.URL+"/files/out.png",
		"VIDEO_URL", f.URL+"/files/out.mp4",
	).Replace(string(raw))
}

func reply(status int, body string) func(http.ResponseWriter) {
	return func(w http.ResponseWriter) {
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
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("got is not JSON: %v\n%s", err, got)
	}
	if diff := cmp.Diff(w, g); diff != "" {
		t.Errorf("JSON mismatch (-want +got):\n%s", diff)
	}
}

type queueScript struct {
	statuses []string
	result   string
	resultOK int
}

func (q queueScript) handler(t *testing.T) func(http.ResponseWriter, *http.Request, *fakeServer) {
	t.Helper()
	var mu sync.Mutex
	polls := 0
	return func(w http.ResponseWriter, r *http.Request, srv *fakeServer) {
		switch {
		case r.Method == http.MethodPost:
			reply(http.StatusOK, srv.fixture(t, "submit.json"))(w)
		case r.Method == http.MethodPut:
			reply(http.StatusOK, `{"status":"OK"}`)(w)
		case strings.HasPrefix(r.URL.Path, "/files/"):
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = io.WriteString(w, "bytes:"+r.URL.Path)
		case strings.HasSuffix(r.URL.Path, "/status"):
			mu.Lock()
			name := q.statuses[min(polls, len(q.statuses)-1)]
			polls++
			mu.Unlock()
			status := http.StatusOK
			if name != "status_completed.json" && name != "status_error.json" {
				status = http.StatusAccepted
			}
			reply(status, srv.fixture(t, name))(w)
		default:
			status := q.resultOK
			if status == 0 {
				status = http.StatusOK
			}
			reply(status, srv.fixture(t, q.result))(w)
		}
	}
}
