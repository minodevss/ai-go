package openaicompat_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
)

type captured struct {
	Path   string
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
		f.requests = append(f.requests, captured{Path: r.URL.Path, Header: r.Header.Clone(), Body: body})
		f.mu.Unlock()
		handle(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func jsonReply(body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func sseReply(chunks ...string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		var b strings.Builder
		for _, c := range chunks {
			b.WriteString("data: " + c + "\n\n")
		}
		b.WriteString("data: [DONE]\n\n")
		_, _ = io.WriteString(w, b.String())
	}
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

func streamAll(t *testing.T, model ai.LanguageModel, c *ai.Call) []ai.StreamPart {
	t.Helper()
	resp, err := model.Stream(t.Context(), c)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var parts []ai.StreamPart
	for p, err := range resp.Parts {
		if err != nil {
			t.Fatalf("stream error = %v", err)
		}
		parts = append(parts, p)
	}
	return parts
}

func partTypes(parts []ai.StreamPart) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = strings.TrimPrefix(fmt.Sprintf("%T", p), "ai.")
	}
	return out
}

func hi() []ai.Message { return []ai.Message{ai.User("hi")} }

const minimalChat = `{"id":"c1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`

func bodyOf(raw []byte) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(raw), &m); err != nil {
		return nil
	}
	return m
}

func as[T any](t *testing.T, v any) T {
	t.Helper()
	got, ok := v.(T)
	if !ok {
		t.Fatalf("%v is %T, want %T", v, v, *new(T))
	}
	return got
}

func (f *fakeServer) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}
