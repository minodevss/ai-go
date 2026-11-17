package google_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/google"
	"github.com/minodevss/ai-go/internal/sse"
)

type captured struct {
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
		r.Body = io.NopCloser(bytes.NewReader(body))
		f.mu.Lock()
		f.requests = append(f.requests, captured{Path: r.URL.Path, Query: r.URL.RawQuery, Header: r.Header.Clone(), Body: body})
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
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
		}
	}
}

func (f *fakeServer) provider(opts ...google.Option) *google.Provider {
	return google.New(append([]google.Option{google.WithAPIKey("test-key"), google.WithBaseURL(f.URL + "/v1beta")}, opts...)...)
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

func requestItems(body []byte) ([]json.RawMessage, error) {
	var req struct {
		Contents []struct {
			Parts []json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	var out []json.RawMessage
	for _, c := range req.Contents {
		out = append(out, c.Parts...)
	}
	return out, nil
}

func responseItems(body []byte) ([]json.RawMessage, error) {
	var resp struct {
		Candidates []struct {
			Content struct {
				Parts []json.RawMessage `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	if len(resp.Candidates) == 0 {
		return nil, nil
	}
	return resp.Candidates[0].Content.Parts, nil
}

type textItem struct {
	Text             string `json:"text"`
	Thought          bool   `json:"thought,omitzero"`
	ThoughtSignature string `json:"thoughtSignature,omitzero"`
}

func streamItems(body []byte) ([]json.RawMessage, error) {
	r := sse.NewReader(bytes.NewReader(body))
	var items []json.RawMessage
	var open *textItem
	flush := func() error {
		if open == nil {
			return nil
		}
		var b bytes.Buffer
		enc := json.NewEncoder(&b)
		enc.SetEscapeHTML(false)
		err := enc.Encode(open)
		open = nil
		items = append(items, bytes.TrimRight(b.Bytes(), "\n"))
		return err
	}
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			return items, flush()
		}
		if err != nil {
			return nil, err
		}
		parts, err := responseItems([]byte(ev.Data))
		if err != nil {
			return nil, err
		}
		for _, raw := range parts {
			var keys map[string]json.RawMessage
			if err := json.Unmarshal(raw, &keys); err != nil {
				return nil, err
			}
			var p textItem
			textOnly := keys["text"] != nil
			for k := range keys {
				if k != "text" && k != "thought" && k != "thoughtSignature" {
					textOnly = false
				}
			}
			if !textOnly {
				if err := flush(); err != nil {
					return nil, err
				}
				items = append(items, raw)
				continue
			}
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			if open != nil && (open.Thought != p.Thought || p.ThoughtSignature != "" && open.Text != "") {
				if err := flush(); err != nil {
					return nil, err
				}
			}
			if p.Text == "" && p.ThoughtSignature == "" {
				continue
			}
			if open == nil {
				open = &textItem{Thought: p.Thought}
			}
			open.Text += p.Text
			if p.ThoughtSignature != "" {
				open.ThoughtSignature = p.ThoughtSignature
				if err := flush(); err != nil {
					return nil, err
				}
			}
		}
	}
}

func callIDs(body []byte) ([]string, []string, error) {
	items, err := requestItems(body)
	if err != nil {
		return nil, nil, err
	}
	var calls, results []string
	for _, raw := range items {
		var it struct {
			FunctionCall *struct {
				ID string `json:"id"`
			} `json:"functionCall"`
			FunctionResponse *struct {
				ID string `json:"id"`
			} `json:"functionResponse"`
		}
		if err := json.Unmarshal(raw, &it); err != nil {
			return nil, nil, err
		}
		if it.FunctionCall != nil {
			calls = append(calls, it.FunctionCall.ID)
		}
		if it.FunctionResponse != nil {
			results = append(results, it.FunctionResponse.ID)
		}
	}
	return calls, results, nil
}

var errCanceled = context.Canceled

func jsonErrorReply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}
