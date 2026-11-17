package ui_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/ui"
)

func sequentialIDs() func() string {
	var n atomic.Int32
	return func() string { return "msg_" + strconv.Itoa(int(n.Add(1))) }
}

func testKey() []byte { return bytes.Repeat([]byte("k"), 32) }

func chunksOf(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for frame := range strings.SplitSeq(body, "\n\n") {
		if frame == "" || frame == "data: [DONE]" {
			continue
		}
		payload, ok := strings.CutPrefix(frame, "data: ")
		if !ok {
			t.Fatalf("frame %q does not start with \"data: \"", frame)
		}
		var c map[string]any
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("frame %q is not JSON: %v", frame, err)
		}
		out = append(out, c)
	}
	return out
}

func typesOf(chunks []map[string]any) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i], _ = c["type"].(string)
	}
	return out
}

func respond(t *testing.T, factory func(context.Context) *ai.TextStream, opts ui.Options) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	ui.Respond(rec, req, factory, opts)
	return rec
}

func streamOf(m ai.LanguageModel, req ai.TextRequest) func(context.Context) *ai.TextStream {
	return func(ctx context.Context) *ai.TextStream { return ai.StreamText(ctx, m, req) }
}

func scripted(turns ...aitest.ModelOption) *aitest.Model { return aitest.NewModel(turns...) }

func jsonRequest(t *testing.T, body string) *http.Request {
	t.Helper()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func readRequest(t *testing.T, body string) ui.Request {
	t.Helper()
	req, err := ui.ReadRequest(jsonRequest(t, body), ui.Limits{})
	if err != nil {
		t.Fatalf("ReadRequest() error = %v", err)
	}
	return req
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return string(b)
}
