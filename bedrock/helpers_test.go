package bedrock_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
	"github.com/minodevss/ai-go/internal/eventstream"
)

const testModel = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

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
		r.Body = io.NopCloser(bytes.NewReader(body))
		f.mu.Lock()
		f.requests = append(f.requests, captured{Path: r.RequestURI, Header: r.Header.Clone(), Body: body})
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

func errorReply(status int, errorType, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Errortype", errorType)
		w.Header().Set("X-Amzn-Requestid", "req-1")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func streamReply(t *testing.T, events ...event) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	raw := encodeEvents(t, events)
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(raw)
	}
}

func (f *fakeServer) provider(opts ...bedrock.Option) *bedrock.Provider {
	defaults := []bedrock.Option{
		bedrock.WithRegion("us-east-1"),
		bedrock.WithBaseURL(f.URL),
		bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"})),
	}
	return bedrock.New(append(defaults, opts...)...)
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

type event struct {
	Type        string          `json:"type"`
	MessageType string          `json:"messageType,omitzero"`
	Payload     json.RawMessage `json:"payload"`
}

func encodeEvents(t *testing.T, events []event) []byte {
	t.Helper()
	out, err := encodeEventsErr(events)
	if err != nil {
		t.Fatalf("encode events: %v", err)
	}
	return out
}

func encodeEventsErr(events []event) ([]byte, error) {
	var out []byte
	for _, e := range events {
		typeHeader, kind := ":event-type", "event"
		if e.MessageType != "" {
			typeHeader, kind = ":exception-type", e.MessageType
		}
		raw, err := eventstream.Encode(eventstream.Message{
			Headers: []eventstream.Header{
				eventstream.StringHeader(typeHeader, e.Type),
				eventstream.StringHeader(":content-type", "application/json"),
				eventstream.StringHeader(":message-type", kind),
			},
			Payload: e.Payload,
		})
		if err != nil {
			return nil, err
		}
		out = append(out, raw...)
	}
	return out, nil
}

func ev(typ, payload string) event { return event{Type: typ, Payload: json.RawMessage(payload)} }

func exception(typ, message string) event {
	return event{Type: typ, MessageType: "exception", Payload: json.RawMessage(`{"message":"` + message + `"}`)}
}

func textStream(text string) []event {
	return []event{
		ev("messageStart", `{"role":"assistant"}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"`+text+`"}}`),
		ev("contentBlockStop", `{"contentBlockIndex":0}`),
		ev("messageStop", `{"stopReason":"end_turn"}`),
		ev("metadata", `{"usage":{"inputTokens":3,"outputTokens":2,"totalTokens":5},"metrics":{"latencyMs":10}}`),
	}
}

const minimalResponse = `{"output":{"message":{"role":"assistant","content":[{"text":"hi"}]}},"stopReason":"end_turn","usage":{"inputTokens":3,"outputTokens":2,"totalTokens":5},"metrics":{"latencyMs":10}}`

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type limiterFunc func(context.Context) error

func (f limiterFunc) Wait(ctx context.Context) error { return f(ctx) }

func noRetries(r ai.TextRequest) ai.TextRequest {
	r.MaxRetries = new(0)
	return r
}

func readAll(t *testing.T, r *http.Request) []byte {
	t.Helper()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	return raw
}
