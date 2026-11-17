package anthropic_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func sseOf(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		typ, _, _ := strings.Cut(strings.TrimPrefix(e, `{"type":"`), `"`)
		b.WriteString("event: " + typ + "\ndata: " + e + "\n\n")
	}
	return b.String()
}

func streamAll(t *testing.T, model ai.LanguageModel, c *ai.Call) ([]ai.StreamPart, error) {
	t.Helper()
	resp, err := model.Stream(t.Context(), c)
	if err != nil {
		return nil, err
	}
	var parts []ai.StreamPart
	for p, err := range resp.Parts {
		if err != nil {
			return parts, err
		}
		parts = append(parts, p)
	}
	return parts, nil
}

func typeName(v any) string {
	return strings.TrimPrefix(fmt.Sprintf("%T", v), "ai.")
}

func partTypes(parts []ai.StreamPart) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = typeName(p)
	}
	return out
}

const startEvent = `{"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"cache_read_input_tokens":10,"output_tokens":1}}}`

func TestLanguageModel_streamRequest(t *testing.T) {
	srv := newFakeServer(t, sseReply(sseOf(startEvent, `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}`, `{"type":"message_stop"}`)))
	if _, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	equalJSON(t, `{"model":"m","max_tokens":16000,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stream":true}`, srv.last(t).Body)
	if got := srv.last(t).Header.Get("Accept"); got != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", got)
	}
}

func TestLanguageModel_streamSignatures(t *testing.T) {
	body := sseOf(
		startEvent,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG-"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"ONE"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"SIG-TWO"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking","data":"RED"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"thinking_delta","thinking":"third"}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"signature_delta","signature":"SIG-THREE"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":9}}`,
		`{"type":"message_stop"}`,
	)
	srv := newFakeServer(t, sseReply(body))
	res, err := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("q")}}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	want := []struct{ text, block string }{
		{"first thought", `{"type":"thinking","thinking":"first thought","signature":"SIG-ONE"}`},
		{"", `{"type":"thinking","thinking":"","signature":"SIG-TWO"}`},
		{"", `{"type":"redacted_thinking","data":"RED"}`},
		{"third", `{"type":"thinking","thinking":"third","signature":"SIG-THREE"}`},
	}
	if len(res.Content) != len(want) {
		t.Fatalf("content has %d parts, want %d: %#v", len(res.Content), len(want), res.Content)
	}
	for i, w := range want {
		r, ok := res.Content[i].(ai.Reasoning)
		if !ok || r.Text != w.text {
			t.Fatalf("part %d = %#v, want reasoning %q", i, res.Content[i], w.text)
		}
		data, _ := ai.ProviderValue[map[string]any](r.Provider, "anthropic")
		var wantBlock map[string]any
		if err := jsonUnmarshal(w.block, &wantBlock); err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(wantBlock, data["block"]); diff != "" {
			t.Errorf("part %d block mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestLanguageModel_streamPartsAndUsage(t *testing.T) {
	body := sseOf(
		startEvent,
		`{"type":"ping"}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		`{"type":"future_event","data":1}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"c","document_index":2,"document_title":"T","start_char_index":4,"end_char_index":9}}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\": \"Se"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"oul\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":15,"output_tokens_details":{"thinking_tokens":4}}}`,
		`{"type":"message_stop"}`,
	)
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	want := []string{"TextStart", "TextDelta", "TextDelta", "TextEnd", "Citation", "ToolInputStart", "ToolInputDelta", "ToolInputDelta", "ToolInputEnd", "ToolCall", "StepEnd"}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Errorf("part types mismatch (-want +got):\n%s", diff)
	}
	cite, ok := parts[4].(ai.Citation)
	if !ok || cite.Kind != "char" || cite.Start != 4 || cite.End != 9 || cite.DocIndex == nil || *cite.DocIndex != 2 || cite.Title != "T" {
		t.Errorf("citation = %#v, want a char citation of document 2", parts[4])
	}
	call, ok := parts[9].(ai.ToolCall)
	if !ok || call.ID != "toolu_1" || string(call.Input) != `{"city":"Seoul"}` {
		t.Errorf("tool call = %#v, want toolu_1 with the accumulated input", parts[9])
	}
	end, ok := parts[10].(ai.StepEnd)
	if !ok || end.FinishReason != ai.FinishToolCalls || end.FinishRaw != "tool_use" {
		t.Fatalf("last part = %#v, want StepEnd with tool-calls", parts[10])
	}
	got := end.Usage
	got.Raw = nil
	if diff := cmp.Diff(ai.Usage{Input: 35, Output: 15, CacheRead: 10, Reasoning: 4}, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
}

func TestLanguageModel_streamForcedToolOutput(t *testing.T) {
	body := sseOf(
		startEvent,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_f","name":"response","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ok\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"true}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	)
	srv := newFakeServer(t, sseReply(body))
	res, err := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{
		Messages: []ai.Message{ai.User("q")},
		Output:   ai.JSONOutput(),
		Provider: ai.Providers(anthropic.Options{OutputViaTool: true}),
	}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if res.Text != `{"ok":true}` || res.FinishReason != ai.FinishStop {
		t.Errorf("text = %q finish = %q, want the tool input as text and stop", res.Text, res.FinishReason)
	}
}

func TestLanguageModel_streamMidStreamErrors(t *testing.T) {
	tests := []struct {
		name          string
		event         string
		wantStatus    int
		wantRetryable bool
		wantIs        error
	}{
		{name: "overloaded", event: `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, wantStatus: 529, wantRetryable: true},
		{name: "rate limit", event: `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, wantStatus: 429, wantRetryable: true, wantIs: ai.ErrRateLimit},
		{name: "api error", event: `{"type":"error","error":{"type":"api_error","message":"oops"}}`, wantStatus: 500, wantRetryable: true},
		{name: "invalid request", event: `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`, wantStatus: 400},
		{name: "authentication", event: `{"type":"error","error":{"type":"authentication_error","message":"x"}}`, wantStatus: 401, wantIs: ai.ErrAuth},
		{name: "permission", event: `{"type":"error","error":{"type":"permission_error","message":"x"}}`, wantStatus: 403, wantIs: ai.ErrAuth},
		{name: "billing", event: `{"type":"error","error":{"type":"billing_error","message":"x"}}`, wantStatus: 402, wantIs: ai.ErrQuota},
		{name: "not found", event: `{"type":"error","error":{"type":"not_found_error","message":"x"}}`, wantStatus: 404},
		{name: "conflict", event: `{"type":"error","error":{"type":"conflict_error","message":"x"}}`, wantStatus: 409, wantRetryable: true},
		{name: "too large", event: `{"type":"error","error":{"type":"request_too_large","message":"x"}}`, wantStatus: 413},
		{name: "prompt too long", event: `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}`, wantStatus: 400, wantIs: ai.ErrContextTooLong},
		{name: "timeout", event: `{"type":"error","error":{"type":"timeout_error","message":"x"}}`, wantStatus: 504, wantRetryable: true},
		{name: "unknown type", event: `{"type":"error","error":{"type":"brand_new_error","message":"?"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := sseOf(startEvent, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`, tt.event)
			srv := newFakeServer(t, sseReply(body))
			parts, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("stream error = %v, want *ai.APIError", err)
			}
			if apiErr.Provider != "anthropic" || apiErr.StatusCode != tt.wantStatus || apiErr.Retryable() != tt.wantRetryable {
				t.Errorf("error = %+v, want status %d retryable %v", apiErr, tt.wantStatus, tt.wantRetryable)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			if len(parts) != 2 {
				t.Errorf("got %d parts before the error, want 2", len(parts))
			}
		})
	}
}

func TestLanguageModel_streamEndsEarly(t *testing.T) {
	srv := newFakeServer(t, sseReply(sseOf(startEvent, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)))
	_, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err == nil || !strings.Contains(err.Error(), "stream ended before the message finished") {
		t.Errorf("stream error = %v, want an early end error", err)
	}
}

func TestLanguageModel_streamInvalidToolInput(t *testing.T) {
	body := sseOf(
		startEvent,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`{"type":"content_block_stop","index":0}`,
	)
	srv := newFakeServer(t, sseReply(body))
	_, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err == nil || !strings.Contains(err.Error(), "not valid JSON") {
		t.Errorf("stream error = %v, want an invalid tool input error", err)
	}
}

func TestLanguageModel_streamHTTPError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(529)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`))
	})
	_, err := srv.provider().Model("m").Stream(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.StatusCode != 529 || !apiErr.Retryable() {
		t.Errorf("Stream() error = %v, want a retryable 529", err)
	}
}

func TestLanguageModel_streamConsumerStops(t *testing.T) {
	body := sseOf(
		startEvent,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"b"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	)
	srv := newFakeServer(t, sseReply(body))
	s := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	for p := range s.Parts() {
		if _, ok := p.(ai.TextDelta); ok {
			break
		}
	}
	if _, err := s.Result(); !errors.Is(err, context.Canceled) {
		t.Errorf("Result() error = %v, want context.Canceled after the consumer stopped", err)
	}
}
