package openai_test

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
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

func partTypes(parts []ai.StreamPart) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = typeName(p)
	}
	return out
}

func typeName(v any) string {
	return strings.TrimPrefix(fmt.Sprintf("%T", v), "ai.")
}

func TestResponsesModel_streamRequest(t *testing.T) {
	srv := newFakeServer(t, sseReply(sseOf(`{"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`)))
	if _, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	equalJSON(t, `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"stream":true}`, srv.last(t).Body)
	if got := srv.last(t).Header.Get("Accept"); got != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", got)
	}
}

func TestResponsesModel_streamWithoutItemEvents(t *testing.T) {
	body := sseOf(
		`{"type":"response.created","response":{"id":"r","status":"in_progress","output":[]}}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]},{"id":"fc_1","type":"function_call","call_id":"call_1","name":"f","arguments":"{}","status":"completed"}],"usage":{"input_tokens":1,"output_tokens":2}}}`,
	)
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	want := []string{"TextStart", "TextDelta", "TextEnd", "ToolInputStart", "ToolInputDelta", "ToolInputEnd", "ToolCall", "StepEnd"}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Errorf("part types mismatch (-want +got):\n%s", diff)
	}
	end, ok := parts[len(parts)-1].(ai.StepEnd)
	if !ok || end.FinishReason != ai.FinishToolCalls || end.Usage.Output != 2 {
		t.Errorf("last part = %#v, want StepEnd with tool-calls and 2 output tokens", parts[len(parts)-1])
	}
}

func TestResponsesModel_streamDeltasReconciled(t *testing.T) {
	body := sseOf(
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`,
		`{"type":"response.content_part.added","output_index":0,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":""}}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"msg_1","content_index":0,"delta":"hel"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`,
	)
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	var text strings.Builder
	for _, p := range parts {
		if d, ok := p.(ai.TextDelta); ok {
			text.WriteString(d.Text)
		}
	}
	if text.String() != "hello" {
		t.Errorf("streamed text = %q, want hello", text.String())
	}
}

func TestResponsesModel_streamReasoningSummaries(t *testing.T) {
	body := sseOf(
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"item_id":"rs_1","summary_index":0,"delta":"one"}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"item_id":"rs_1","summary_index":1,"delta":"two"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"one"},{"type":"summary_text","text":"two"}],"encrypted_content":"E"}}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`,
	)
	srv := newFakeServer(t, sseReply(body))
	res := ai.StreamText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxSteps: 1})
	got, err := res.Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	reasoning, ok := got.Content[0].(ai.Reasoning)
	if !ok || reasoning.Text != "one\n\ntwo" {
		t.Errorf("content[0] = %#v, want reasoning \"one\\n\\ntwo\"", got.Content[0])
	}
}

func TestResponsesModel_streamErrors(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		wantCode string
		wantIs   error
		wantText string
	}{
		{name: "failed response", body: sseOf(`{"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"server_error","message":"boom"}}}`), wantCode: "server_error"},
		{name: "error event", body: sseOf(`{"type":"error","code":"rate_limit_exceeded","message":"slow"}`), wantCode: "rate_limit_exceeded", wantIs: ai.ErrRateLimit},
		{name: "quota error event", body: sseOf(`{"type":"error","code":"insufficient_quota","message":"no"}`), wantCode: "insufficient_quota", wantIs: ai.ErrQuota},
		{name: "truncated", body: sseOf(`{"type":"response.created","response":{"id":"r","output":[]}}`), wantText: "ended before the response finished"},
		{name: "invalid event", body: "data: {nope\n\n", wantText: "decode stream event"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, sseReply(tt.body))
			_, err := streamAll(t, srv.provider().Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			if err == nil {
				t.Fatal("stream error = nil")
			}
			if apiErr, ok := errors.AsType[*ai.APIError](err); tt.wantCode != "" && (!ok || apiErr.Code != tt.wantCode) {
				t.Errorf("error = %v, want APIError code %q", err, tt.wantCode)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false (error %v)", tt.wantIs, err)
			}
			if tt.wantText != "" && !strings.Contains(err.Error(), tt.wantText) {
				t.Errorf("error = %v, want text %q", err, tt.wantText)
			}
		})
	}
}

func TestResponsesModel_streamHTTPError(t *testing.T) {
	srv2 := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"code":"invalid_api_key","message":"bad"}}`))
	})
	_, err := srv2.provider().Model("m").Stream(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("Stream() error = %v, want auth error", err)
	}
}

func TestResponsesModel_streamStopsEarly(t *testing.T) {
	body := sseOf(
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"msg_1","content_index":0,"delta":"a"}`,
		`{"type":"response.output_text.delta","output_index":0,"item_id":"msg_1","content_index":0,"delta":"b"}`,
	)
	srv := newFakeServer(t, sseReply(body))
	resp, err := srv.provider().Model("m").Stream(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	n := 0
	for _, err := range resp.Parts {
		if err != nil {
			t.Fatalf("stream error = %v", err)
		}
		n++
		break
	}
	if n != 1 {
		t.Errorf("received %d parts before break, want 1", n)
	}
}

func TestPartTypes(t *testing.T) {
	got := partTypes([]ai.StreamPart{ai.TextStart{}, ai.TextDelta{}})
	if diff := cmp.Diff([]string{"TextStart", "TextDelta"}, got); diff != "" {
		t.Errorf("partTypes mismatch (-want +got):\n%s", diff)
	}
}
