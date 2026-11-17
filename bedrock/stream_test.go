package bedrock_test

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
	"github.com/minodevss/ai-go/internal/eventstream"
)

func collect(t *testing.T, f *fakeServer, req ai.TextRequest) ([]ai.StreamPart, *ai.TextResult, error) {
	t.Helper()
	s := ai.StreamText(t.Context(), f.provider().Model(testModel), req)
	var parts []ai.StreamPart
	for p := range s.Parts() {
		parts = append(parts, p)
	}
	res, err := s.Result()
	return parts, res, err
}

func TestLanguageModel_Stream_text(t *testing.T) {
	f := newFakeServer(t, streamReply(t, textStream("Hello")...))
	parts, res, err := collect(t, f, hello())
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if res.Text != "Hello" || res.Usage.Input != 3 || res.Usage.Output != 2 || res.FinishReason != ai.FinishStop {
		t.Errorf("Result() = %q %+v %q, want Hello with 3 in, 2 out, stop", res.Text, res.Usage, res.FinishReason)
	}
	var sawDelta bool
	for _, p := range parts {
		if d, ok := p.(ai.TextDelta); ok && d.Text == "Hello" {
			sawDelta = true
		}
	}
	if !sawDelta {
		t.Errorf("parts = %#v, want a text delta", parts)
	}
	got := f.last(t)
	if got.Path != "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream" || got.Header.Get("Accept") != "application/vnd.amazon.eventstream" {
		t.Errorf("request = %q accept %q, want converse-stream with the event stream accept header", got.Path, got.Header.Get("Accept"))
	}
}

func TestLanguageModel_Stream_guardrailStreamProcessingMode(t *testing.T) {
	f := newFakeServer(t, streamReply(t, textStream("x")...))
	_, _, err := collect(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Provider: ai.Providers(bedrock.Options{Guardrail: &bedrock.Guardrail{Identifier: "g", Version: "1", StreamProcessingMode: bedrock.ProcessingSync}})})
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	equalJSON(t, `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"guardrailConfig":{"guardrailIdentifier":"g","guardrailVersion":"1","streamProcessingMode":"sync"}}`, f.last(t).Body)
}

func TestLanguageModel_Stream_toolUseWithReasoning(t *testing.T) {
	f := newFakeServer(t, streamReply(t,
		ev("messageStart", `{"role":"assistant"}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"Need weather"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"SIG"}}}`),
		ev("contentBlockStop", `{"contentBlockIndex":0}`),
		ev("contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tooluse_1","name":"get_weather"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"city\":"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"\"Seoul\"}"}}}`),
		ev("contentBlockStop", `{"contentBlockIndex":1}`),
		ev("messageStop", `{"stopReason":"tool_use"}`),
		ev("metadata", `{"usage":{"inputTokens":9,"outputTokens":7,"totalTokens":16,"cacheReadInputTokens":100,"cacheWriteInputTokens":0}}`),
	))
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather")
	_, res, err := collect(t, f, withOneStep(ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{weather}}))
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if res.FinishReason != ai.FinishToolCalls || res.Usage.Input != 109 || res.Usage.CacheRead != 100 {
		t.Errorf("Result() finish %q usage %+v, want tool-calls with input 109 and cache read 100", res.FinishReason, res.Usage)
	}
	if len(res.Content) != 2 {
		t.Fatalf("Content = %#v, want reasoning and a tool call", res.Content)
	}
	call, ok := res.Content[1].(ai.ToolCall)
	if !ok || call.ID != "tooluse_1" || string(call.Input) != `{"city":"Seoul"}` {
		t.Errorf("Content[1] = %#v, want the get_weather call", res.Content[1])
	}
}

func TestLanguageModel_Stream_forcedToolBecomesText(t *testing.T) {
	f := newFakeServer(t, streamReply(t,
		ev("contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tooluse_1","name":"response"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"title\":"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"x\"}"}}}`),
		ev("contentBlockStop", `{"contentBlockIndex":0}`),
		ev("messageStop", `{"stopReason":"tool_use"}`),
		ev("metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`),
	))
	schema := ai.ObjectOutput[struct {
		Title string `json:"title"`
	}]()
	res, err := ai.StreamText(t.Context(), f.provider().Model(testModel), ai.TextRequest{
		Messages: []ai.Message{ai.User("go")},
		Output:   schema,
		Provider: ai.Providers(bedrock.Options{OutputViaTool: true}),
	}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if res.Text != `{"title":"x"}` || res.FinishReason != ai.FinishStop {
		t.Errorf("Result() = %q %q, want the tool input as text and stop", res.Text, res.FinishReason)
	}
}

func TestLanguageModel_Stream_exceptionMidStream(t *testing.T) {
	events := []event{
		ev("messageStart", `{"role":"assistant"}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"par"}}`),
		exception("modelStreamErrorException", "stream broke"),
	}
	tests := []struct {
		name          string
		exception     event
		wantCode      string
		wantStatus    int
		wantRetryable bool
		wantIs        error
	}{
		{name: "model stream error", exception: events[2], wantCode: "ModelStreamErrorException", wantStatus: http.StatusFailedDependency},
		{name: "throttling", exception: exception("throttlingException", "slow"), wantCode: "ThrottlingException", wantStatus: http.StatusTooManyRequests, wantRetryable: true, wantIs: ai.ErrRateLimit},
		{name: "service unavailable", exception: exception("serviceUnavailableException", "down"), wantCode: "ServiceUnavailableException", wantStatus: http.StatusServiceUnavailable, wantRetryable: true},
		{name: "internal server", exception: exception("internalServerException", "oops"), wantCode: "InternalServerException", wantStatus: http.StatusInternalServerError, wantRetryable: true},
		{name: "validation", exception: exception("validationException", "bad"), wantCode: "ValidationException", wantStatus: http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, streamReply(t, events[0], events[1], tt.exception))
			_, res, err := collect(t, f, hello())
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Result() error = %v, want *ai.APIError", err)
			}
			if apiErr.Code != tt.wantCode || apiErr.StatusCode != tt.wantStatus || apiErr.Retryable() != tt.wantRetryable {
				t.Errorf("APIError = code %q status %d retryable %v, want %q %d %v", apiErr.Code, apiErr.StatusCode, apiErr.Retryable(), tt.wantCode, tt.wantStatus, tt.wantRetryable)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(err, %v) = false", tt.wantIs)
			}
			if res != nil && res.Text != "" {
				t.Errorf("Result().Text = %q, want the failed step's partial output excluded", res.Text)
			}
		})
	}
}

func TestLanguageModel_Stream_corruptFrames(t *testing.T) {
	good := encodeEvents(t, textStream("Hello"))
	flipped := bytes.Clone(good)
	flipped[len(flipped)/2] ^= 0xff
	tests := []struct {
		name    string
		body    []byte
		wantErr error
		wantMsg string
	}{
		{name: "bit flip", body: flipped, wantErr: eventstream.ErrChecksum},
		{name: "truncated frame", body: good[:len(good)-5], wantErr: io.ErrUnexpectedEOF},
		{name: "ends before message stop", body: encodeEvents(t, textStream("Hello")[:2]), wantMsg: "stream ended before the message finished"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				_, _ = w.Write(tt.body)
			})
			_, _, err := collect(t, f, hello())
			if err == nil {
				t.Fatal("Result() error = nil, want an error")
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Errorf("Result() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantMsg != "" && !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("Result() error = %v, want %q", err, tt.wantMsg)
			}
		})
	}
}

func TestLanguageModel_Stream_httpError(t *testing.T) {
	f := newFakeServer(t, errorReply(http.StatusTooManyRequests, "ThrottlingException", `{"message":"slow"}`))
	s := ai.StreamText(t.Context(), f.provider().Model(testModel), noRetries(hello()))
	for range s.Parts() {
	}
	if !errors.Is(s.Err(), ai.ErrRateLimit) {
		t.Errorf("Err() = %v, want ErrRateLimit", s.Err())
	}
}

func TestLanguageModel_Stream_citationsAndToolResult(t *testing.T) {
	f := newFakeServer(t, streamReply(t,
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"The grass is green."}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"citation":{"title":"doc","location":{"documentPage":{"documentIndex":1,"start":2,"end":3}},"sourceContent":[{"text":"green"}]}}}`),
		ev("contentBlockStop", `{"contentBlockIndex":0}`),
		ev("contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"srv_1","name":"web_search","type":"server_tool_use"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"q\":\"x\"}"}}}`),
		ev("contentBlockStop", `{"contentBlockIndex":1}`),
		ev("contentBlockStart", `{"contentBlockIndex":2,"start":{"toolResult":{"toolUseId":"srv_1","status":"error","type":"web_search_result"}}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"toolResult":[{"text":"rate "}]}}`),
		ev("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"toolResult":[{"text":"limited"}]}}`),
		ev("contentBlockStop", `{"contentBlockIndex":2}`),
		ev("messageStop", `{"stopReason":"end_turn"}`),
		ev("metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`),
	))
	_, res, err := collect(t, f, hello())
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	text, ok := res.Content[0].(ai.Text)
	if !ok || len(text.Citations) != 1 {
		t.Fatalf("Content[0] = %#v, want text with one citation", res.Content[0])
	}
	want := ai.Citation{Kind: "page", Title: "doc", CitedText: "green", DocIndex: new(1), Start: 2, End: 3}
	got := text.Citations[0]
	got.Provider = nil
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("citation mismatch (-want +got):\n%s", diff)
	}
	call, ok := res.Content[1].(ai.ToolCall)
	if !ok || !call.ProviderExecuted || call.Name != "web_search" {
		t.Errorf("Content[1] = %#v, want a provider-executed web_search call", res.Content[1])
	}
	result, ok := res.Content[2].(ai.ToolResult)
	if !ok || !result.IsError || result.Name != "web_search" || string(result.Output.JSON) != `[{"text":"rate limited"}]` {
		t.Errorf("Content[2] = %#v, want an error result named web_search with merged text", res.Content[2])
	}
	if res.FinishReason != ai.FinishStop {
		t.Errorf("FinishReason = %q, want stop", res.FinishReason)
	}
}
