package aiotel_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aiotel"
)

type harness struct {
	spans   *tracetest.InMemoryExporter
	metrics *sdkmetric.ManualReader
	tel     *aiotel.Telemetry
}

func newHarness(t *testing.T, opts ...aiotel.Option) *harness {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		if err := tp.Shutdown(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("tracer shutdown error = %v", err)
		}
		if err := mp.Shutdown(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("meter shutdown error = %v", err)
		}
	})
	tel, err := aiotel.New(append([]aiotel.Option{aiotel.WithTracerProvider(tp), aiotel.WithMeterProvider(mp)}, opts...)...)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return &harness{spans: exporter, metrics: reader, tel: tel}
}

func (h *harness) span(t *testing.T, name string) tracetest.SpanStub {
	t.Helper()
	for _, s := range h.spans.GetSpans() {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("span %q not found in %v", name, h.spans.GetSpans().Snapshots())
	return tracetest.SpanStub{}
}

func (h *harness) collect(t *testing.T) map[string]metricdata.Metrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := h.metrics.Collect(t.Context(), &rm); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	out := map[string]metricdata.Metrics{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			out[m.Name] = m
		}
	}
	return out
}

func attrs(s tracetest.SpanStub) map[string]attribute.Value {
	out := map[string]attribute.Value{}
	for _, kv := range s.Attributes {
		out[string(kv.Key)] = kv.Value
	}
	return out
}

var info = ai.StepInfo{Provider: "google", ModelID: "gemini-3-pro", Number: 1}

func runStep(t *testing.T, h *harness, step ai.Step, err error) {
	t.Helper()
	ctx := h.tel.StartStep(t.Context(), info)
	h.tel.FinishStep(ctx, info, step, err)
}

func TestTelemetry_stepSpan(t *testing.T) {
	h := newHarness(t)
	runStep(t, h, ai.Step{
		Content:      ai.Parts{ai.Text{Text: "secret answer"}},
		Usage:        ai.Usage{Input: 100, Output: 40, CacheRead: 30, Reasoning: 10},
		FinishReason: ai.FinishStop,
	}, nil)
	s := h.span(t, "chat gemini-3-pro")
	if s.SpanKind != trace.SpanKindClient {
		t.Errorf("SpanKind = %v, want client", s.SpanKind)
	}
	got := attrs(s)
	want := map[string]any{
		"gen_ai.operation.name":                "chat",
		"gen_ai.provider.name":                 "gcp.gemini",
		"gen_ai.request.model":                 "gemini-3-pro",
		"gen_ai.usage.input_tokens":            int64(100),
		"gen_ai.usage.output_tokens":           int64(40),
		"gen_ai.usage.cache_read.input_tokens": int64(30),
		"gen_ai.usage.reasoning.output_tokens": int64(10),
		"gen_ai.response.finish_reasons":       []string{"stop"},
		"ai.step.number":                       int64(1),
	}
	for k, v := range want {
		if diff := cmp.Diff(v, got[k].AsInterface()); diff != "" {
			t.Errorf("attribute %s mismatch (-want +got):\n%s", k, diff)
		}
	}
	for _, k := range []string{"gen_ai.usage.cache_write.input_tokens", "gen_ai.output.messages"} {
		if _, ok := got[k]; ok {
			t.Errorf("attribute %s present, want absent", k)
		}
	}
	if s.Status.Code != codes.Unset {
		t.Errorf("Status = %v, want unset", s.Status)
	}
}

func TestTelemetry_contentIsOptIn(t *testing.T) {
	h := newHarness(t, aiotel.CaptureContent())
	runStep(t, h, ai.Step{
		Content: ai.Parts{
			ai.Text{Text: "hello"},
			ai.ToolCall{ID: "c1", Name: "lookup", Input: json.RawMessage(`{"q":"x"}`)},
		},
		FinishReason: ai.FinishToolCalls,
	}, nil)
	got := attrs(h.span(t, "chat gemini-3-pro"))["gen_ai.output.messages"].AsString()
	var msgs []map[string]any
	if err := json.Unmarshal([]byte(got), &msgs); err != nil {
		t.Fatalf("gen_ai.output.messages = %q is not JSON: %v", got, err)
	}
	want := []map[string]any{{
		"role":          "assistant",
		"finish_reason": "tool-calls",
		"parts": []any{
			map[string]any{"type": "text", "content": "hello"},
			map[string]any{"type": "tool_call", "id": "c1", "name": "lookup", "arguments": map[string]any{"q": "x"}},
		},
	}}
	if diff := cmp.Diff(want, msgs); diff != "" {
		t.Errorf("output messages mismatch (-want +got):\n%s", diff)
	}
}

func TestTelemetry_stepError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"api status", &ai.APIError{Provider: "google", StatusCode: 503, Body: "private body"}, "503"},
		{"api code", &ai.APIError{StatusCode: 429, Code: "rate_limit_exceeded"}, "rate_limit_exceeded"},
		{"timeout", context.DeadlineExceeded, "timeout"},
		{"canceled", context.Canceled, "canceled"},
		{"other", errors.New("boom with private data"), "_OTHER"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t)
			runStep(t, h, ai.Step{}, tt.err)
			s := h.span(t, "chat gemini-3-pro")
			if got := attrs(s)["error.type"].AsString(); got != tt.want {
				t.Errorf("error.type = %q, want %q", got, tt.want)
			}
			if s.Status.Code != codes.Error || s.Status.Description != tt.want {
				t.Errorf("Status = %+v, want error with description %q", s.Status, tt.want)
			}
			if len(s.Events) != 0 {
				t.Errorf("events = %v, want none without CaptureContent", s.Events)
			}
		})
	}
}

func TestTelemetry_errorMessageOnlyWithContent(t *testing.T) {
	h := newHarness(t, aiotel.CaptureContent())
	runStep(t, h, ai.Step{}, errors.New("boom"))
	if events := h.span(t, "chat gemini-3-pro").Events; len(events) != 1 || events[0].Name != "exception" {
		t.Errorf("events = %v, want one exception event", events)
	}
}

func TestTelemetry_toolSpanAndContext(t *testing.T) {
	h := newHarness(t)
	call := ai.ToolCall{ID: "call-1", Name: "weather", Input: json.RawMessage(`{"city":"Seoul"}`)}
	ctx := h.tel.StartTool(t.Context(), call)
	active := trace.SpanFromContext(ctx).SpanContext()
	if !active.IsValid() {
		t.Fatal("tool context has no active span")
	}
	h.tel.FinishTool(ctx, call, ai.ToolResult{}, 250*time.Millisecond, ai.ToolErrorf("no data"))
	s := h.span(t, "execute_tool weather")
	if s.SpanContext.SpanID() != active.SpanID() {
		t.Errorf("tool span id = %v, want the span carried by the context %v", s.SpanContext.SpanID(), active.SpanID())
	}
	if s.SpanKind != trace.SpanKindInternal {
		t.Errorf("SpanKind = %v, want internal", s.SpanKind)
	}
	got := attrs(s)
	if got["gen_ai.tool.name"].AsString() != "weather" || got["gen_ai.tool.call.id"].AsString() != "call-1" || got["gen_ai.operation.name"].AsString() != "execute_tool" {
		t.Errorf("attributes = %v", s.Attributes)
	}
	if got["error.type"].AsString() != "tool_error" || s.Status.Code != codes.Error {
		t.Errorf("error.type = %v, status = %v, want tool_error", got["error.type"], s.Status)
	}
	if _, ok := got["gen_ai.tool.call.arguments"]; ok {
		t.Error("tool arguments recorded without CaptureContent")
	}
	m := h.collect(t)["gen_ai.execute_tool.duration"]
	hist, ok := m.Data.(metricdata.Histogram[float64])
	if !ok || len(hist.DataPoints) != 1 || hist.DataPoints[0].Sum != 0.25 {
		t.Errorf("execute_tool.duration = %+v, want one point of 0.25s", m.Data)
	}
}

func TestTelemetry_toolArgumentsWithContent(t *testing.T) {
	h := newHarness(t, aiotel.CaptureContent())
	call := ai.ToolCall{ID: "c", Name: "weather", Input: json.RawMessage(`{"city":"Seoul"}`)}
	ctx := h.tel.StartTool(t.Context(), call)
	h.tel.FinishTool(ctx, call, ai.ToolResult{CallID: "c", Name: "weather", Output: ai.ToolOutput{JSON: json.RawMessage(`{"temp":21}`)}}, time.Millisecond, nil)
	got := attrs(h.span(t, "execute_tool weather"))
	if got["gen_ai.tool.call.arguments"].AsString() != `{"city":"Seoul"}` || got["gen_ai.tool.call.result"].AsString() != `{"temp":21}` {
		t.Errorf("tool content attributes = %v", got)
	}
}

func TestTelemetry_toolResultIsOptIn(t *testing.T) {
	h := newHarness(t)
	call := ai.ToolCall{ID: "c", Name: "weather"}
	ctx := h.tel.StartTool(t.Context(), call)
	h.tel.FinishTool(ctx, call, ai.ToolResult{Output: ai.ToolOutput{Text: "secret"}}, time.Millisecond, nil)
	if _, ok := attrs(h.span(t, "execute_tool weather"))["gen_ai.tool.call.result"]; ok {
		t.Error("tool result recorded without CaptureContent")
	}
}

var requestWithContent = ai.StepRequest{
	Instructions: []ai.Text{{Text: "be brief"}},
	Messages: []ai.Message{
		ai.User("weather?"),
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "c1", Name: "weather", Input: json.RawMessage(`{"city":"Seoul"}`)}}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "c1", Name: "weather", Output: ai.ToolOutput{Text: "rainy"}}}},
	},
}

func TestTelemetry_requestContentWithCapture(t *testing.T) {
	h := newHarness(t, aiotel.CaptureContent())
	withRequest := info
	withRequest.Request = &requestWithContent
	h.tel.FinishStep(h.tel.StartStep(t.Context(), withRequest), withRequest, ai.Step{}, nil)
	got := attrs(h.span(t, "chat gemini-3-pro"))
	if diff := cmp.Diff(`[{"type":"text","content":"be brief"}]`, got["gen_ai.system_instructions"].AsString()); diff != "" {
		t.Errorf("system instructions mismatch (-want +got):\n%s", diff)
	}
	want := `[{"role":"user","parts":[{"type":"text","content":"weather?"}]},` +
		`{"role":"assistant","parts":[{"type":"tool_call","id":"c1","name":"weather","arguments":{"city":"Seoul"}}]},` +
		`{"role":"tool","parts":[{"type":"tool_call_response","id":"c1","response":"rainy"}]}]`
	if diff := cmp.Diff(want, got["gen_ai.input.messages"].AsString()); diff != "" {
		t.Errorf("input messages mismatch (-want +got):\n%s", diff)
	}
}

func TestTelemetry_requestContentIsOptIn(t *testing.T) {
	h := newHarness(t)
	withRequest := info
	withRequest.Request = &requestWithContent
	h.tel.FinishStep(h.tel.StartStep(t.Context(), withRequest), withRequest, ai.Step{}, nil)
	got := attrs(h.span(t, "chat gemini-3-pro"))
	for _, k := range []string{"gen_ai.input.messages", "gen_ai.system_instructions"} {
		if _, ok := got[k]; ok {
			t.Errorf("attribute %s recorded without CaptureContent", k)
		}
	}
}

func TestTelemetry_invokeAgentParentsStepsAndTools(t *testing.T) {
	h := newHarness(t)
	ctx, end := h.tel.InvokeAgent(t.Context(), "support")
	stepCtx := h.tel.StartStep(ctx, info)
	h.tel.FinishStep(stepCtx, info, ai.Step{}, nil)
	call := ai.ToolCall{ID: "c", Name: "lookup"}
	toolCtx := h.tel.StartTool(ctx, call)
	h.tel.FinishTool(toolCtx, call, ai.ToolResult{}, time.Millisecond, nil)
	end(nil)
	end(errors.New("ignored second call"))
	agent := h.span(t, "invoke_agent support")
	if attrs(agent)["gen_ai.agent.name"].AsString() != "support" || agent.Status.Code != codes.Unset {
		t.Errorf("agent span = %+v", agent)
	}
	for _, name := range []string{"chat gemini-3-pro", "execute_tool lookup"} {
		if got := h.span(t, name).Parent.SpanID(); got != agent.SpanContext.SpanID() {
			t.Errorf("%s parent = %v, want the agent span %v", name, got, agent.SpanContext.SpanID())
		}
	}
}

func sumPoints(t *testing.T, m metricdata.Metrics) []metricdata.DataPoint[int64] {
	t.Helper()
	sum, ok := m.Data.(metricdata.Sum[int64])
	if !ok {
		t.Fatalf("metric %s data = %T, want Sum[int64]", m.Name, m.Data)
	}
	return sum.DataPoints
}

func TestTelemetry_metrics(t *testing.T) {
	h := newHarness(t)
	runStep(t, h, ai.Step{Usage: ai.Usage{Input: 100, Output: 40, CacheRead: 30, Reasoning: 10}}, nil)
	runStep(t, h, ai.Step{}, context.DeadlineExceeded)
	got := h.collect(t)
	for name, want := range map[string]int64{
		"gen_ai.client.inference.usage.input_tokens":            100,
		"gen_ai.client.inference.usage.output_tokens":           40,
		"gen_ai.client.inference.usage.cache_read.input_tokens": 30,
		"gen_ai.client.inference.usage.reasoning.output_tokens": 10,
	} {
		points := sumPoints(t, got[name])
		if len(points) != 1 || points[0].Value != want {
			t.Errorf("%s = %+v, want a single point of %d", name, points, want)
		}
	}
	if _, ok := got["gen_ai.client.inference.usage.cache_write.input_tokens"]; ok {
		t.Error("cache_write recorded although zero")
	}
	modality, _ := sumPoints(t, got["gen_ai.client.inference.usage.input_tokens"])[0].Attributes.Value("gen_ai.token.modality")
	if modality.AsString() != "unknown" {
		t.Errorf("token modality = %v, want unknown", modality)
	}
	duration, ok := got["gen_ai.client.inference.duration"].Data.(metricdata.Histogram[float64])
	if !ok || len(duration.DataPoints) != 2 {
		t.Fatalf("duration = %+v, want two series (ok and error)", got["gen_ai.client.inference.duration"].Data)
	}
	var failed int
	for _, p := range duration.DataPoints {
		if v, ok := p.Attributes.Value("error.type"); ok && v.AsString() == "timeout" {
			failed++
		}
	}
	if failed != 1 {
		t.Errorf("series with error.type=timeout = %d, want 1", failed)
	}
	operation, ok := got["gen_ai.client.inference.operation.input_tokens"].Data.(metricdata.Histogram[int64])
	if !ok || len(operation.DataPoints) != 2 {
		t.Errorf("operation.input_tokens = %+v, want two series", got["gen_ai.client.inference.operation.input_tokens"].Data)
	}
}

func TestNew_usesGlobalProviders(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(previous) })
	tel, err := aiotel.New()
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	runWith := tel.StartStep(t.Context(), info)
	tel.FinishStep(runWith, info, ai.Step{}, nil)
	if got := len(exporter.GetSpans()); got != 1 {
		t.Errorf("spans on the global provider = %d, want 1", got)
	}
}

func TestNew_defaultsRecordNothingWithoutSDK(t *testing.T) {
	tel, err := aiotel.New()
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := tel.StartTool(t.Context(), ai.ToolCall{Name: "x"})
	tel.FinishTool(ctx, ai.ToolCall{Name: "x"}, ai.ToolResult{}, time.Second, nil)
}
