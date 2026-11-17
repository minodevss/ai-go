// Package aiotel reports ai.Telemetry events as OpenTelemetry spans and metrics using the GenAI semantic conventions, which have Development status and may change.
package aiotel

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"

	"github.com/minodevss/ai-go"
)

const scope = "github.com/minodevss/ai-go/aiotel"

const (
	operationChat        = "chat"
	operationExecuteTool = "execute_tool"
	operationInvokeAgent = "invoke_agent"
	modalityUnknown      = "unknown"
)

type options struct {
	tracers trace.TracerProvider
	meters  metric.MeterProvider
	content bool
}

type Option func(*options)

// WithTracerProvider replaces the global TracerProvider.
func WithTracerProvider(tp trace.TracerProvider) Option {
	return func(o *options) {
		if tp != nil {
			o.tracers = tp
		}
	}
}

// WithMeterProvider replaces the global MeterProvider.
func WithMeterProvider(mp metric.MeterProvider) Option {
	return func(o *options) {
		if mp != nil {
			o.meters = mp
		}
	}
}

// CaptureContent records instructions, input and output messages, tool arguments and tool results on spans. Without it no prompt, output or tool data leaves the process.
func CaptureContent() Option {
	return func(o *options) { o.content = true }
}

// Telemetry implements ai.Telemetry. It is safe for concurrent use.
type Telemetry struct {
	tracer  trace.Tracer
	content bool

	inferenceDuration metric.Float64Histogram
	toolDuration      metric.Float64Histogram
	inputTokens       metric.Int64Counter
	outputTokens      metric.Int64Counter
	cacheRead         metric.Int64Counter
	cacheWrite        metric.Int64Counter
	reasoningTokens   metric.Int64Counter
	operationInput    metric.Int64Histogram
	operationOutput   metric.Int64Histogram
}

var _ ai.Telemetry = (*Telemetry)(nil)

// New uses the global providers unless options inject others.
func New(opts ...Option) (*Telemetry, error) {
	o := options{tracers: otel.GetTracerProvider(), meters: otel.GetMeterProvider()}
	for _, opt := range opts {
		opt(&o)
	}
	meter := o.meters.Meter(scope)
	t := &Telemetry{tracer: o.tracers.Tracer(scope), content: o.content}
	var errs []error
	histogram := func(name, unit, desc string, buckets []float64) metric.Float64Histogram {
		h, err := meter.Float64Histogram(name, metric.WithUnit(unit), metric.WithDescription(desc), metric.WithExplicitBucketBoundaries(buckets...))
		errs = append(errs, err)
		return h
	}
	intHistogram := func(name, desc string) metric.Int64Histogram {
		h, err := meter.Int64Histogram(name, metric.WithUnit("{token}"), metric.WithDescription(desc), metric.WithExplicitBucketBoundaries(tokenBuckets()...))
		errs = append(errs, err)
		return h
	}
	counter := func(name, desc string) metric.Int64Counter {
		c, err := meter.Int64Counter(name, metric.WithUnit("{token}"), metric.WithDescription(desc))
		errs = append(errs, err)
		return c
	}
	t.inferenceDuration = histogram("gen_ai.client.inference.duration", "s", "GenAI client inference operation duration.", durationBuckets())
	t.toolDuration = histogram("gen_ai.execute_tool.duration", "s", "The duration of a single tool execution.", durationBuckets())
	t.inputTokens = counter("gen_ai.client.inference.usage.input_tokens", "The number of input tokens used, including cached tokens.")
	t.outputTokens = counter("gen_ai.client.inference.usage.output_tokens", "The number of output tokens used, including reasoning tokens.")
	t.cacheRead = counter("gen_ai.client.inference.usage.cache_read.input_tokens", "The number of input tokens served from a provider-managed cache.")
	t.cacheWrite = counter("gen_ai.client.inference.usage.cache_write.input_tokens", "The number of input tokens written to a provider-managed cache.")
	t.reasoningTokens = counter("gen_ai.client.inference.usage.reasoning.output_tokens", "The number of output tokens used for reasoning.")
	t.operationInput = intHistogram("gen_ai.client.inference.operation.input_tokens", "The number of input tokens used per inference operation.")
	t.operationOutput = intHistogram("gen_ai.client.inference.operation.output_tokens", "The number of output tokens used per inference operation.")
	if err := errors.Join(errs...); err != nil {
		return nil, errors.New("aiotel: create instruments: " + err.Error())
	}
	return t, nil
}

type stepKey struct{}

type started struct{ at time.Time }

// InvokeAgent opens an invoke_agent span around a local agent run; spans of steps and tools started from the returned context become its children. Call the returned function once with the run's error.
func (t *Telemetry) InvokeAgent(ctx context.Context, name string) (context.Context, func(error)) {
	spanName := operationInvokeAgent
	attrs := []attribute.KeyValue{attribute.String("gen_ai.operation.name", operationInvokeAgent)}
	if name != "" {
		spanName += " " + name
		attrs = append(attrs, attribute.String("gen_ai.agent.name", name))
	}
	ctx, span := t.tracer.Start(ctx, spanName, trace.WithSpanKind(trace.SpanKindInternal), trace.WithAttributes(attrs...))
	var once sync.Once
	return ctx, func(err error) {
		once.Do(func() {
			t.finishSpan(span, err)
			span.End()
		})
	}
}

func (t *Telemetry) StartStep(ctx context.Context, info ai.StepInfo) context.Context {
	provider := providerName(info.Provider)
	name := operationChat
	if info.ModelID != "" {
		name += " " + info.ModelID
	}
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", operationChat),
		attribute.String("gen_ai.provider.name", provider),
		attribute.String("gen_ai.request.model", info.ModelID),
		attribute.Int("ai.step.number", info.Number),
	}
	if t.content && info.Request != nil {
		attrs = append(attrs, requestContent(*info.Request)...)
	}
	ctx, _ = t.tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	return context.WithValue(ctx, stepKey{}, started{at: time.Now()})
}

func (t *Telemetry) FinishStep(ctx context.Context, info ai.StepInfo, step ai.Step, err error) {
	span := trace.SpanFromContext(ctx)
	took := time.Duration(0)
	if s, ok := ctx.Value(stepKey{}).(started); ok {
		took = time.Since(s.at)
	}
	base := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", operationChat),
		attribute.String("gen_ai.provider.name", providerName(info.Provider)),
		attribute.String("gen_ai.request.model", info.ModelID),
	}
	if err != nil {
		base = append(base, attribute.String("error.type", errorType(err)))
	}
	span.SetAttributes(usageAttributes(step.Usage)...)
	if step.FinishReason != "" {
		span.SetAttributes(attribute.StringSlice("gen_ai.response.finish_reasons", []string{string(step.FinishReason)}))
	}
	if t.content && len(step.Content) > 0 {
		span.SetAttributes(attribute.String("gen_ai.output.messages", outputMessages(step)))
	}
	t.finishSpan(span, err)
	span.End()
	m := metric.WithAttributes(base...)
	t.inferenceDuration.Record(ctx, took.Seconds(), m)
	t.operationInput.Record(ctx, int64(step.Usage.Input), m)
	t.operationOutput.Record(ctx, int64(step.Usage.Output), m)
	unknown := metric.WithAttributes(append(base, attribute.String("gen_ai.token.modality", modalityUnknown))...)
	addPositive(ctx, t.inputTokens, step.Usage.Input, unknown)
	addPositive(ctx, t.outputTokens, step.Usage.Output, unknown)
	addPositive(ctx, t.cacheRead, step.Usage.CacheRead, unknown)
	addPositive(ctx, t.cacheWrite, step.Usage.CacheWrite, unknown)
	addPositive(ctx, t.reasoningTokens, step.Usage.Reasoning, m)
}

func (t *Telemetry) StartTool(ctx context.Context, call ai.ToolCall) context.Context {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", operationExecuteTool),
		attribute.String("gen_ai.tool.name", call.Name),
		attribute.String("gen_ai.tool.call.id", call.ID),
		attribute.String("gen_ai.tool.type", "function"),
	}
	if t.content && len(call.Input) > 0 {
		attrs = append(attrs, attribute.String("gen_ai.tool.call.arguments", string(call.Input)))
	}
	ctx, _ = t.tracer.Start(ctx, operationExecuteTool+" "+call.Name, trace.WithSpanKind(trace.SpanKindInternal), trace.WithAttributes(attrs...))
	return ctx
}

func (t *Telemetry) FinishTool(ctx context.Context, call ai.ToolCall, result ai.ToolResult, took time.Duration, err error) {
	span := trace.SpanFromContext(ctx)
	if t.content && err == nil {
		span.SetAttributes(attribute.String("gen_ai.tool.call.result", toolResponse(result.Output)))
	}
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", operationExecuteTool),
		attribute.String("gen_ai.tool.name", call.Name),
	}
	if err != nil {
		attrs = append(attrs, attribute.String("error.type", errorType(err)))
	}
	t.finishSpan(span, err)
	span.End()
	t.toolDuration.Record(ctx, took.Seconds(), metric.WithAttributes(attrs...))
}

func (t *Telemetry) finishSpan(span trace.Span, err error) {
	if err == nil {
		return
	}
	kind := errorType(err)
	span.SetAttributes(attribute.String("error.type", kind))
	span.SetStatus(codes.Error, kind)
	if t.content {
		span.RecordError(err)
	}
}

func addPositive(ctx context.Context, c metric.Int64Counter, n int, opt metric.AddOption) {
	if n > 0 {
		c.Add(ctx, int64(n), opt)
	}
}

func usageAttributes(u ai.Usage) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.Int("gen_ai.usage.input_tokens", u.Input),
		attribute.Int("gen_ai.usage.output_tokens", u.Output),
	}
	if u.CacheRead > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.cache_read.input_tokens", u.CacheRead))
	}
	if u.CacheWrite > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.cache_write.input_tokens", u.CacheWrite))
	}
	if u.Reasoning > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.reasoning.output_tokens", u.Reasoning))
	}
	return attrs
}

func providerName(name string) string {
	switch name {
	case "google":
		return "gcp.gemini"
	case "vertex":
		return "gcp.vertex_ai"
	case "bedrock":
		return "aws.bedrock"
	case "azure":
		return "azure.ai.openai"
	case "mistral":
		return "mistral_ai"
	case "xai":
		return "x_ai"
	}
	return name
}

func durationBuckets() []float64 {
	return []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
}

func tokenBuckets() []float64 {
	return []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
}

func errorType(err error) string {
	var api *ai.APIError
	var tool *ai.ToolError
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.As(err, &api):
		if api.Code != "" {
			return api.Code
		}
		if api.StatusCode != 0 {
			return strconv.Itoa(api.StatusCode)
		}
		return "api_error"
	case errors.As(err, &tool):
		return "tool_error"
	default:
		return "_OTHER"
	}
}

type messagePart struct {
	Type      string          `json:"type"`
	Content   string          `json:"content,omitzero"`
	ID        string          `json:"id,omitzero"`
	Name      string          `json:"name,omitzero"`
	Arguments json.RawMessage `json:"arguments,omitzero"`
	Response  json.RawMessage `json:"response,omitzero"`
	MimeType  string          `json:"mime_type,omitzero"`
	URI       string          `json:"uri,omitzero"`
}

type outputMessage struct {
	Role         string        `json:"role"`
	Parts        []messagePart `json:"parts"`
	FinishReason string        `json:"finish_reason,omitzero"`
}

func outputMessages(step ai.Step) string {
	return marshalOrEmpty([]outputMessage{{Role: "assistant", Parts: messageParts(step.Content), FinishReason: string(step.FinishReason)}})
}

func requestContent(req ai.StepRequest) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	if len(req.Instructions) > 0 {
		system := make([]messagePart, len(req.Instructions))
		for i, in := range req.Instructions {
			system[i] = messagePart{Type: "text", Content: in.Text}
		}
		attrs = append(attrs, attribute.String("gen_ai.system_instructions", marshalOrEmpty(system)))
	}
	if len(req.Messages) > 0 {
		messages := make([]outputMessage, len(req.Messages))
		for i, m := range req.Messages {
			messages[i] = outputMessage{Role: string(m.Role), Parts: messageParts(m.Content)}
		}
		attrs = append(attrs, attribute.String("gen_ai.input.messages", marshalOrEmpty(messages)))
	}
	return attrs
}

func messageParts(parts ai.Parts) []messagePart {
	var out []messagePart
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, messagePart{Type: "text", Content: v.Text})
		case ai.Reasoning:
			out = append(out, messagePart{Type: "reasoning", Content: v.Text})
		case ai.ToolCall:
			out = append(out, messagePart{Type: "tool_call", ID: v.ID, Name: v.Name, Arguments: v.Input})
		case ai.ToolResult:
			out = append(out, messagePart{Type: "tool_call_response", ID: v.CallID, Response: json.RawMessage(toolResponse(v.Output))})
		case ai.File:
			out = append(out, filePart(v))
		default:
		}
	}
	return out
}

func filePart(f ai.File) messagePart {
	if f.URL != "" {
		return messagePart{Type: "uri", URI: f.URL, MimeType: f.MediaType}
	}
	return messagePart{Type: "file", MimeType: f.MediaType}
}

func toolResponse(out ai.ToolOutput) string {
	if len(out.JSON) > 0 {
		return string(out.JSON)
	}
	var text strings.Builder
	text.WriteString(out.Text)
	for _, p := range out.Parts {
		if t, ok := p.(ai.Text); ok {
			text.WriteString(t.Text)
		}
	}
	return marshalOrEmpty(text.String())
}

func marshalOrEmpty(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		return "[]"
	}
	return string(raw)
}
