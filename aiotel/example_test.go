package aiotel_test

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aiotel"
)

func newTelemetry(opts ...aiotel.Option) (*aiotel.Telemetry, *tracetest.InMemoryExporter) {
	exporter := tracetest.NewInMemoryExporter()
	opts = append(opts, aiotel.WithTracerProvider(trace.NewTracerProvider(trace.WithSyncer(exporter))))
	tel, err := aiotel.New(opts...)
	if err != nil {
		panic(err)
	}
	return tel, exporter
}

func ExampleNew() {
	tel, exporter := newTelemetry()
	info := ai.StepInfo{Provider: "openai", ModelID: "gpt-5", Number: 1}
	ctx := tel.StartStep(context.Background(), info)
	tel.FinishStep(ctx, info, ai.Step{}, nil)
	fmt.Println(exporter.GetSpans()[0].Name)
	// Output: chat gpt-5
}

func ExampleWithTracerProvider() {
	exporter := tracetest.NewInMemoryExporter()
	tel, err := aiotel.New(aiotel.WithTracerProvider(trace.NewTracerProvider(trace.WithSyncer(exporter))))
	if err != nil {
		fmt.Println(err)
		return
	}
	_, end := tel.InvokeAgent(context.Background(), "planner")
	end(nil)
	fmt.Println(exporter.GetSpans()[0].Name)
	// Output: invoke_agent planner
}

func ExampleWithMeterProvider() {
	reader := sdkmetric.NewManualReader()
	tel, err := aiotel.New(aiotel.WithMeterProvider(sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))))
	if err != nil {
		fmt.Println(err)
		return
	}
	call := ai.ToolCall{ID: "c1", Name: "search"}
	ctx := tel.StartTool(context.Background(), call)
	tel.FinishTool(ctx, call, ai.ToolResult{}, 120*time.Millisecond, nil)
	var data metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &data); err != nil {
		fmt.Println(err)
		return
	}
	for _, scope := range data.ScopeMetrics {
		for _, m := range scope.Metrics {
			fmt.Println(m.Name)
		}
	}
	// Output: gen_ai.execute_tool.duration
}

func ExampleCaptureContent() {
	tel, exporter := newTelemetry(aiotel.CaptureContent())
	call := ai.ToolCall{ID: "c1", Name: "search", Input: json.RawMessage(`{"q":"go"}`)}
	ctx := tel.StartTool(context.Background(), call)
	tel.FinishTool(ctx, call, ai.ToolResult{}, time.Millisecond, nil)
	for _, attr := range exporter.GetSpans()[0].Attributes {
		if attr.Key == "gen_ai.tool.call.arguments" {
			fmt.Println(attr.Value.AsString())
		}
	}
	// Output: {"q":"go"}
}

func ExampleTelemetry_InvokeAgent() {
	tel, exporter := newTelemetry()
	ctx, end := tel.InvokeAgent(context.Background(), "planner")
	info := ai.StepInfo{Provider: "openai", ModelID: "gpt-5", Number: 1}
	stepCtx := tel.StartStep(ctx, info)
	tel.FinishStep(stepCtx, info, ai.Step{}, nil)
	end(nil)
	spans := exporter.GetSpans()
	fmt.Println(spans[0].Name, spans[0].Parent.SpanID() == spans[1].SpanContext.SpanID())
	// Output: chat gpt-5 true
}

func ExampleTelemetry_StartStep() {
	tel, exporter := newTelemetry()
	info := ai.StepInfo{Provider: "anthropic", ModelID: "claude-sonnet", Number: 1}
	ctx := tel.StartStep(context.Background(), info)
	tel.FinishStep(ctx, info, ai.Step{Usage: ai.Usage{Input: 10}}, nil)
	fmt.Println(exporter.GetSpans()[0].Name)
	// Output: chat claude-sonnet
}

func ExampleTelemetry_FinishStep() {
	tel, exporter := newTelemetry()
	info := ai.StepInfo{Provider: "openai", ModelID: "gpt-5", Number: 1}
	ctx := tel.StartStep(context.Background(), info)
	tel.FinishStep(ctx, info, ai.Step{}, ai.ErrRateLimit)
	fmt.Println(exporter.GetSpans()[0].Status.Code)
	// Output: Error
}

func ExampleTelemetry_StartTool() {
	tel, exporter := newTelemetry()
	call := ai.ToolCall{ID: "c1", Name: "search"}
	ctx := tel.StartTool(context.Background(), call)
	tel.FinishTool(ctx, call, ai.ToolResult{}, time.Millisecond, nil)
	fmt.Println(exporter.GetSpans()[0].Name)
	// Output: execute_tool search
}

func ExampleTelemetry_FinishTool() {
	tel, exporter := newTelemetry()
	call := ai.ToolCall{ID: "c1", Name: "search"}
	ctx := tel.StartTool(context.Background(), call)
	tel.FinishTool(ctx, call, ai.ToolResult{}, time.Millisecond, context.DeadlineExceeded)
	fmt.Println(exporter.GetSpans()[0].Status.Code)
	// Output: Error
}
