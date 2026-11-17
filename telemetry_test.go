package ai_test

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/minodevss/ai-go"
)

type recorder struct {
	mu     sync.Mutex
	events []string
	inputs []int
	output []string
}

func (r *recorder) add(s string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, s)
}

func (r *recorder) StartStep(ctx context.Context, info ai.StepInfo) context.Context {
	r.add("step-start:" + strconv.Itoa(info.Number))
	r.mu.Lock()
	defer r.mu.Unlock()
	if info.Request != nil {
		r.inputs = append(r.inputs, len(info.Request.Messages))
	}
	return ctx
}

func (r *recorder) FinishStep(_ context.Context, info ai.StepInfo, _ ai.Step, _ error) {
	r.add("step-end:" + strconv.Itoa(info.Number))
}

func (r *recorder) StartTool(ctx context.Context, call ai.ToolCall) context.Context {
	r.add("tool-start:" + call.Name)
	return ctx
}

func (r *recorder) FinishTool(_ context.Context, call ai.ToolCall, result ai.ToolResult, _ time.Duration, _ error) {
	r.add("tool-end:" + call.Name)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.output = append(r.output, string(result.Output.JSON))
}
