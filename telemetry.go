package ai

import (
	"context"
	"time"
)

// Telemetry receives metadata by default. The context returned by StartStep and StartTool flows to the model call and the tool.
// StepInfo.Request and the ToolResult passed to FinishTool are shared with the loop, are read-only, and cost nothing to pass; an implementation decides whether to copy or record them.
type Telemetry interface {
	StartStep(ctx context.Context, info StepInfo) context.Context
	FinishStep(ctx context.Context, info StepInfo, step Step, err error)
	StartTool(ctx context.Context, call ToolCall) context.Context
	FinishTool(ctx context.Context, call ToolCall, result ToolResult, took time.Duration, err error)
}

type StepInfo struct {
	Provider string
	ModelID  string
	Number   int
	Request  *StepRequest
}
