package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/minodevss/ai-go/internal/schema"
	"github.com/minodevss/ai-go/internal/toolmsg"
)

type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Strict      bool
	// Execute is nil for client-side tools; the loop then ends and returns the call.
	Execute  func(ctx context.Context, input json.RawMessage) (ToolOutput, error)
	Approval func(ctx context.Context, in ApprovalInput) Approval
	// Untrusted marks tools whose output may carry injected instructions.
	Untrusted bool
	Timeout   time.Duration
	Provider  ProviderData
}

type ApprovalInput struct {
	Call    ToolCall
	Tainted bool
}

type Decision int

const (
	DecisionAsk Decision = iota
	DecisionAllow
	DecisionDeny
)

// Approval's zero value asks the user.
type Approval struct {
	Decision Decision
	Reason   string
}

func Allow() Approval { return Approval{Decision: DecisionAllow} }

func Deny(reason string) Approval { return Approval{Decision: DecisionDeny, Reason: reason} }

func AskUser(reason string) Approval { return Approval{Decision: DecisionAsk, Reason: reason} }

func AlwaysAsk(context.Context, ApprovalInput) Approval { return Approval{} }

// NewTool panics if In has no JSON schema.
func NewTool[In, Out any](name, description string, fn func(context.Context, In) (Out, error)) Tool {
	inputSchema, err := schema.For[In]()
	if err != nil {
		panic(fmt.Sprintf("ai: NewTool %q: %v", name, err))
	}
	return Tool{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
		Execute: func(ctx context.Context, input json.RawMessage) (ToolOutput, error) {
			var in In
			if err := json.Unmarshal(toolmsg.NormalizeInput(input), &in); err != nil {
				return ToolOutput{}, ToolErrorf("invalid input: %v", err)
			}
			out, err := fn(ctx, in)
			if err != nil {
				return ToolOutput{}, err
			}
			return toToolOutput(out)
		},
	}
}

// NewClientTool declares a tool the caller executes; the loop ends when the model calls it.
func NewClientTool[In any](name, description string) Tool {
	t := NewTool(name, description, func(context.Context, In) (struct{}, error) { return struct{}{}, nil })
	t.Execute = nil
	return t
}

func (t Tool) WithApproval(f func(context.Context, ApprovalInput) Approval) Tool {
	t.Approval = f
	return t
}

func (t Tool) WithTimeout(d time.Duration) Tool {
	t.Timeout = d
	return t
}

func toToolOutput(out any) (ToolOutput, error) {
	switch v := out.(type) {
	case ToolOutput:
		return v, nil
	case *ToolOutput:
		if v == nil {
			return ToolOutput{JSON: json.RawMessage("null")}, nil
		}
		return *v, nil
	default:
		raw, err := json.Marshal(out)
		if err != nil {
			return ToolOutput{}, fmt.Errorf("ai: encode tool output: %w", err)
		}
		return ToolOutput{JSON: raw}, nil
	}
}
