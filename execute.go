package ai

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/minodevss/ai-go/internal/schema"
	"github.com/minodevss/ai-go/internal/toolmsg"
)

type callItem struct {
	call     ToolCall
	msgIdx   int
	approved bool
	denied   bool
	reason   string
}

type handled struct {
	results  []ToolResult
	failures []ToolFailed
	requests []ToolApprovalRequest
	client   int
	ran      int

	unfinished []ToolCall
}

type slot struct {
	result  *ToolResult
	failure error
	request *ToolApprovalRequest
	run     *Tool
	ran     bool
}

func (r *runner) handleCalls(ctx context.Context, tools []Tool, items []callItem) (handled, error) {
	var h handled
	slots := make([]slot, len(items))
	tainted := r.tainted(tools)
	for i, it := range items {
		switch {
		case it.denied:
			slots[i].result = errorResult(it.call, toolmsg.Denied(it.reason))
		default:
			h.client += r.plan(ctx, tools, it, tainted, &slots[i])
		}
	}
	err := r.runPlanned(ctx, items, slots)
	for i, s := range slots {
		if s.ran && (s.failure == nil || err == nil || !errors.Is(s.failure, err)) {
			h.ran++
		}
		if s.run != nil && s.result == nil {
			h.unfinished = append(h.unfinished, items[i].call)
		}
		if s.failure != nil {
			h.failures = append(h.failures, ToolFailed{CallID: items[i].call.ID, Name: items[i].call.Name, Err: s.failure})
		}
		if s.result != nil {
			h.results = append(h.results, *s.result)
		}
		if s.request != nil {
			h.requests = append(h.requests, *s.request)
		}
	}
	return h, err
}

func (r *runner) plan(ctx context.Context, tools []Tool, it callItem, tainted bool, s *slot) int {
	call := it.call
	idx := slices.IndexFunc(tools, func(t Tool) bool { return t.Name == call.Name })
	if idx < 0 {
		err := &NoSuchToolError{Name: call.Name, Available: toolNames(tools)}
		s.failure = err
		s.result = errorResult(call, noSuchToolText(err))
		return 0
	}
	tool := tools[idx]
	if tool.Execute == nil {
		return 1
	}
	if err := r.validateInput(tool, call.Input); err != nil {
		s.failure = err
		s.result = errorResult(call, fmt.Sprintf("invalid input for tool %q: %v", tool.Name, err))
		return 0
	}
	if tool.Approval == nil {
		s.run = &tool
		return 0
	}
	approval := safePolicy(ctx, tool, ApprovalInput{Call: call, Tainted: tainted})
	switch {
	case approval.Decision == DecisionDeny:
		s.result = errorResult(call, toolmsg.Denied(approval.Reason))
	case approval.Decision == DecisionAllow, it.approved:
		s.run = &tool
	default:
		s.request = &ToolApprovalRequest{
			ApprovalID: rand.Text(),
			CallID:     call.ID,
			Reason:     approval.Reason,
			InputHash:  inputHash(call.Name, call.Input),
		}
	}
	return 0
}

func (r *runner) runPlanned(ctx context.Context, items []callItem, slots []slot) error {
	limit := r.req.MaxToolConcurrency
	if limit <= 0 {
		limit = defaultToolLimit
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
launch:
	for i := range slots {
		if slots[i].run == nil {
			continue
		}
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			break launch
		}
		wg.Go(func() {
			defer func() { <-sem }()
			slots[i].ran = true
			slots[i].result, slots[i].failure = r.execTool(ctx, *slots[i].run, items[i].call)
		})
	}
	wg.Wait()
	return ctx.Err()
}

func (r *runner) execTool(ctx context.Context, tool Tool, call ToolCall) (*ToolResult, error) {
	if r.req.Telemetry != nil {
		ctx = r.req.Telemetry.StartTool(ctx, call)
	}
	start := time.Now()
	out, err := r.invoke(ctx, tool, call)
	result := &ToolResult{CallID: call.ID, Name: call.Name, Output: out}
	if err != nil {
		result = errorResult(call, visibleToolError(err))
	}
	if r.req.Telemetry != nil {
		r.req.Telemetry.FinishTool(ctx, call, *result, time.Since(start), err)
	}
	return result, err
}

func (r *runner) invoke(ctx context.Context, tool Tool, call ToolCall) (out ToolOutput, err error) {
	timeout := tool.Timeout
	if timeout <= 0 {
		timeout = r.req.Timeouts.Tool
	}
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, timeout, &timeoutError{msg: toolmsg.TimedOut})
		defer cancel()
	}
	defer func() {
		if v := recover(); v != nil {
			err = fmt.Errorf("ai: tool %q panicked: %v", tool.Name, v)
		}
	}()
	out, err = tool.Execute(ctx, toolmsg.NormalizeInput(call.Input))
	if err != nil {
		err = mapTimeout(ctx, err)
	}
	return out, err
}

func visibleToolError(err error) string {
	if toolErr, ok := errors.AsType[*ToolError](err); ok {
		return toolErr.Message
	}
	if te, ok := errors.AsType[*timeoutError](err); ok && te.msg == toolmsg.TimedOut {
		return toolmsg.TimedOut
	}
	return toolmsg.Failure
}

func safePolicy(ctx context.Context, tool Tool, in ApprovalInput) (a Approval) {
	defer func() {
		if v := recover(); v != nil {
			a = Approval{}
		}
	}()
	return tool.Approval(ctx, in)
}

func (r *runner) validateInput(tool Tool, input json.RawMessage) error {
	if len(tool.InputSchema) == 0 {
		return nil
	}
	key := string(tool.InputSchema)
	v, ok := r.validators[key]
	if !ok {
		var err error
		v, err = schema.Compile(tool.InputSchema)
		if err != nil {
			return err
		}
		r.validators[key] = v
	}
	return v.Validate(toolmsg.NormalizeInput(input))
}

func (r *runner) tainted(tools []Tool) bool {
	untrusted := map[string]bool{}
	for _, t := range tools {
		if t.Untrusted {
			untrusted[t.Name] = true
		}
	}
	if len(untrusted) == 0 {
		return false
	}
	for _, m := range r.messages {
		for _, p := range m.Content {
			if res, ok := p.(ToolResult); ok && untrusted[res.Name] {
				return true
			}
		}
	}
	return false
}

func errorResult(call ToolCall, text string) *ToolResult {
	return &ToolResult{CallID: call.ID, Name: call.Name, Output: ToolOutput{Text: text}, IsError: true}
}

func noSuchToolText(err *NoSuchToolError) string {
	if len(err.Available) == 0 {
		return fmt.Sprintf("tool %q does not exist; no tools are available", err.Name)
	}
	return fmt.Sprintf("tool %q does not exist; available tools: %s", err.Name, strings.Join(err.Available, ", "))
}

func toolNames(tools []Tool) []string {
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	return names
}

func inputHash(name string, input json.RawMessage) [32]byte {
	var compact bytes.Buffer
	raw := toolmsg.NormalizeInput(input)
	if err := json.Compact(&compact, raw); err != nil {
		compact.Reset()
		compact.Write(raw)
	}
	return sha256.Sum256(slices.Concat([]byte(name), []byte{0}, compact.Bytes()))
}
