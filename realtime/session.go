package realtime

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"
	"sync/atomic"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schema"
	"github.com/minodevss/ai-go/internal/toolmsg"
)

const (
	approvalTimedOut = "no approval was given"
)

type decision struct {
	approved bool
	reason   string
}

type pending struct {
	call    ai.ToolCall
	answer  chan decision
	request ApprovalRequest
}

// Session is safe for concurrent use. It ends when ctx given to Connect ends, when Close is called, or when the provider connection fails; Err then reports why.
type Session struct {
	conn    Conn
	cfg     Config
	tools   map[string]ai.Tool
	cancel  context.CancelFunc
	events  chan Event
	wg      sync.WaitGroup
	once    sync.Once
	closed  atomic.Bool
	tainted atomic.Bool
	userEnd atomic.Bool

	mu        sync.Mutex
	err       error
	consumed  bool
	misuse    error
	approvals map[string]*pending
	running   map[string]context.CancelCauseFunc
	closeErr  error
}

// Connect opens a session. ctx bounds its whole lifetime, not only the connection.
func Connect(ctx context.Context, model Model, cfg Config) (*Session, error) {
	if model == nil {
		return nil, errors.New("realtime: nil model")
	}
	cfg = cfg.clone()
	tools, err := indexTools(cfg.Tools)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	conn, err := model.Connect(ctx, &cfg)
	if err != nil {
		cancel()
		return nil, err
	}
	if conn == nil {
		cancel()
		return nil, fmt.Errorf("realtime: %s returned no connection", model.Provider())
	}
	s := &Session{
		conn:      conn,
		cfg:       cfg,
		tools:     tools,
		cancel:    cancel,
		events:    make(chan Event, orDefault(cfg.EventBuffer, defaultEventBuffer)),
		approvals: map[string]*pending{},
		running:   map[string]context.CancelCauseFunc{},
	}
	s.wg.Go(func() { s.pump(ctx) })
	return s, nil
}

func indexTools(tools []ai.Tool) (map[string]ai.Tool, error) {
	out := make(map[string]ai.Tool, len(tools))
	for _, t := range tools {
		if _, dup := out[t.Name]; dup {
			return nil, fmt.Errorf("%w: %q", ai.ErrDuplicateTool, t.Name)
		}
		out[t.Name] = t
	}
	return out, nil
}

func orDefault[T ~int | ~int64](v, def T) T {
	if v <= 0 {
		return def
	}
	return v
}

func (s *Session) SendAudio(ctx context.Context, pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	if err := s.checkInput(len(pcm)); err != nil {
		return err
	}
	return s.conn.SendAudio(ctx, pcm)
}

func (s *Session) SendText(ctx context.Context, text string) error {
	if err := s.checkInput(len(text)); err != nil {
		return err
	}
	return s.conn.SendText(ctx, text)
}

// Interrupt stops the model's current response and cancels the tool calls it started.
func (s *Session) Interrupt(ctx context.Context) error {
	if s.closed.Load() {
		return ErrClosed
	}
	return s.conn.Interrupt(ctx)
}

func (s *Session) checkInput(n int) error {
	if s.closed.Load() {
		return ErrClosed
	}
	if limit := orDefault(s.cfg.MaxInputBytes, defaultMaxInputBytes); n > limit {
		return fmt.Errorf("%w: %d bytes, limit %d", ErrInputTooLarge, n, limit)
	}
	return nil
}

// Approve answers an ApprovalRequest event. Each approval id works once; unknown or already answered ids fail with ai.ErrInvalidApproval. The tool still runs only with the input shown in the request, and its Approval policy is checked again.
func (s *Session) Approve(ctx context.Context, approvalID string, approved bool) error {
	s.mu.Lock()
	p, ok := s.approvals[approvalID]
	delete(s.approvals, approvalID)
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: unknown approval %q", ai.ErrInvalidApproval, approvalID)
	}
	select {
	case p.answer <- decision{approved: approved}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// SendToolResult answers a ToolCall of a tool declared without Execute.
func (s *Session) SendToolResult(ctx context.Context, callID string, out ai.ToolOutput, isError bool) error {
	if s.closed.Load() {
		return ErrClosed
	}
	return s.conn.SendToolResult(ctx, callID, out, isError)
}

// Events can be ranged once. It ends when the session ends; stopping the loop early does not close the session.
func (s *Session) Events() iter.Seq[Event] {
	return func(yield func(Event) bool) {
		s.mu.Lock()
		if s.consumed {
			s.misuse = ErrEventsConsumed
			s.mu.Unlock()
			return
		}
		s.consumed = true
		s.mu.Unlock()
		for ev := range s.events {
			if !yield(ev) {
				return
			}
		}
	}
}

// Err returns why the session ended: nil after Close or a clean provider shutdown.
func (s *Session) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.misuse != nil {
		return s.misuse
	}
	return s.err
}

// Close ends the session, cancels running tools and waits for them. It is safe to call more than once.
func (s *Session) Close() error {
	s.once.Do(func() {
		s.userEnd.Store(true)
		s.closed.Store(true)
		s.cancel()
		err := s.conn.Close()
		s.wg.Wait()
		s.mu.Lock()
		s.closeErr = err
		s.mu.Unlock()
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeErr
}

func (s *Session) pump(ctx context.Context) {
	defer close(s.events)
	var runs sync.WaitGroup
	defer func() {
		s.closed.Store(true)
		s.cancel()
		runs.Wait()
	}()
	for ev, err := range s.conn.Events() {
		if err != nil {
			s.endWith(err)
			return
		}
		if !s.dispatch(ctx, &runs, ev) {
			s.endWith(ctx.Err())
			return
		}
	}
	s.endWith(ctx.Err())
}

func (s *Session) endWith(err error) {
	if err != nil && !s.userEnd.Load() {
		s.setErr(err)
	}
}

func (s *Session) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}

func (s *Session) dispatch(ctx context.Context, runs *sync.WaitGroup, ev Event) bool {
	switch e := ev.(type) {
	case Interrupted:
		s.cancelTools(nil)
	case ToolCanceled:
		s.cancelTools(&e.CallID)
	case ToolCall:
		if !s.emit(ctx, ev) {
			return false
		}
		s.startTool(ctx, runs, e)
		return true
	default:
	}
	return s.emit(ctx, ev)
}

func (s *Session) emit(ctx context.Context, ev Event) bool {
	select {
	case s.events <- ev:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Session) cancelTools(callID *string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, cancel := range s.running {
		if callID == nil || *callID == id {
			cancel(ErrInterrupted)
		}
	}
	for id, p := range s.approvals {
		if callID == nil || *callID == p.call.ID {
			delete(s.approvals, id)
		}
	}
}

func (s *Session) startTool(ctx context.Context, runs *sync.WaitGroup, e ToolCall) {
	tool, known := s.tools[e.Name]
	if known && tool.Execute == nil {
		return
	}
	call := ai.ToolCall{ID: e.ID, Name: e.Name, Input: append(json.RawMessage(nil), e.Input...)}
	toolCtx, cancel := context.WithCancelCause(ctx)
	s.mu.Lock()
	_, duplicate := s.running[call.ID]
	full := len(s.running) >= orDefault(s.cfg.MaxInFlightTools, defaultMaxInFlightTools)
	if !full && !duplicate {
		s.running[call.ID] = cancel
	}
	s.mu.Unlock()
	switch {
	case duplicate:
		cancel(nil)
		runs.Go(func() {
			s.finishTool(ctx, call, errorOutput(fmt.Sprintf("tool call id %q is already running", call.ID)), true)
		})
	case !known:
		cancel(nil)
		runs.Go(func() { s.finishTool(ctx, call, errorOutput(fmt.Sprintf("tool %q does not exist", call.Name)), true) })
	case full:
		cancel(nil)
		runs.Go(func() { s.finishTool(ctx, call, errorOutput("too many tool calls in flight"), true) })
	default:
		runs.Go(func() {
			defer func() {
				s.mu.Lock()
				delete(s.running, call.ID)
				s.mu.Unlock()
				cancel(nil)
			}()
			s.runTool(toolCtx, tool, call)
		})
	}
}

func (s *Session) runTool(ctx context.Context, tool ai.Tool, call ai.ToolCall) {
	if err := validateInput(tool, call.Input); err != nil {
		s.finishTool(ctx, call, errorOutput(fmt.Sprintf("invalid input for tool %q: %v", tool.Name, err)), true)
		return
	}
	if tool.Approval != nil {
		if denied := s.gate(ctx, tool, call); denied != "" {
			if ctx.Err() == nil {
				s.finishTool(ctx, call, errorOutput(denied), true)
			}
			return
		}
	}
	out, err := s.execute(ctx, tool, call)
	if ctx.Err() != nil {
		return
	}
	if tool.Untrusted {
		s.tainted.Store(true)
	}
	if err != nil {
		s.finishTool(ctx, call, errorOutput(visibleError(ctx, err)), true)
		return
	}
	s.finishTool(ctx, call, out, false)
}

func (s *Session) gate(ctx context.Context, tool ai.Tool, call ai.ToolCall) string {
	verdict := policy(ctx, tool, ai.ApprovalInput{Call: call, Tainted: s.tainted.Load()})
	switch verdict.Decision {
	case ai.DecisionAllow:
		return ""
	case ai.DecisionDeny:
		return toolmsg.Denied(verdict.Reason)
	default:
	}
	answer, err := s.ask(ctx, call, verdict.Reason)
	if err != nil {
		if errors.Is(err, errTooManyPendingApprovals) {
			return err.Error()
		}
		return approvalTimedOut
	}
	if !answer.approved {
		return toolmsg.Denied(answer.reason)
	}
	if again := policy(ctx, tool, ai.ApprovalInput{Call: call, Tainted: s.tainted.Load()}); again.Decision == ai.DecisionDeny {
		return toolmsg.Denied(again.Reason)
	}
	return ""
}

func (s *Session) ask(ctx context.Context, call ai.ToolCall, reason string) (decision, error) {
	p := &pending{
		call:   call,
		answer: make(chan decision, 1),
		request: ApprovalRequest{
			ApprovalID: rand.Text(),
			CallID:     call.ID,
			Name:       call.Name,
			Input:      append(json.RawMessage(nil), call.Input...),
			Reason:     reason,
		},
	}
	s.mu.Lock()
	if len(s.approvals) >= orDefault(s.cfg.MaxPendingApprovals, defaultMaxPendingApprovals) {
		s.mu.Unlock()
		return decision{}, errTooManyPendingApprovals
	}
	s.approvals[p.request.ApprovalID] = p
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.approvals, p.request.ApprovalID)
		s.mu.Unlock()
	}()
	if !s.emit(ctx, p.request) {
		return decision{}, ctx.Err()
	}
	waitCtx, cancel := context.WithTimeout(ctx, orDefault(s.cfg.ApprovalTimeout, defaultApprovalTimeout))
	defer cancel()
	select {
	case d := <-p.answer:
		return d, nil
	case <-waitCtx.Done():
		return decision{}, waitCtx.Err()
	}
}

func (s *Session) execute(ctx context.Context, tool ai.Tool, call ai.ToolCall) (out ai.ToolOutput, err error) {
	if t := s.cfg.Telemetry; t != nil {
		ctx = t.StartTool(ctx, call)
		start := time.Now()
		defer func() {
			t.FinishTool(ctx, call, ai.ToolResult{CallID: call.ID, Name: call.Name, Output: out, IsError: err != nil}, time.Since(start), err)
		}()
	}
	if tool.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, tool.Timeout, context.DeadlineExceeded)
		defer cancel()
	}
	defer func() {
		if v := recover(); v != nil {
			err = fmt.Errorf("realtime: tool %q panicked: %v", tool.Name, v)
		}
	}()
	return tool.Execute(ctx, toolmsg.NormalizeInput(call.Input))
}

func (s *Session) finishTool(ctx context.Context, call ai.ToolCall, out ai.ToolOutput, isError bool) {
	if !s.emit(ctx, ToolResult{CallID: call.ID, Name: call.Name, Output: out, IsError: isError}) {
		return
	}
	if err := s.conn.SendToolResult(ctx, call.ID, out, isError); err != nil && ctx.Err() == nil {
		s.setErr(fmt.Errorf("realtime: send tool result: %w", err))
		s.cancel()
	}
}

func policy(ctx context.Context, tool ai.Tool, in ai.ApprovalInput) (a ai.Approval) {
	defer func() {
		if v := recover(); v != nil {
			a = ai.Approval{}
		}
	}()
	return tool.Approval(ctx, in)
}

func validateInput(tool ai.Tool, input json.RawMessage) error {
	if len(tool.InputSchema) == 0 {
		return nil
	}
	v, err := schema.Compile(tool.InputSchema)
	if err != nil {
		return err
	}
	return v.Validate(toolmsg.NormalizeInput(input))
}

func errorOutput(text string) ai.ToolOutput { return ai.ToolOutput{Text: text} }

func visibleError(ctx context.Context, err error) string {
	if toolErr, ok := errors.AsType[*ai.ToolError](err); ok {
		return toolErr.Message
	}
	if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
		return toolmsg.TimedOut
	}
	return toolmsg.Failure
}
