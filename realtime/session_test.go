package realtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/realtime"
)

type harness struct {
	t      *testing.T
	model  *aitest.RealtimeModel
	sess   *realtime.Session
	mu     sync.Mutex
	events []realtime.Event
}

func newHarness(t *testing.T, cfg realtime.Config) *harness {
	t.Helper()
	h := &harness{t: t, model: aitest.NewRealtimeModel()}
	sess, err := realtime.Connect(t.Context(), h.model, cfg)
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	h.sess = sess
	t.Cleanup(func() {
		if err := sess.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	go func() {
		for ev := range sess.Events() {
			h.mu.Lock()
			h.events = append(h.events, ev)
			h.mu.Unlock()
		}
	}()
	return h
}

func (h *harness) emit(ev realtime.Event) {
	h.t.Helper()
	if err := h.model.Emit(h.t.Context(), ev); err != nil {
		h.t.Fatalf("Emit() error = %v", err)
	}
	synctest.Wait()
}

func (h *harness) got() []realtime.Event {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]realtime.Event(nil), h.events...)
}

func (h *harness) approvalRequest() realtime.ApprovalRequest {
	h.t.Helper()
	for _, ev := range h.got() {
		if req, ok := ev.(realtime.ApprovalRequest); ok {
			return req
		}
	}
	h.t.Fatalf("no ApprovalRequest in %+v", h.got())
	return realtime.ApprovalRequest{}
}

func (h *harness) toolResults() []realtime.ToolResult {
	var out []realtime.ToolResult
	for _, ev := range h.got() {
		if r, ok := ev.(realtime.ToolResult); ok {
			out = append(out, r)
		}
	}
	return out
}

func (h *harness) sentResults() []aitest.RealtimeInput {
	var out []aitest.RealtimeInput
	for _, in := range h.model.Sent() {
		if in.Kind == "tool-result" {
			out = append(out, in)
		}
	}
	return out
}

type moveIn struct {
	Dir string `json:"dir" jsonschema:"direction"`
}

func moveTool(calls *int) ai.Tool {
	return ai.NewTool("move", "moves", func(_ context.Context, in moveIn) (string, error) {
		*calls++
		return "moved " + in.Dir, nil
	})
}

func call(id, name, input string) realtime.ToolCall {
	return realtime.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

func TestSession_passesEventsAndInputs(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{Instructions: "be nice", Voice: "marin", Resume: "handle-1"})
		want := []realtime.Event{
			realtime.AudioDelta{Data: []byte{1, 2}},
			realtime.TextDelta{Text: "hi"},
			realtime.Transcript{Role: ai.RoleUser, Text: "hello", Final: true},
			realtime.Interrupted{},
			realtime.TurnEnd{Usage: ai.Usage{Input: 1, Output: 2}, FinishReason: ai.FinishStop},
			realtime.GoAway{After: time.Minute},
			realtime.Resumable{Handle: "h2"},
		}
		for _, ev := range want {
			h.emit(ev)
		}
		if diff := cmp.Diff(want, h.got()); diff != "" {
			t.Errorf("events mismatch (-want +got):\n%s", diff)
		}
		ctx := t.Context()
		for _, err := range []error{h.sess.SendAudio(ctx, []byte("pcm")), h.sess.SendText(ctx, "hey"), h.sess.Interrupt(ctx)} {
			if err != nil {
				t.Fatalf("send error = %v", err)
			}
		}
		wantSent := []aitest.RealtimeInput{{Kind: "audio", Audio: []byte("pcm")}, {Kind: "text", Text: "hey"}, {Kind: "interrupt"}}
		if diff := cmp.Diff(wantSent, h.model.Sent()); diff != "" {
			t.Errorf("sent mismatch (-want +got):\n%s", diff)
		}
		if cfg := h.model.Config(); cfg.Instructions != "be nice" || cfg.Voice != "marin" || cfg.Resume != "handle-1" {
			t.Errorf("provider config = %+v, want the session config", cfg)
		}
		if err := h.sess.SendAudio(ctx, nil); err != nil || len(h.model.Sent()) != 3 {
			t.Errorf("empty SendAudio() = %v with %d sent, want a no-op", err, len(h.model.Sent()))
		}
	})
}

func TestSession_runsServerTools(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{moveTool(&calls)}})
		h.emit(call("c1", "move", `{"dir":"north"}`))
		if calls != 1 {
			t.Fatalf("tool ran %d times, want 1", calls)
		}
		wantOut := ai.ToolOutput{JSON: json.RawMessage(`"moved north"`)}
		wantEvents := []realtime.Event{call("c1", "move", `{"dir":"north"}`), realtime.ToolResult{CallID: "c1", Name: "move", Output: wantOut}}
		if diff := cmp.Diff(wantEvents, h.got()); diff != "" {
			t.Errorf("events mismatch (-want +got):\n%s", diff)
		}
		wantSent := []aitest.RealtimeInput{{Kind: "tool-result", CallID: "c1", Output: wantOut}}
		if diff := cmp.Diff(wantSent, h.sentResults()); diff != "" {
			t.Errorf("sent mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestSession_toolFailures(t *testing.T) {
	visible := ai.NewTool("visible", "", func(context.Context, moveIn) (string, error) { return "", ai.ToolErrorf("no such room") })
	secret := ai.NewTool("secret", "", func(context.Context, moveIn) (string, error) { return "", errors.New("db password is hunter2") })
	panics := ai.NewTool("panics", "", func(context.Context, moveIn) (string, error) { panic("boom hunter2") })
	slow := ai.NewTool("slow", "", func(ctx context.Context, _ moveIn) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}).WithTimeout(time.Second)
	tests := []struct {
		name  string
		input string
		tool  string
		want  string
	}{
		{name: "tool error is shown", tool: "visible", input: `{"dir":"x"}`, want: "no such room"},
		{name: "other errors are hidden", tool: "secret", input: `{"dir":"x"}`, want: "tool failed"},
		{name: "panics are hidden", tool: "panics", input: `{"dir":"x"}`, want: "tool failed"},
		{name: "timeout", tool: "slow", input: `{"dir":"x"}`, want: "tool timed out"},
		{name: "unknown tool", tool: "nope", input: `{}`, want: `tool "nope" does not exist`},
		{name: "schema violation", tool: "visible", input: `{"dir":5}`, want: "invalid input"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				h := newHarness(t, realtime.Config{Tools: []ai.Tool{visible, secret, panics, slow}})
				h.emit(call("c1", tt.tool, tt.input))
				time.Sleep(2 * time.Second)
				synctest.Wait()
				res := h.toolResults()
				if len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Output.Text, tt.want) || strings.Contains(res[0].Output.Text, "hunter2") {
					t.Errorf("results = %+v, want an error containing %q and no secrets", res, tt.want)
				}
				if sent := h.sentResults(); len(sent) != 1 || !sent[0].IsError {
					t.Errorf("sent results = %+v, want one error result", sent)
				}
			})
		})
	}
}

func askTool(calls *int, policy func(context.Context, ai.ApprovalInput) ai.Approval) ai.Tool {
	return moveTool(calls).WithApproval(policy)
}

func TestSession_approvalAsk(t *testing.T) {
	tests := []struct {
		name     string
		approve  bool
		wantRuns int
		wantText string
	}{
		{name: "approved", approve: true, wantRuns: 1, wantText: ""},
		{name: "denied", approve: false, wantRuns: 0, wantText: "denied"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				calls := 0
				h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, func(context.Context, ai.ApprovalInput) ai.Approval { return ai.AskUser("moves are risky") })}})
				h.emit(call("c1", "move", `{"dir":"up"}`))
				req := h.approvalRequest()
				if calls != 0 || len(h.sentResults()) != 0 {
					t.Fatalf("tool ran before approval: runs = %d, sent = %+v", calls, h.sentResults())
				}
				if req.CallID != "c1" || req.Name != "move" || req.Reason != "moves are risky" || string(req.Input) != `{"dir":"up"}` || len(req.ApprovalID) < 20 {
					t.Errorf("ApprovalRequest = %+v, want call c1 with reason, input and a long random id", req)
				}
				if err := h.sess.Approve(t.Context(), req.ApprovalID, tt.approve); err != nil {
					t.Fatalf("Approve() error = %v", err)
				}
				synctest.Wait()
				if calls != tt.wantRuns {
					t.Errorf("tool ran %d times, want %d", calls, tt.wantRuns)
				}
				res := h.toolResults()
				if len(res) != 1 || res[0].IsError != !tt.approve || !strings.Contains(res[0].Output.Text, tt.wantText) {
					t.Errorf("results = %+v, want error=%v containing %q", res, !tt.approve, tt.wantText)
				}
			})
		})
	}
}

func TestSession_forgedApprovalsDoNotRunTools(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		req := h.approvalRequest()
		for _, forged := range []string{"", "c1", "guess", strings.ToLower(req.ApprovalID), req.ApprovalID + "x"} {
			if err := h.sess.Approve(t.Context(), forged, true); !errors.Is(err, ai.ErrInvalidApproval) {
				t.Errorf("Approve(%q) error = %v, want ErrInvalidApproval", forged, err)
			}
		}
		synctest.Wait()
		if calls != 0 || len(h.toolResults()) != 0 {
			t.Fatalf("forged approval ran the tool: runs = %d, results = %+v", calls, h.toolResults())
		}
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); err != nil {
			t.Fatalf("Approve(real) error = %v", err)
		}
		synctest.Wait()
		if calls != 1 {
			t.Errorf("tool ran %d times after the real approval, want 1", calls)
		}
	})
}

func TestSession_approvalIsSingleUse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		req := h.approvalRequest()
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); err != nil {
			t.Fatalf("Approve() error = %v", err)
		}
		synctest.Wait()
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); !errors.Is(err, ai.ErrInvalidApproval) {
			t.Errorf("second Approve() error = %v, want ErrInvalidApproval", err)
		}
		h.emit(call("c2", "move", `{"dir":"down"}`))
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); !errors.Is(err, ai.ErrInvalidApproval) {
			t.Errorf("replayed Approve() for a new call error = %v, want ErrInvalidApproval", err)
		}
		synctest.Wait()
		if calls != 1 {
			t.Errorf("tool ran %d times, want 1: an approval must not carry over to another call", calls)
		}
	})
}

func TestSession_approvalRunsOnlyTheShownInput(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var seen []string
		tool := ai.NewTool("move", "", func(_ context.Context, in moveIn) (string, error) {
			seen = append(seen, in.Dir)
			return "ok", nil
		}).WithApproval(ai.AlwaysAsk)
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{tool}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		req := h.approvalRequest()
		h.emit(call("c1", "move", `{"dir":"rm -rf"}`))
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); err != nil {
			t.Fatalf("Approve() error = %v", err)
		}
		synctest.Wait()
		for _, dir := range seen {
			if dir != "up" {
				t.Errorf("tool ran with %q, want only the approved input", dir)
			}
		}
	})
}

func TestSession_policyIsRecheckedAfterApproval(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls, polls := 0, 0
		policy := func(context.Context, ai.ApprovalInput) ai.Approval {
			if polls++; polls == 1 {
				return ai.AskUser("first look")
			}
			return ai.Deny("policy changed")
		}
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, policy)}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		if err := h.sess.Approve(t.Context(), h.approvalRequest().ApprovalID, true); err != nil {
			t.Fatalf("Approve() error = %v", err)
		}
		synctest.Wait()
		res := h.toolResults()
		if calls != 0 || len(res) != 1 || !strings.Contains(res[0].Output.Text, "policy changed") {
			t.Errorf("runs = %d, results = %+v, want a denial from the second policy check", calls, res)
		}
	})
}

func TestSession_policyDecisions(t *testing.T) {
	tests := []struct {
		name         string
		policy       func(context.Context, ai.ApprovalInput) ai.Approval
		wantRuns     int
		wantApproval bool
		wantText     string
	}{
		{name: "allow runs without asking", policy: func(context.Context, ai.ApprovalInput) ai.Approval { return ai.Allow() }, wantRuns: 1},
		{name: "deny never asks", policy: func(context.Context, ai.ApprovalInput) ai.Approval { return ai.Deny("not today") }, wantText: "not today"},
		{name: "zero value asks", policy: func(context.Context, ai.ApprovalInput) ai.Approval { return ai.Approval{} }, wantApproval: true},
		{name: "panicking policy asks", policy: func(context.Context, ai.ApprovalInput) ai.Approval { panic("policy bug") }, wantApproval: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				calls := 0
				h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, tt.policy)}})
				h.emit(call("c1", "move", `{"dir":"up"}`))
				asked := false
				for _, ev := range h.got() {
					_, isReq := ev.(realtime.ApprovalRequest)
					asked = asked || isReq
				}
				if calls != tt.wantRuns || asked != tt.wantApproval {
					t.Errorf("runs = %d, asked = %v, want %d and %v", calls, asked, tt.wantRuns, tt.wantApproval)
				}
				if tt.wantText != "" {
					if res := h.toolResults(); len(res) != 1 || !strings.Contains(res[0].Output.Text, tt.wantText) {
						t.Errorf("results = %+v, want text %q", res, tt.wantText)
					}
				}
			})
		})
	}
}

func TestSession_unansweredApprovalIsDeniedAfterTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{ApprovalTimeout: time.Minute, Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		req := h.approvalRequest()
		time.Sleep(59 * time.Second)
		synctest.Wait()
		if len(h.toolResults()) != 0 {
			t.Fatal("tool resolved before the timeout")
		}
		time.Sleep(2 * time.Second)
		synctest.Wait()
		res := h.toolResults()
		if calls != 0 || len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Output.Text, "no approval") {
			t.Errorf("runs = %d, results = %+v, want a denial", calls, res)
		}
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); !errors.Is(err, ai.ErrInvalidApproval) {
			t.Errorf("late Approve() error = %v, want ErrInvalidApproval", err)
		}
	})
}

func TestSession_pendingApprovalsAreBounded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{MaxPendingApprovals: 2, Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}})
		for _, id := range []string{"c1", "c2", "c3"} {
			h.emit(call(id, "move", `{"dir":"up"}`))
		}
		requests := 0
		for _, ev := range h.got() {
			if _, ok := ev.(realtime.ApprovalRequest); ok {
				requests++
			}
		}
		res := h.toolResults()
		if requests != 2 || len(res) != 1 || res[0].CallID != "c3" || !strings.Contains(res[0].Output.Text, "too many pending") {
			t.Errorf("requests = %d, results = %+v, want 2 requests and c3 rejected", requests, res)
		}
	})
}

func TestSession_inFlightToolsAreBounded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		block := make(chan struct{})
		tool := ai.NewTool("wait", "", func(ctx context.Context, _ moveIn) (string, error) {
			select {
			case <-block:
			case <-ctx.Done():
			}
			return "ok", nil
		})
		h := newHarness(t, realtime.Config{MaxInFlightTools: 1, Tools: []ai.Tool{tool}})
		h.emit(call("c1", "wait", `{"dir":"a"}`))
		h.emit(call("c2", "wait", `{"dir":"a"}`))
		if res := h.toolResults(); len(res) != 1 || res[0].CallID != "c2" || !strings.Contains(res[0].Output.Text, "too many") {
			t.Errorf("results = %+v, want c2 rejected while c1 runs", res)
		}
		close(block)
		synctest.Wait()
		if res := h.toolResults(); len(res) != 2 {
			t.Errorf("results = %+v, want c1 to finish after unblocking", res)
		}
	})
}

func TestSession_interruptCancelsRunningTools(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var cause error
		started := false
		tool := ai.NewTool("long", "", func(ctx context.Context, _ moveIn) (string, error) {
			started = true
			<-ctx.Done()
			cause = context.Cause(ctx)
			return "late", nil
		})
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{tool}})
		h.emit(call("c1", "long", `{"dir":"a"}`))
		if !started {
			t.Fatal("tool did not start")
		}
		h.emit(realtime.Interrupted{})
		if !errors.Is(cause, realtime.ErrInterrupted) {
			t.Errorf("tool ctx cause = %v, want ErrInterrupted", cause)
		}
		if len(h.sentResults()) != 0 || len(h.toolResults()) != 0 {
			t.Errorf("a canceled tool reported a result: sent = %+v, events = %+v", h.sentResults(), h.toolResults())
		}
	})
}

func TestSession_providerCancelsOneTool(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		canceled := map[string]bool{}
		var mu sync.Mutex
		tool := ai.NewTool("long", "", func(ctx context.Context, in moveIn) (string, error) {
			<-ctx.Done()
			mu.Lock()
			canceled[in.Dir] = true
			mu.Unlock()
			return "", ctx.Err()
		})
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{tool}})
		h.emit(call("c1", "long", `{"dir":"one"}`))
		h.emit(call("c2", "long", `{"dir":"two"}`))
		h.emit(realtime.ToolCanceled{CallID: "c1"})
		mu.Lock()
		defer mu.Unlock()
		if !canceled["one"] || canceled["two"] {
			t.Errorf("canceled = %v, want only the first call canceled", canceled)
		}
	})
}

func TestSession_pendingApprovalIsDroppedOnInterrupt(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		calls := 0
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}})
		h.emit(call("c1", "move", `{"dir":"up"}`))
		req := h.approvalRequest()
		h.emit(realtime.Interrupted{})
		if err := h.sess.Approve(t.Context(), req.ApprovalID, true); !errors.Is(err, ai.ErrInvalidApproval) {
			t.Errorf("Approve() after interrupt error = %v, want ErrInvalidApproval", err)
		}
		synctest.Wait()
		if calls != 0 {
			t.Errorf("tool ran %d times after its call was interrupted", calls)
		}
	})
}

func TestSession_taintTracksUntrustedOutput(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var tainted []bool
		fetch := ai.NewTool("fetch", "", func(context.Context, moveIn) (string, error) { return "page", nil })
		fetch.Untrusted = true
		act := ai.NewTool("act", "", func(context.Context, moveIn) (string, error) { return "done", nil }).
			WithApproval(func(_ context.Context, in ai.ApprovalInput) ai.Approval {
				tainted = append(tainted, in.Tainted)
				return ai.Allow()
			})
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{fetch, act}})
		h.emit(call("c1", "act", `{"dir":"a"}`))
		h.emit(call("c2", "fetch", `{"dir":"a"}`))
		h.emit(call("c3", "act", `{"dir":"a"}`))
		if diff := cmp.Diff([]bool{false, true}, tainted); diff != "" {
			t.Errorf("Tainted flags mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestSession_clientToolsAreOnlyDeclared(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{Tools: []ai.Tool{ai.NewClientTool[moveIn]("ui", "client side")}})
		h.emit(call("c1", "ui", `{"dir":"a"}`))
		if len(h.toolResults()) != 0 || len(h.sentResults()) != 0 {
			t.Fatalf("a client tool was answered by the session: %+v", h.toolResults())
		}
		if err := h.sess.SendToolResult(t.Context(), "c1", ai.ToolOutput{Text: "clicked"}, false); err != nil {
			t.Fatalf("SendToolResult() error = %v", err)
		}
		if sent := h.sentResults(); len(sent) != 1 || sent[0].Output.Text != "clicked" {
			t.Errorf("sent = %+v, want the application's answer", sent)
		}
	})
}

func TestSession_inputLimits(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{MaxInputBytes: 8})
		ctx := t.Context()
		if err := h.sess.SendAudio(ctx, make([]byte, 9)); !errors.Is(err, realtime.ErrInputTooLarge) {
			t.Errorf("SendAudio() error = %v, want ErrInputTooLarge", err)
		}
		if err := h.sess.SendText(ctx, "123456789"); !errors.Is(err, realtime.ErrInputTooLarge) {
			t.Errorf("SendText() error = %v, want ErrInputTooLarge", err)
		}
		if err := h.sess.SendAudio(ctx, make([]byte, 8)); err != nil {
			t.Errorf("SendAudio() at the limit error = %v", err)
		}
		if len(h.model.Sent()) != 1 {
			t.Errorf("sent %d inputs, want only the one within the limit", len(h.model.Sent()))
		}
	})
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{})
		if err := h.sess.SendAudio(t.Context(), make([]byte, 1<<20+1)); !errors.Is(err, realtime.ErrInputTooLarge) {
			t.Errorf("SendAudio() above the 1 MiB default error = %v, want ErrInputTooLarge", err)
		}
	})
}

func TestSession_slowConsumerPausesProvider(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		model := aitest.NewRealtimeModel()
		sess, err := realtime.Connect(t.Context(), model, realtime.Config{EventBuffer: 2})
		if err != nil {
			t.Fatalf("Connect() error = %v", err)
		}
		t.Cleanup(func() { _ = sess.Close() })
		accepted := 0
		go func() {
			for range 10 {
				if model.Emit(t.Context(), realtime.TextDelta{Text: "x"}) != nil {
					return
				}
				accepted++
			}
		}()
		synctest.Wait()
		if accepted > 4 {
			t.Errorf("provider emitted %d events while nobody read, want it paused by the bounded buffer", accepted)
		}
		n := 0
		for range sess.Events() {
			if n++; n == 10 {
				break
			}
		}
		if n != 10 {
			t.Errorf("read %d events, want all 10 after the consumer caught up", n)
		}
	})
}

func TestSession_eventsAreSingleUse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{})
		synctest.Wait()
		for range h.sess.Events() {
			t.Error("second Events() loop yielded")
		}
		if !errors.Is(h.sess.Err(), realtime.ErrEventsConsumed) {
			t.Errorf("Err() = %v, want ErrEventsConsumed", h.sess.Err())
		}
	})
}

func TestSession_providerFailureEndsSession(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newHarness(t, realtime.Config{})
		lost := errors.New("connection reset")
		if err := h.model.FailStream(t.Context(), lost); err != nil {
			t.Fatalf("Fail() error = %v", err)
		}
		synctest.Wait()
		if !errors.Is(h.sess.Err(), lost) {
			t.Errorf("Err() = %v, want the provider error", h.sess.Err())
		}
		if err := h.sess.SendText(t.Context(), "x"); !errors.Is(err, realtime.ErrClosed) {
			t.Errorf("SendText() after failure error = %v, want ErrClosed", err)
		}
	})
}

func TestSession_contextEndsSession(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		sess, err := realtime.Connect(ctx, aitest.NewRealtimeModel(), realtime.Config{})
		if err != nil {
			t.Fatalf("Connect() error = %v", err)
		}
		cancel()
		for range sess.Events() {
		}
		if !errors.Is(sess.Err(), context.Canceled) {
			t.Errorf("Err() = %v, want context.Canceled", sess.Err())
		}
		if err := sess.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestSession_closeIsIdempotentAndClean(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		model := aitest.NewRealtimeModel()
		sess, err := realtime.Connect(t.Context(), model, realtime.Config{})
		if err != nil {
			t.Fatalf("Connect() error = %v", err)
		}
		ended := make(chan struct{})
		go func() {
			for range sess.Events() {
			}
			close(ended)
		}()
		synctest.Wait()
		if err := errors.Join(sess.Close(), sess.Close()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
		<-ended
		if sess.Err() != nil {
			t.Errorf("Err() = %v, want nil after Close", sess.Err())
		}
		ctx := t.Context()
		if !errors.Is(sess.SendAudio(ctx, []byte("x")), realtime.ErrClosed) || !errors.Is(sess.SendText(ctx, "x"), realtime.ErrClosed) ||
			!errors.Is(sess.Interrupt(ctx), realtime.ErrClosed) || !errors.Is(sess.SendToolResult(ctx, "c", ai.ToolOutput{}, false), realtime.ErrClosed) {
			t.Error("sends after Close must fail with ErrClosed")
		}
	})
}

func TestSession_closeCancelsRunningTools(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		exited := false
		tool := ai.NewTool("long", "", func(ctx context.Context, _ moveIn) (string, error) {
			<-ctx.Done()
			exited = true
			return "", ctx.Err()
		})
		model := aitest.NewRealtimeModel()
		sess, err := realtime.Connect(t.Context(), model, realtime.Config{Tools: []ai.Tool{tool}})
		if err != nil {
			t.Fatalf("Connect() error = %v", err)
		}
		go func() {
			for range sess.Events() {
			}
		}()
		if err := model.Emit(t.Context(), call("c1", "long", `{"dir":"a"}`)); err != nil {
			t.Fatalf("Emit() error = %v", err)
		}
		synctest.Wait()
		if err := sess.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if !exited {
			t.Error("Close() returned while a tool was still running")
		}
	})
}

func TestConnect_errors(t *testing.T) {
	tests := []struct {
		name  string
		model realtime.Model
		cfg   realtime.Config
		want  string
	}{
		{name: "nil model", cfg: realtime.Config{}, want: "nil model"},
		{name: "duplicate tools", model: aitest.NewRealtimeModel(), cfg: realtime.Config{Tools: []ai.Tool{ai.NewClientTool[moveIn]("a", ""), ai.NewClientTool[moveIn]("a", "")}}, want: "duplicate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := realtime.Connect(t.Context(), tt.model, tt.cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Connect() error = %v, want %q", err, tt.want)
			}
		})
	}
	failing := aitest.NewRealtimeModel(aitest.Fail(errors.New("dial failed")))
	if _, err := realtime.Connect(t.Context(), failing, realtime.Config{}); err == nil || !strings.Contains(err.Error(), "dial failed") {
		t.Errorf("Connect() error = %v, want the dial error", err)
	}
}

type toolTelemetry struct {
	mu       sync.Mutex
	started  []string
	finished []error
}

func (r *toolTelemetry) StartStep(ctx context.Context, _ ai.StepInfo) context.Context { return ctx }

func (r *toolTelemetry) FinishStep(context.Context, ai.StepInfo, ai.Step, error) {}

func (r *toolTelemetry) StartTool(ctx context.Context, c ai.ToolCall) context.Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.started = append(r.started, c.Name)
	return ctx
}

func (r *toolTelemetry) FinishTool(_ context.Context, _ ai.ToolCall, _ ai.ToolResult, _ time.Duration, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.finished = append(r.finished, err)
}

func TestSession_telemetryWrapsTools(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		tel := &toolTelemetry{}
		calls := 0
		h := newHarness(t, realtime.Config{Telemetry: tel, Tools: []ai.Tool{moveTool(&calls)}})
		h.emit(call("c1", "move", `{"dir":"a"}`))
		if diff := cmp.Diff([]string{"move"}, tel.started); diff != "" || len(tel.finished) != 1 || tel.finished[0] != nil {
			t.Errorf("telemetry = %+v, want one finished move call (diff %s)", tel, diff)
		}
	})
}

func TestMintClientSecret(t *testing.T) {
	tests := []struct {
		name    string
		opts    realtime.MintOptions
		wantTTL time.Duration
	}{
		{name: "default is one minute", opts: realtime.MintOptions{}, wantTTL: time.Minute},
		{name: "explicit", opts: realtime.MintOptions{TTL: 5 * time.Minute}, wantTTL: 5 * time.Minute},
		{name: "one hour is allowed", opts: realtime.MintOptions{TTL: time.Hour}, wantTTL: time.Hour},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				model := aitest.NewRealtimeModel()
				secret, err := realtime.MintClientSecret(t.Context(), model, realtime.Config{Instructions: "locked text", Voice: "marin"}, tt.opts)
				if err != nil {
					t.Fatalf("MintClientSecret() error = %v", err)
				}
				if got := model.MintRequests(); len(got) != 1 || got[0].TTL != tt.wantTTL {
					t.Errorf("provider saw %+v, want TTL %v", got, tt.wantTTL)
				}
				if !secret.ExpiresAt.Equal(time.Now().Add(tt.wantTTL)) || secret.Value == "" {
					t.Errorf("secret = %+v, want a value expiring in %v", secret, tt.wantTTL)
				}
				if cfg := model.Config(); cfg.Instructions != "locked text" || cfg.Voice != "marin" {
					t.Errorf("provider config = %+v, want the instructions and voice", cfg)
				}
			})
		})
	}
}

func TestMintClientSecret_rejectsBadTTL(t *testing.T) {
	for _, ttl := range []time.Duration{-time.Second, time.Hour + time.Nanosecond, 24 * time.Hour} {
		model := aitest.NewRealtimeModel()
		if _, err := realtime.MintClientSecret(t.Context(), model, realtime.Config{}, realtime.MintOptions{TTL: ttl}); err == nil {
			t.Errorf("MintClientSecret(TTL %v) succeeded", ttl)
		}
		if len(model.MintRequests()) != 0 {
			t.Errorf("TTL %v reached the provider", ttl)
		}
	}
}

func TestMintClientSecret_toolsRunOnTheClient(t *testing.T) {
	calls := 0
	model := aitest.NewRealtimeModel()
	_, err := realtime.MintClientSecret(t.Context(), model, realtime.Config{Tools: []ai.Tool{askTool(&calls, ai.AlwaysAsk)}}, realtime.MintOptions{})
	if !errors.Is(err, realtime.ErrApprovalUnenforceable) || len(model.MintRequests()) != 0 {
		t.Fatalf("MintClientSecret() error = %v, want ErrApprovalUnenforceable before any provider call", err)
	}
	tool := moveTool(&calls)
	secret, err := realtime.MintClientSecret(t.Context(), model, realtime.Config{Tools: []ai.Tool{tool}}, realtime.MintOptions{})
	if err != nil {
		t.Fatalf("MintClientSecret() error = %v", err)
	}
	if got := model.Config().Tools; len(got) != 1 || got[0].Execute != nil || got[0].Name != "move" {
		t.Errorf("provider tools = %+v, want the declaration without Execute", got)
	}
	if tool.Execute == nil {
		t.Error("MintClientSecret() modified the caller's tool")
	}
	if len(secret.Warnings) != 1 || !strings.Contains(secret.Warnings[0].Message, "runs on the client") {
		t.Errorf("warnings = %+v, want a client-side tool warning", secret.Warnings)
	}
}

func TestMintClientSecret_unsupportedModel(t *testing.T) {
	_, err := realtime.MintClientSecret(t.Context(), plainModel{}, realtime.Config{}, realtime.MintOptions{})
	if !errors.Is(err, realtime.ErrUnsupported) {
		t.Errorf("MintClientSecret() error = %v, want ErrUnsupported", err)
	}
}

type plainModel struct{}

func (plainModel) Provider() string { return "plain" }

func (plainModel) ModelID() string { return "plain" }

func (plainModel) Connect(context.Context, *realtime.Config) (realtime.Conn, error) {
	return nil, errors.New("no")
}
