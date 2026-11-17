package ai_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func TestGenerateText_text(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("hello"), aitest.Usage(7, 3)))
	res := mustGenerate(t, m, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})

	if res.Text != "hello" {
		t.Errorf("Text = %q, want %q", res.Text, "hello")
	}
	if res.FinishReason != ai.FinishStop {
		t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishStop)
	}
	if got, want := res.Usage, (ai.Usage{Input: 7, Output: 3}); !cmp.Equal(got, want) {
		t.Errorf("Usage mismatch (-want +got):\n%s", cmp.Diff(want, got))
	}
	want := []ai.Message{ai.Assistant("hello")}
	if diff := cmp.Diff(want, res.Messages); diff != "" {
		t.Errorf("Messages mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateText_toolLoop(t *testing.T) {
	m := aitest.NewModel(
		toolTurn("search", `{"query":"go"}`),
		aitest.Turn(aitest.Text("found it"), aitest.Usage(4, 2)),
	)
	res := mustGenerate(t, m, ai.TextRequest{
		Messages: []ai.Message{ai.User("find go")},
		Tools:    []ai.Tool{searchTool()},
	})

	if got, want := len(res.Steps), 2; got != want {
		t.Fatalf("len(Steps) = %d, want %d", got, want)
	}
	if res.Text != "found it" {
		t.Errorf("Text = %q, want %q", res.Text, "found it")
	}
	if got, want := res.Usage.Input, 14; got != want {
		t.Errorf("Usage.Input = %d, want %d", got, want)
	}
	results := resultsOf(res.Messages)
	if len(results) != 1 || string(results[0].Output.JSON) != `{"hits":["go-1"]}` {
		t.Fatalf("tool results = %+v", results)
	}
	if got, want := res.Messages[1].Role, ai.RoleTool; got != want {
		t.Errorf("Messages[1].Role = %q, want %q", got, want)
	}
	second := m.Calls()[1]
	if got, want := len(second.Messages), 3; got != want {
		t.Errorf("second call messages = %d, want %d", got, want)
	}
}

func TestGenerateText_stepLimit(t *testing.T) {
	loop := func(n int) []aitest.ModelOption {
		turns := make([]aitest.ModelOption, n)
		for i := range turns {
			turns[i] = toolTurn("search", `{"query":"x"}`)
		}
		return turns
	}
	tests := []struct {
		name      string
		maxSteps  int
		turns     int
		wantSteps int
	}{
		{name: "explicit cap", maxSteps: 3, turns: 10, wantSteps: 3},
		{name: "default cap is 20", maxSteps: 0, turns: 30, wantSteps: 20},
		{name: "one step", maxSteps: 1, turns: 5, wantSteps: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := mustGenerate(t, aitest.NewModel(loop(tt.turns)...), ai.TextRequest{
				Tools:    []ai.Tool{searchTool()},
				MaxSteps: tt.maxSteps,
			})
			if got := len(res.Steps); got != tt.wantSteps {
				t.Errorf("len(Steps) = %d, want %d", got, tt.wantSteps)
			}
			if res.FinishReason != ai.FinishStepLimit {
				t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishStepLimit)
			}
		})
	}
}

func TestGenerateText_stepLimitNotHitWhenModelStops(t *testing.T) {
	m := aitest.NewModel(toolTurn("search", `{"query":"x"}`), aitest.Turn(aitest.Text("done")))
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool()}, MaxSteps: 2})
	if res.FinishReason != ai.FinishStop {
		t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishStop)
	}
}

func TestGenerateText_stopWhen(t *testing.T) {
	tests := []struct {
		name      string
		stop      ai.StopCondition
		wantSteps int
	}{
		{name: "has tool call", stop: ai.HasToolCall("search"), wantSteps: 1},
		{name: "other tool never matches", stop: ai.HasToolCall("other"), wantSteps: 3},
		{name: "token budget", stop: ai.MaxTotalTokens(30), wantSteps: 2},
		{name: "custom", stop: func(steps []ai.Step) bool { return len(steps) == 2 }, wantSteps: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewModel(
				toolTurn("search", `{"query":"a"}`),
				toolTurn("search", `{"query":"b"}`),
				toolTurn("search", `{"query":"c"}`),
			)
			res := mustGenerate(t, m, ai.TextRequest{
				Tools:    []ai.Tool{searchTool()},
				StopWhen: []ai.StopCondition{tt.stop},
				MaxSteps: 3,
			})
			if got := len(res.Steps); got != tt.wantSteps {
				t.Errorf("len(Steps) = %d, want %d", got, tt.wantSteps)
			}
		})
	}
}

func TestGenerateText_prepareStepEditsClone(t *testing.T) {
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("ok")))
	req := ai.TextRequest{
		Instructions: ai.Texts("base"),
		Messages:     []ai.Message{ai.User("hi")},
		Tools:        []ai.Tool{searchTool()},
		Temperature:  new(0.5),
		PrepareStep: func(_ context.Context, s ai.StepState, step *ai.StepRequest) error {
			step.Instructions = append(step.Instructions, ai.Text{Text: "extra"})
			step.Messages[0] = ai.User("changed")
			step.Temperature = new(float64(s.Number))
			if s.Number == 1 {
				step.Tools = nil
			}
			return nil
		},
	}
	mustGenerate(t, m, req)

	if got, want := len(req.Instructions), 1; got != want {
		t.Errorf("caller Instructions mutated: len = %d, want %d", got, want)
	}
	if got, want := req.Messages[0], ai.User("hi"); !cmp.Equal(got, want) {
		t.Errorf("caller Messages mutated: %+v", got)
	}
	if got, want := *req.Temperature, 0.5; got != want {
		t.Errorf("caller Temperature mutated: %v", got)
	}
	calls := m.Calls()
	if got, want := len(calls[0].Instructions), 2; got != want {
		t.Errorf("call 0 instructions = %d, want %d", got, want)
	}
	if got, want := *calls[1].Temperature, 1.0; got != want {
		t.Errorf("call 1 Temperature = %v, want %v", got, want)
	}
	if len(calls[1].Tools) != 0 || len(calls[0].Tools) != 1 {
		t.Errorf("tools per call = %d, %d; want 1, 0", len(calls[0].Tools), len(calls[1].Tools))
	}
}

func TestGenerateText_prepareStepError(t *testing.T) {
	boom := errors.New("boom")
	m := aitest.NewModel(aitest.Turn(aitest.Text("x")))
	res, err := ai.GenerateText(t.Context(), m, ai.TextRequest{
		PrepareStep: func(context.Context, ai.StepState, *ai.StepRequest) error { return boom },
	})
	if !errors.Is(err, boom) {
		t.Errorf("error = %v, want wrapping %v", err, boom)
	}
	if len(m.Calls()) != 0 || len(res.Steps) != 0 {
		t.Errorf("model called %d times, steps %d; want 0, 0", len(m.Calls()), len(res.Steps))
	}
}

func TestGenerateText_onStepEnd(t *testing.T) {
	boom := errors.New("stop here")
	var seen []int
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), toolTurn("search", `{"query":"b"}`), aitest.Turn(aitest.Text("x")))
	res, err := ai.GenerateText(t.Context(), m, ai.TextRequest{
		Tools: []ai.Tool{searchTool()},
		OnStepEnd: func(_ context.Context, s ai.Step) error {
			seen = append(seen, s.Number)
			if s.Number == 1 {
				return boom
			}
			return nil
		},
	})
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want wrapping %v", err, boom)
	}
	if diff := cmp.Diff([]int{0, 1}, seen); diff != "" {
		t.Errorf("OnStepEnd steps mismatch (-want +got):\n%s", diff)
	}
	if got, want := len(res.Steps), 2; got != want {
		t.Errorf("completed steps = %d, want %d", got, want)
	}
}

func TestGenerateText_providerPause(t *testing.T) {
	m := aitest.NewModel(
		aitest.Turn(aitest.Text("part one "), aitest.Finish(ai.FinishPause)),
		aitest.Turn(aitest.Text("part two")),
	)
	res := mustGenerate(t, m, ai.TextRequest{Messages: []ai.Message{ai.User("go")}})

	if got, want := len(res.Steps), 2; got != want {
		t.Fatalf("len(Steps) = %d, want %d", got, want)
	}
	if res.Text != "part two" {
		t.Errorf("Text = %q, want %q", res.Text, "part two")
	}
	second := m.Calls()[1].Messages
	if got, want := second[len(second)-1].Role, ai.RoleAssistant; got != want {
		t.Errorf("second call ends with role %q, want %q", got, want)
	}
}

func TestGenerateText_pauseHitsStepLimit(t *testing.T) {
	m := aitest.NewModel(
		aitest.Turn(aitest.Text("a"), aitest.Finish(ai.FinishPause)),
		aitest.Turn(aitest.Text("b"), aitest.Finish(ai.FinishPause)),
	)
	res := mustGenerate(t, m, ai.TextRequest{MaxSteps: 2})
	if res.FinishReason != ai.FinishStepLimit {
		t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishStepLimit)
	}
}

func TestGenerateText_toolConcurrencyAndOrder(t *testing.T) {
	tests := []struct {
		name     string
		limit    int
		wantPeak int32
	}{
		{name: "default is 8", limit: 0, wantPeak: 8},
		{name: "limit 2", limit: 2, wantPeak: 2},
		{name: "limit 1 is sequential", limit: 1, wantPeak: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var running, peak atomic.Int32
				slow := ai.NewTool("slow", "d", func(_ context.Context, in struct {
					N int `json:"n"`
				},
				) (int, error) {
					cur := running.Add(1)
					for {
						old := peak.Load()
						if cur <= old || peak.CompareAndSwap(old, cur) {
							break
						}
					}
					time.Sleep(time.Duration(10-in.N) * time.Millisecond)
					running.Add(-1)
					return in.N, nil
				})
				var items []aitest.Item
				for i := range 10 {
					items = append(items, aitest.ToolCall("slow", `{"n":`+string(rune('0'+i))+`}`))
				}
				m := aitest.NewModel(aitest.Turn(items...), aitest.Turn(aitest.Text("done")))
				res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{slow}, MaxToolConcurrency: tt.limit})

				if got := peak.Load(); got != tt.wantPeak {
					t.Errorf("peak concurrency = %d, want %d", got, tt.wantPeak)
				}
				var got []string
				for _, r := range resultsOf(res.Messages) {
					got = append(got, string(r.Output.JSON))
				}
				want := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("result order mismatch (-want +got):\n%s", diff)
				}
			})
		})
	}
}

func TestGenerateText_toolFailures(t *testing.T) {
	secret := errors.New("db password is hunter2")
	failing := func(name string, err error) ai.Tool {
		return ai.NewTool(name, "d", func(context.Context, struct{}) (int, error) { return 0, err })
	}
	panicking := ai.NewTool("panics", "d", func(context.Context, struct{}) (int, error) { panic("kaboom hunter2") })
	tests := []struct {
		name     string
		tool     ai.Tool
		wantText string
	}{
		{name: "plain error is hidden", tool: failing("t", secret), wantText: "tool failed"},
		{name: "ToolErrorf is visible", tool: failing("t", ai.ToolErrorf("no such city %q", "Oz")), wantText: `no such city "Oz"`},
		{name: "wrapped ToolError is visible", tool: failing("t", errors.Join(secret, ai.ToolErrorf("try again"))), wantText: "try again"},
		{name: "panic is hidden", tool: panicking, wantText: "tool failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name := tt.tool.Name
			m := aitest.NewModel(toolTurn(name, `{}`), aitest.Turn(aitest.Text("ok")))
			res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{tt.tool}})

			results := resultsOf(res.Messages)
			if len(results) != 1 || !results[0].IsError {
				t.Fatalf("results = %+v, want one error result", results)
			}
			if got := results[0].Output.Text; got != tt.wantText {
				t.Errorf("model-visible text = %q, want %q", got, tt.wantText)
			}
			if len(res.Steps) != 2 {
				t.Errorf("len(Steps) = %d, want 2 (error is fed back)", len(res.Steps))
			}
		})
	}
}

func TestGenerateText_toolFeedback(t *testing.T) {
	tests := []struct {
		name     string
		call     string
		input    string
		wantPart string
	}{
		{name: "unknown tool", call: "nope", input: `{}`, wantPart: `available tools: search`},
		{name: "schema violation", call: "search", input: `{"query":42}`, wantPart: `invalid input for tool "search"`},
		{name: "missing field", call: "search", input: `{}`, wantPart: `invalid input for tool "search"`},
		{name: "not JSON", call: "search", input: `{bad`, wantPart: `invalid input for tool "search"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewModel(toolTurn(tt.call, tt.input), aitest.Turn(aitest.Text("recovered")))
			res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})

			results := resultsOf(res.Messages)
			if len(results) != 1 || !results[0].IsError {
				t.Fatalf("results = %+v, want one error result", results)
			}
			if got := results[0].Output.Text; !strings.Contains(got, tt.wantPart) {
				t.Errorf("result text = %q, want it to contain %q", got, tt.wantPart)
			}
			if res.Text != "recovered" {
				t.Errorf("Text = %q, want %q", res.Text, "recovered")
			}
		})
	}
}

func TestGenerateText_clientTool(t *testing.T) {
	final := ai.NewClientTool[searchIn]("final_answer", "submit")
	m := aitest.NewModel(
		aitest.Turn(aitest.ToolCall("search", `{"query":"q"}`), aitest.ToolCall("final_answer", `{"query":"done"}`)),
		aitest.Turn(aitest.Text("never reached")),
	)
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool(), final}})

	if got, want := len(res.Steps), 1; got != want {
		t.Fatalf("len(Steps) = %d, want %d", got, want)
	}
	if res.FinishReason != ai.FinishToolCalls {
		t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishToolCalls)
	}
	var names []string
	for _, p := range res.Content {
		if c, ok := p.(ai.ToolCall); ok {
			names = append(names, c.Name)
		}
	}
	if diff := cmp.Diff([]string{"search", "final_answer"}, names); diff != "" {
		t.Errorf("content tool calls mismatch (-want +got):\n%s", diff)
	}
	if got := len(resultsOf(res.Messages)); got != 1 {
		t.Errorf("executed results = %d, want 1 (only the server tool)", got)
	}
}

func TestGenerateText_duplicateTools(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("x")))
	_, err := ai.GenerateText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool(), searchTool()}})
	if !errors.Is(err, ai.ErrDuplicateTool) {
		t.Errorf("error = %v, want %v", err, ai.ErrDuplicateTool)
	}
	if len(m.Calls()) != 0 {
		t.Errorf("model was called despite duplicate tools")
	}
}

func TestGenerateText_prepareStepDuplicateTools(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("x")))
	_, err := ai.GenerateText(t.Context(), m, ai.TextRequest{
		Tools: []ai.Tool{searchTool()},
		PrepareStep: func(_ context.Context, _ ai.StepState, s *ai.StepRequest) error {
			s.Tools = append(s.Tools, searchTool())
			return nil
		},
	})
	if !errors.Is(err, ai.ErrDuplicateTool) {
		t.Errorf("error = %v, want %v", err, ai.ErrDuplicateTool)
	}
}

func TestGenerateText_errorKeepsCompletedSteps(t *testing.T) {
	boom := errors.New("provider down")
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("partial"), aitest.Err(boom)))
	res, err := ai.GenerateText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})

	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want %v", err, boom)
	}
	if got, want := len(res.Steps), 1; got != want {
		t.Errorf("completed steps = %d, want %d", got, want)
	}
	if got, want := len(res.Messages), 2; got != want {
		t.Errorf("messages = %d, want %d (assistant call + tool result)", got, want)
	}
	if res.FinishReason != ai.FinishError {
		t.Errorf("FinishReason = %q, want %q", res.FinishReason, ai.FinishError)
	}
}

func TestGenerateText_requestNotMutated(t *testing.T) {
	req := ai.TextRequest{
		Instructions: ai.Texts("sys"),
		Messages:     []ai.Message{ai.User("hi")},
		Tools:        []ai.Tool{searchTool()},
		Headers:      map[string][]string{"X-A": {"1"}},
	}
	before := req.Clone()
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, m, req)

	opts := []cmp.Option{
		cmpopts.IgnoreFields(ai.Tool{}, "Execute", "Approval"),
		cmpopts.IgnoreFields(ai.TextRequest{}, "PrepareStep", "OnStepEnd", "StopWhen", "Output", "Telemetry"),
		cmpopts.EquateComparable(ai.ToolChoice{}),
	}
	if diff := cmp.Diff(before, req, opts...); diff != "" {
		t.Errorf("request mutated (-before +after):\n%s", diff)
	}
}

func TestGenerateText_toolTimeout(t *testing.T) {
	tests := []struct {
		name string
		tool ai.Tool
		req  ai.TextRequest
	}{
		{name: "Tool.Timeout", tool: slowTool().WithTimeout(time.Second)},
		{name: "Timeouts.Tool", tool: slowTool(), req: ai.TextRequest{Timeouts: ai.Timeouts{Tool: time.Second}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				req := tt.req
				req.Tools = []ai.Tool{tt.tool}
				m := aitest.NewModel(toolTurn("slow", `{}`), aitest.Turn(aitest.Text("ok")))
				res := mustGenerate(t, m, req)
				results := resultsOf(res.Messages)
				if len(results) != 1 || results[0].Output.Text != "tool timed out" || !results[0].IsError {
					t.Errorf("results = %+v, want a timed out error result", results)
				}
			})
		})
	}
}

func slowTool() ai.Tool {
	return ai.NewTool("slow", "d", func(ctx context.Context, _ struct{}) (int, error) {
		select {
		case <-time.After(time.Hour):
			return 1, nil
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	})
}

func TestGenerateText_cancelDuringTool(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		m := aitest.NewModel(toolTurn("slow", `{}`), aitest.Turn(aitest.Text("ok")))
		go func() {
			time.Sleep(time.Second)
			cancel()
		}()
		res, err := ai.GenerateText(ctx, m, ai.TextRequest{Tools: []ai.Tool{slowTool()}})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want context.Canceled", err)
		}
		if len(res.Steps) != 0 {
			t.Errorf("completed steps = %d, want 0 (failed step is excluded)", len(res.Steps))
		}
	})
}

func TestGenerateText_providerExecutedCallsAreNotRun(t *testing.T) {
	var ran atomic.Bool
	tool := ai.NewTool("web", "d", func(context.Context, struct{}) (int, error) {
		ran.Store(true)
		return 1, nil
	})
	call := ai.ToolCall{ID: "srv1", Name: "web", Input: rawJSON(`{}`), ProviderExecuted: true}
	result := ai.ToolResult{CallID: "srv1", Name: "web", Output: ai.ToolOutput{Text: "r"}, ProviderExecuted: true}
	m := aitest.NewModel(aitest.Turn(aitest.Part(call), aitest.Part(result), aitest.Text("answer")))
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{tool}})

	if ran.Load() {
		t.Error("provider-executed tool call was executed locally")
	}
	if got, want := len(res.Steps), 1; got != want {
		t.Errorf("len(Steps) = %d, want %d", got, want)
	}
}

func TestGenerateText_telemetry(t *testing.T) {
	rec := &recorder{}
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool()}, Telemetry: rec})

	want := []string{"step-start:0", "tool-start:search", "tool-end:search", "step-end:0", "step-start:1", "step-end:1"}
	if diff := cmp.Diff(want, rec.events); diff != "" {
		t.Errorf("telemetry events mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateText_telemetryReceivesRequestAndToolResult(t *testing.T) {
	rec := &recorder{}
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, m, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{searchTool()}, Telemetry: rec})

	if diff := cmp.Diff([]int{1, 3}, rec.inputs); diff != "" {
		t.Errorf("step request message counts mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{`{"hits":["a-1"]}`}, rec.output); diff != "" {
		t.Errorf("tool results mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateText_nilTelemetryIsOff(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	if res := mustGenerate(t, m, ai.TextRequest{}); res.Text != "ok" {
		t.Errorf("Text = %q, want %q", res.Text, "ok")
	}
}
