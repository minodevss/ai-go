package ai_test

import (
	"context"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func TestStreamText_consumerStopKeepsApprovedToolResult(t *testing.T) {
	d := &deleter{}
	req, pending := pendingFixture(t, d)
	r := requestsOf(pending)[0]
	hist := resume(req, pending, approvalMessage(ai.ToolApprovalResponse{ApprovalID: r.ApprovalID, CallID: r.CallID, Approved: true}))
	s := ai.StreamText(t.Context(), aitest.NewModel(aitest.Turn(aitest.Text("done"))), hist)
	for p := range s.Parts() {
		if _, ok := p.(ai.ToolResult); ok {
			break
		}
	}
	out, _ := s.Result()
	if d.runs.Load() != 1 {
		t.Fatalf("tool runs = %d, want 1", d.runs.Load())
	}
	if len(resultsOf(out.Messages)) != 1 {
		t.Errorf("Messages = %+v, want the finished tool result recorded", out.Messages)
	}
}

func TestStreamText_consumerStopKeepsToolResultOfStep(t *testing.T) {
	d := &deleter{}
	m := aitest.NewModel(toolTurn("delete", `{"id":"1"}`), aitest.Turn(aitest.Text("done")))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{Messages: []ai.Message{ai.User("x")}, Tools: []ai.Tool{d.tool(nil)}})
	for p := range s.Parts() {
		if _, ok := p.(ai.ToolResult); ok {
			break
		}
	}
	out, _ := s.Result()
	if len(resultsOf(out.Messages)) != 1 || len(out.Steps) != 1 {
		t.Errorf("Messages = %+v, Steps = %d; want the executed tool recorded", out.Messages, len(out.Steps))
	}
}

func TestGenerateText_cancelKeepsFinishedToolResults(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	finished := make(chan struct{})
	quick := ai.NewTool("quick", "x", func(context.Context, deleteIn) (string, error) {
		close(finished)
		return "ok", nil
	})
	slow := ai.NewTool("slow", "x", func(c context.Context, _ deleteIn) (string, error) {
		<-finished
		cancel()
		<-c.Done()
		return "", c.Err()
	})
	m := aitest.NewModel(aitest.Turn(aitest.ToolCall("quick", `{"id":"1"}`), aitest.ToolCall("slow", `{"id":"2"}`), aitest.Usage(100, 50)))
	res, err := ai.GenerateText(ctx, m, ai.TextRequest{Messages: []ai.Message{ai.User("x")}, Tools: []ai.Tool{quick, slow}})
	if err == nil {
		t.Fatal("GenerateText() error = nil, want cancellation")
	}
	var names []string
	for _, r := range resultsOf(res.Messages) {
		names = append(names, r.Name)
	}
	if len(names) != 2 || names[0] != "quick" || names[1] != "slow" {
		t.Errorf("result names in Messages = %v, want quick and slow", names)
	}
	if res.Usage.Input != 100 {
		t.Errorf("Usage.Input = %d, want 100", res.Usage.Input)
	}
}

func TestGenerateText_usageKeptWhenToolCancelsRun(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	tool := ai.NewTool("slow", "x", func(c context.Context, _ deleteIn) (string, error) {
		cancel()
		<-c.Done()
		return "", c.Err()
	})
	m := aitest.NewModel(aitest.Turn(aitest.ToolCall("slow", `{"id":"1"}`), aitest.Usage(100, 50)))
	res, err := ai.GenerateText(ctx, m, ai.TextRequest{Messages: []ai.Message{ai.User("x")}, Tools: []ai.Tool{tool}})
	if err == nil {
		t.Fatal("GenerateText() error = nil, want cancellation")
	}
	if res.Usage.Input != 100 || res.Usage.Output != 50 {
		t.Errorf("Usage = %+v, want the failed step's 100/50", res.Usage)
	}
	if len(res.Steps) != 0 {
		t.Errorf("Steps = %d, want 0", len(res.Steps))
	}
}

func TestApproval_autoDenyResultFollowsItsAssistantMessage(t *testing.T) {
	d := &deleter{}
	req, pending := pendingFixture(t, d)
	hist := resume(req, pending, ai.User("never mind, do something else"))
	m := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, m, hist)
	msgs := m.Calls()[0].Messages
	for i, msg := range msgs {
		for _, p := range msg.Content {
			if _, ok := p.(ai.ToolCall); !ok {
				continue
			}
			if i+1 >= len(msgs) || msgs[i+1].Role != ai.RoleTool || len(resultsOf(msgs[i+1:i+2])) != 1 {
				t.Errorf("message after the tool call = %+v, want its tool result", msgs[i+1:])
			}
		}
	}
	if d.runs.Load() != 0 {
		t.Errorf("tool runs = %d, want 0", d.runs.Load())
	}
}

func TestBatch_decodeChoiceAndObject(t *testing.T) {
	type pick struct {
		N int `json:"n"`
	}
	bm := aitest.NewBatchModel(aitest.Turn(aitest.Text(`{"result":"a"}`)), aitest.Turn(aitest.Text(`{"n":3}`)), aitest.Turn(aitest.Text(`{"result":"a"}`)), aitest.Turn(aitest.Text(`{"n":3}`)))
	items := []ai.BatchItem{
		{ID: "c", Request: ai.TextRequest{Messages: []ai.Message{ai.User("x")}, Output: ai.ChoiceOutput("a", "b")}},
		{ID: "o", Request: ai.TextRequest{Messages: []ai.Message{ai.User("y")}, Output: ai.ObjectOutput[pick]()}},
	}
	job, err := ai.StartBatch(t.Context(), bm, items)
	if err != nil {
		t.Fatal(err)
	}
	decode := func(job *ai.BatchJob) {
		t.Helper()
		seen := 0
		for r := range job.Results(t.Context()) {
			seen++
			switch r.ID {
			case "c":
				if v, err := ai.DecodeChoice[string](r.Result); err != nil || v != "a" {
					t.Errorf("DecodeChoice = %q, %v", v, err)
				}
			case "o":
				if v, err := ai.DecodeObject[pick](r.Result); err != nil || v.N != 3 {
					t.Errorf("DecodeObject = %+v, %v", v, err)
				}
			}
		}
		if seen != 2 {
			t.Errorf("results = %d, want 2", seen)
		}
	}
	decode(job)
	resumed := ai.ResumeBatch(bm, job.ID()).WithOutput("c", items[0].Request.Output).WithOutput("o", items[1].Request.Output)
	decode(resumed)
}
