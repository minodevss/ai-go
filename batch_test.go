package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func batchItems(ids ...string) []ai.BatchItem {
	items := make([]ai.BatchItem, len(ids))
	for i, id := range ids {
		items[i] = ai.BatchItem{ID: id, Request: ai.TextRequest{Messages: []ai.Message{ai.User("q " + id)}}}
	}
	return items
}

func TestStartBatch_resultsByID(t *testing.T) {
	m := aitest.NewBatchModel(aitest.Turn(aitest.Text("first")), aitest.Turn(aitest.Text("second")), aitest.Turn(aitest.Text("third")), aitest.FailItems("b"))
	req := ai.TextRequest{
		Instructions: ai.Texts("be brief"),
		Messages:     []ai.Message{ai.User("q a")},
		Temperature:  new(0.5),
		Tools:        []ai.Tool{ai.NewClientTool[searchIn]("lookup", "client tool")},
	}
	job, err := ai.StartBatch(t.Context(), m, []ai.BatchItem{{ID: "a", Request: req}, {ID: "b", Request: req}, {ID: "c", Request: req}})
	if err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if job.ID() != "batch_1" {
		t.Errorf("ID() = %q, want batch_1", job.ID())
	}
	sent := m.Submitted()
	if len(sent) != 3 || sent[0].ID != "a" || sent[1].ID != "b" || *sent[0].Call.Temperature != 0.5 || len(sent[0].Call.Tools) != 1 || sent[0].Call.Instructions[0].Text != "be brief" {
		t.Errorf("submitted = %+v, want three calls built from the requests", sent)
	}
	if err := job.Wait(t.Context(), time.Second); err != nil {
		t.Fatalf("Wait() error = %v", err)
	}
	results := map[string]ai.BatchResult{}
	var order []string
	for r := range job.Results(t.Context()) {
		results[r.ID] = r
		order = append(order, r.ID)
	}
	if err := job.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(results) != 3 || results["b"].Err == nil || results["b"].Result != nil {
		t.Fatalf("results = %+v, want b to fail", results)
	}
	if order[0] == "a" {
		t.Errorf("results came in request order %v; the fake returns them reversed to prove callers must match by ID", order)
	}
	for id, want := range map[string]string{"a": "first", "c": "second"} {
		if got := results[id].Result; got == nil || got.Text != want || len(got.Steps) != 1 || len(got.Messages) != 1 {
			t.Errorf("result %s = %+v, want text %q with one step and message", id, got, want)
		}
	}
}

func TestStartBatch_rejectsLoopsAndStreams(t *testing.T) {
	execTool := ai.NewTool("exec", "runs", func(context.Context, searchIn) (searchOut, error) { return searchOut{}, nil })
	base := func() ai.TextRequest { return ai.TextRequest{Messages: []ai.Message{ai.User("q")}} }
	tests := []struct {
		name   string
		mutate func(*ai.TextRequest)
	}{
		{name: "tool Execute", mutate: func(r *ai.TextRequest) { r.Tools = []ai.Tool{execTool} }},
		{name: "PrepareStep", mutate: func(r *ai.TextRequest) {
			r.PrepareStep = func(context.Context, ai.StepState, *ai.StepRequest) error { return nil }
		}},
		{name: "OnStepEnd", mutate: func(r *ai.TextRequest) { r.OnStepEnd = func(context.Context, ai.Step) error { return nil } }},
		{name: "StopWhen", mutate: func(r *ai.TextRequest) { r.StopWhen = []ai.StopCondition{ai.MaxTotalTokens(1)} }},
		{name: "MaxSteps", mutate: func(r *ai.TextRequest) { r.MaxSteps = 3 }},
		{name: "approval response", mutate: func(r *ai.TextRequest) {
			r.Messages = append(r.Messages, ai.Message{Role: ai.RoleTool, Content: ai.Parts{ai.ToolApprovalResponse{ApprovalID: "x"}}})
		}},
		{name: "duplicate tools", mutate: func(r *ai.TextRequest) {
			r.Tools = []ai.Tool{ai.NewClientTool[searchIn]("t", ""), ai.NewClientTool[searchIn]("t", "")}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewBatchModel()
			req := base()
			tt.mutate(&req)
			_, err := ai.StartBatch(t.Context(), m, []ai.BatchItem{{ID: "a", Request: req}})
			if err == nil || len(m.Submitted()) != 0 {
				t.Errorf("StartBatch() error = %v with %d submitted, want a rejection before submitting", err, len(m.Submitted()))
			}
		})
	}
}

func TestStartBatch_validatesItems(t *testing.T) {
	tests := []struct {
		name  string
		items []ai.BatchItem
		want  string
	}{
		{name: "no items", want: "no items"},
		{name: "empty id", items: batchItems(""), want: "no ID"},
		{name: "duplicate id", items: batchItems("a", "a"), want: "duplicate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.StartBatch(t.Context(), aitest.NewBatchModel(), tt.items)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("StartBatch() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestStartBatch_needsConcreteProvider(t *testing.T) {
	concrete := aitest.NewBatchModel()
	plain := aitest.NewModel()
	tests := []struct {
		name  string
		model ai.LanguageModel
	}{
		{name: "fallback", model: ai.Fallback(concrete, plain)},
		{name: "wrapped", model: ai.WrapModel(concrete, ai.Middleware{})},
		{name: "model without batches", model: plain},
		{name: "nil", model: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.StartBatch(t.Context(), tt.model, batchItems("a"))
			if !errors.Is(err, ai.ErrUnsupported) || len(concrete.Submitted()) != 0 {
				t.Errorf("StartBatch() error = %v, want ErrUnsupported without submitting", err)
			}
		})
	}
}

func TestBatchJob_waitPollsUntilDone(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewBatchModel(aitest.States(ai.BatchValidating, ai.BatchRunning, ai.BatchRunning, ai.BatchFinalizing, ai.BatchCompleted))
		job, err := ai.StartBatch(t.Context(), m, batchItems("a"))
		if err != nil {
			t.Fatalf("StartBatch() error = %v", err)
		}
		start := time.Now()
		if err := job.Wait(t.Context(), 10*time.Second); err != nil {
			t.Fatalf("Wait() error = %v", err)
		}
		if got := time.Since(start); got != 40*time.Second {
			t.Errorf("Wait() took %v, want four 10s intervals", got)
		}
		st, err := job.Status(t.Context())
		if err != nil || st.State != ai.BatchCompleted || st.Total != 1 {
			t.Errorf("Status() = %+v, %v, want completed with 1 request", st, err)
		}
	})
}

func TestBatchJob_waitDefaultInterval(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewBatchModel(aitest.States(ai.BatchRunning, ai.BatchCompleted))
		job, _ := ai.StartBatch(t.Context(), m, batchItems("a"))
		start := time.Now()
		if err := job.Wait(t.Context(), 0); err != nil || time.Since(start) != 30*time.Second {
			t.Errorf("Wait() = %v after %v, want nil after the 30s default", err, time.Since(start))
		}
	})
}

func TestBatchJob_waitOutcomes(t *testing.T) {
	tests := []struct {
		name    string
		state   ai.BatchState
		wantErr bool
	}{
		{name: "completed", state: ai.BatchCompleted},
		{name: "expired keeps partial results readable", state: ai.BatchExpired},
		{name: "canceled", state: ai.BatchCanceled},
		{name: "failed", state: ai.BatchFailed, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewBatchModel(aitest.States(tt.state))
			job, _ := ai.StartBatch(t.Context(), m, batchItems("a"))
			err := job.Wait(t.Context(), time.Second)
			if _, isJob := errors.AsType[*ai.JobError](err); (err != nil) != tt.wantErr || isJob != tt.wantErr {
				t.Errorf("Wait() error = %v, want error=%v", err, tt.wantErr)
			}
		})
	}
}

func TestBatchJob_waitStopsWithContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewBatchModel(aitest.States(ai.BatchRunning))
		job, _ := ai.StartBatch(t.Context(), m, batchItems("a"))
		ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
		defer cancel()
		if err := job.Wait(ctx, 20*time.Second); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Wait() error = %v, want the deadline", err)
		}
	})
}

func TestBatchJob_cancel(t *testing.T) {
	m := aitest.NewBatchModel(aitest.States(ai.BatchRunning))
	job, _ := ai.StartBatch(t.Context(), m, batchItems("a"))
	if err := job.Cancel(t.Context()); err != nil || !m.Canceled() {
		t.Fatalf("Cancel() = %v, canceled = %v, want a canceled batch", err, m.Canceled())
	}
	if err := job.Wait(t.Context(), time.Second); err != nil {
		t.Errorf("Wait() after cancel = %v, want nil", err)
	}
}

func TestResumeBatch(t *testing.T) {
	m := aitest.NewBatchModel()
	job := ai.ResumeBatch(m, "batch_9")
	if job.ID() != "batch_9" {
		t.Errorf("ID() = %q, want batch_9", job.ID())
	}
	if st, err := job.Status(t.Context()); err != nil || st.State != ai.BatchCompleted {
		t.Errorf("Status() = %+v, %v, want completed", st, err)
	}
	none := ai.ResumeBatch(aitest.NewModel(), "x")
	if _, err := none.Status(t.Context()); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("Status() on a model without batches error = %v, want ErrUnsupported", err)
	}
	for range none.Results(t.Context()) {
		t.Error("Results() yielded for a model without batches")
	}
	if !errors.Is(none.Err(), ai.ErrUnsupported) || !errors.Is(none.Cancel(t.Context()), ai.ErrUnsupported) {
		t.Error("Err() and Cancel() on a model without batches must report ErrUnsupported")
	}
}

func TestBatchJob_statusRetriesTransientErrors(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := &flakyBatches{BatchModel: aitest.NewBatchModel(), failures: 2}
		job, err := ai.StartBatch(t.Context(), m, batchItems("a"))
		if err != nil {
			t.Fatalf("StartBatch() error = %v", err)
		}
		if _, err := job.Status(t.Context()); err != nil || m.statusCalls != 3 {
			t.Errorf("Status() error = %v after %d calls, want success on the third", err, m.statusCalls)
		}
	})
}

func TestStartBatch_retriesOnlyRateLimits(t *testing.T) {
	tests := []struct {
		name         string
		err          error
		wantAttempts int
	}{
		{name: "rate limit is retried", err: &ai.APIError{StatusCode: 429, Code: "rate_limit_exceeded"}, wantAttempts: 2},
		{name: "server error is not retried because the batch may exist", err: apiErr(500), wantAttempts: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := &flakyBatches{BatchModel: aitest.NewBatchModel(), startFailure: tt.err}
				_, _ = ai.StartBatch(t.Context(), m, batchItems("a"))
				if m.startCalls != tt.wantAttempts {
					t.Errorf("StartBatch calls = %d, want %d", m.startCalls, tt.wantAttempts)
				}
			})
		})
	}
}

type flakyBatches struct {
	*aitest.BatchModel
	failures     int
	statusCalls  int
	startFailure error
	startCalls   int
}

func (f *flakyBatches) StartBatch(ctx context.Context, calls []ai.BatchCall) (string, error) {
	f.startCalls++
	if f.startFailure != nil && f.startCalls == 1 {
		return "", f.startFailure
	}
	return f.BatchModel.StartBatch(ctx, calls)
}

func (f *flakyBatches) BatchStatus(ctx context.Context, id string) (*ai.BatchStatus, error) {
	f.statusCalls++
	if f.statusCalls <= f.failures {
		return nil, apiErr(503)
	}
	return f.BatchModel.BatchStatus(ctx, id)
}
