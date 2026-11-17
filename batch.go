package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"time"

	"github.com/minodevss/ai-go/internal/retry"
)

const defaultBatchInterval = 30 * time.Second

type BatchItem struct {
	ID      string
	Request TextRequest
}

type BatchResult struct {
	ID     string
	Result *TextResult
	Err    error
}

// BatchJob is not safe for concurrent use. Err returns the error that ended the latest Results iteration.
type BatchJob struct {
	batches Batches
	id      string
	outputs map[string]Output
	err     error
}

// StartBatch submits single-step requests to a provider's batch API. The model must itself implement Batches: Fallback and wrapped models are rejected, because their routing and middleware cannot run inside a provider batch. Requests with tool Execute functions, PrepareStep, OnStepEnd, StopWhen or MaxSteps above 1 are rejected, because a batch cannot run a loop.
func StartBatch(ctx context.Context, model LanguageModel, items []BatchItem) (*BatchJob, error) {
	batches, ok := model.(Batches)
	if !ok {
		return nil, fmt.Errorf("%w: batches need a concrete provider model, not %T", ErrUnsupported, model)
	}
	calls, err := batchCalls(model, items)
	if err != nil {
		return nil, err
	}
	canRetry := func(err error) bool { return errors.Is(err, ErrRateLimit) }
	id, err := withRetry(ctx, defaultMaxRetries, canRetry, func(ctx context.Context) (string, error) {
		return batches.StartBatch(ctx, calls)
	})
	if err != nil {
		return nil, err
	}
	if id == "" {
		return nil, errors.New("ai: batch: provider returned no batch id")
	}
	outputs := make(map[string]Output, len(items))
	for _, item := range items {
		if item.Request.Output != nil {
			outputs[item.ID] = item.Request.Output
		}
	}
	return &BatchJob{batches: batches, id: id, outputs: outputs}, nil
}

// ResumeBatch returns a handle to a batch started earlier. Call WithOutput for each item whose result goes through DecodeObject or DecodeChoice.
func ResumeBatch(model LanguageModel, id string) *BatchJob {
	batches, _ := model.(Batches)
	return &BatchJob{batches: batches, id: id}
}

// WithOutput sets the Output that DecodeObject and DecodeChoice use for the result of item id, and returns j.
func (j *BatchJob) WithOutput(id string, out Output) *BatchJob {
	if j.outputs == nil {
		j.outputs = map[string]Output{}
	}
	j.outputs[id] = out
	return j
}

func batchCalls(model LanguageModel, items []BatchItem) ([]BatchCall, error) {
	if len(items) == 0 {
		return nil, errors.New("ai: batch: no items")
	}
	seen := make(map[string]bool, len(items))
	calls := make([]BatchCall, len(items))
	for i, item := range items {
		if item.ID == "" {
			return nil, fmt.Errorf("ai: batch: item %d has no ID", i)
		}
		if seen[item.ID] {
			return nil, fmt.Errorf("ai: batch: duplicate item ID %q", item.ID)
		}
		seen[item.ID] = true
		if err := checkSingleStep(item.Request); err != nil {
			return nil, fmt.Errorf("ai: batch: item %q: %w", item.ID, err)
		}
		r := newRunner(model, item.Request.Clone(), false, func(StreamPart) bool { return true })
		if err := r.setup(); err != nil {
			return nil, fmt.Errorf("ai: batch: item %q: %w", item.ID, err)
		}
		calls[i] = BatchCall{ID: item.ID, Call: r.call(r.stepRequest())}
	}
	return calls, nil
}

func checkSingleStep(req TextRequest) error {
	switch {
	case slices.ContainsFunc(req.Tools, func(t Tool) bool { return t.Execute != nil }):
		return errors.New("ai: tools with Execute need a loop, which a batch cannot run")
	case req.PrepareStep != nil, req.OnStepEnd != nil, len(req.StopWhen) > 0:
		return errors.New("ai: step hooks and stop conditions need a loop, which a batch cannot run")
	case req.MaxSteps > 1:
		return errors.New("ai: MaxSteps above 1 needs a loop, which a batch cannot run")
	}
	for _, m := range req.Messages {
		for _, p := range m.Content {
			switch p.(type) {
			case ToolApprovalRequest, ToolApprovalResponse, *ToolApprovalRequest, *ToolApprovalResponse:
				return errors.New("ai: tool approvals need a loop, which a batch cannot run")
			default:
			}
		}
	}
	return nil
}

func (j *BatchJob) ID() string { return j.id }

func (j *BatchJob) Status(ctx context.Context) (BatchStatus, error) {
	if j.batches == nil {
		return BatchStatus{}, errNoBatches
	}
	st, err := withRetry(ctx, defaultMaxRetries, isRetryable, func(ctx context.Context) (*BatchStatus, error) {
		return j.batches.BatchStatus(ctx, j.id)
	})
	if err != nil {
		return BatchStatus{}, err
	}
	if st == nil {
		return BatchStatus{}, errors.New("ai: batch: provider returned no status")
	}
	return *st, nil
}

// Wait polls every interval (0 = 30s) until the batch is done. It returns nil for every final state except BatchFailed, so partial results of expired or canceled batches stay readable.
func (j *BatchJob) Wait(ctx context.Context, interval time.Duration) error {
	if interval <= 0 {
		interval = defaultBatchInterval
	}
	for {
		st, err := j.Status(ctx)
		if err != nil {
			return err
		}
		if st.State == BatchFailed {
			msg := st.Message
			if msg == "" {
				msg = "ai: batch failed"
			}
			return &JobError{Kind: "batch", ID: j.id, Err: errors.New(msg)}
		}
		if st.State.Done() {
			return nil
		}
		if err := retry.Sleep(ctx, interval); err != nil {
			return err
		}
	}
}

// Results yields one result per request line the provider returned, in provider order, not request order. Match by ID.
func (j *BatchJob) Results(ctx context.Context) iter.Seq[BatchResult] {
	return func(yield func(BatchResult) bool) {
		j.err = nil
		if j.batches == nil {
			j.err = errNoBatches
			return
		}
		for e, err := range j.batches.BatchResults(ctx, j.id) {
			if err != nil {
				j.err = err
				return
			}
			res := BatchResult{ID: e.ID, Err: e.Err}
			if e.Response != nil {
				res.Result = batchText(e.Response)
				res.Result.output = j.outputs[e.ID]
			}
			if !yield(res) {
				return
			}
		}
	}
}

func (j *BatchJob) Err() error { return j.err }

func (j *BatchJob) Cancel(ctx context.Context) error {
	if j.batches == nil {
		return errNoBatches
	}
	_, err := withRetry(ctx, defaultMaxRetries, isRetryable, func(ctx context.Context) (struct{}, error) {
		return struct{}{}, j.batches.CancelBatch(ctx, j.id)
	})
	return err
}

var errNoBatches = fmt.Errorf("%w: the model does not support batches", ErrUnsupported)

func batchText(resp *Response) *TextResult {
	content := normalizeParts(resp.Content)
	calls := toolCallsOf(content)
	step := Step{
		Content:      content,
		ToolCalls:    calls,
		Usage:        resp.Usage,
		FinishReason: resp.FinishReason,
		FinishRaw:    resp.FinishRaw,
		Warnings:     resp.Warnings,
		Provider:     resp.Provider,
		Header:       resp.Header,
	}
	res := &TextResult{
		Text:         joinText(content),
		Content:      content,
		Steps:        []Step{step},
		Usage:        resp.Usage,
		FinishReason: resp.FinishReason,
		FinishRaw:    resp.FinishRaw,
		Warnings:     slices.Clone(resp.Warnings),
		Provider:     resp.Provider,
	}
	if len(content) > 0 {
		res.Messages = []Message{{Role: RoleAssistant, Content: slices.Clone(content), Provider: resp.Provider.clone()}}
	}
	return res
}
