package aitest

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"

	"github.com/minodevss/ai-go"
)

type batchOptionFunc func(*BatchModel)

func (f batchOptionFunc) applyBatch(m *BatchModel) { f(m) }

func (t TurnScript) applyBatch(m *BatchModel) { m.model.turns = append(m.model.turns, t) }

// States scripts the states BatchStatus reports, one per call; the last repeats.
func States(states ...ai.BatchState) BatchOption {
	return batchOptionFunc(func(m *BatchModel) { m.states = states })
}

// FailItems makes the results of the given item ids carry an error instead of a response.
func FailItems(ids ...string) BatchOption {
	return batchOptionFunc(func(m *BatchModel) {
		for _, id := range ids {
			m.failing[id] = true
		}
	})
}

// BatchModel is a language model with the ai.Batches capability. Request i of a batch gets turn i, produced when the results are read.
type BatchModel struct {
	mu       sync.Mutex
	model    *Model
	states   []ai.BatchState
	failing  map[string]bool
	calls    []ai.BatchCall
	polls    int
	canceled bool
	failures failures
}

func NewBatchModel(opts ...BatchOption) *BatchModel {
	m := &BatchModel{model: NewModel(), failing: map[string]bool{}, states: []ai.BatchState{ai.BatchCompleted}}
	for _, opt := range opts {
		opt.applyBatch(m)
	}
	return m
}

func (m *BatchModel) Provider() string { return "aitest" }

func (m *BatchModel) ModelID() string { return "scripted-batch" }

func (m *BatchModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	return m.model.Generate(ctx, c)
}

func (m *BatchModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	return m.model.Stream(ctx, c)
}

func (m *BatchModel) Submitted() []ai.BatchCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.BatchCall(nil), m.calls...)
}

func (m *BatchModel) Canceled() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.canceled
}

func (m *BatchModel) StartBatch(_ context.Context, calls []ai.BatchCall) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.failures.next(); err != nil {
		return "", err
	}
	m.calls = append([]ai.BatchCall(nil), calls...)
	return "batch_1", nil
}

func (m *BatchModel) BatchStatus(_ context.Context, _ string) (*ai.BatchStatus, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	state := m.states[min(m.polls, len(m.states)-1)]
	m.polls++
	if m.canceled {
		state = ai.BatchCanceled
	}
	return &ai.BatchStatus{State: state, Total: len(m.calls)}, nil
}

func (m *BatchModel) BatchResults(ctx context.Context, _ string) iter.Seq2[ai.BatchEntry, error] {
	m.mu.Lock()
	calls := append([]ai.BatchCall(nil), m.calls...)
	m.mu.Unlock()
	return func(yield func(ai.BatchEntry, error) bool) {
		entries := make([]ai.BatchEntry, len(calls))
		for i, call := range calls {
			entries[i].ID = call.ID
			if m.failing[call.ID] {
				entries[i].Err = fmt.Errorf("aitest: item %s failed", call.ID)
				continue
			}
			resp, err := m.model.Generate(ctx, call.Call)
			if err != nil {
				yield(ai.BatchEntry{}, err)
				return
			}
			entries[i].Response = resp
		}
		for _, e := range slices.Backward(entries) {
			if !yield(e, nil) {
				return
			}
		}
	}
}

func (m *BatchModel) CancelBatch(context.Context, string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.canceled = true
	return nil
}

var (
	_ ai.LanguageModel = (*BatchModel)(nil)
	_ ai.Batches       = (*BatchModel)(nil)
)
