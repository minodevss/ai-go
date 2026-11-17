package aitest

import (
	"context"
	"strings"
	"sync"

	"github.com/minodevss/ai-go"
)

type embeddingOptionFunc func(*EmbeddingModel)

func (f embeddingOptionFunc) applyEmbedding(m *EmbeddingModel) { f(m) }

func MaxBatch(n int) EmbeddingOption {
	return embeddingOptionFunc(func(m *EmbeddingModel) { m.maxBatch = n })
}

func EmbedFunc(f func(text string) []float32) EmbeddingOption {
	return embeddingOptionFunc(func(m *EmbeddingModel) { m.embed = f })
}

type EmbeddingModel struct {
	mu       sync.Mutex
	maxBatch int
	embed    func(string) []float32
	calls    []ai.EmbedCall
	failures failures
}

func NewEmbeddingModel(opts ...EmbeddingOption) *EmbeddingModel {
	m := &EmbeddingModel{embed: func(text string) []float32 { return []float32{float32(len(text))} }}
	for _, opt := range opts {
		opt.applyEmbedding(m)
	}
	return m
}

func (m *EmbeddingModel) Provider() string { return "aitest" }

func (m *EmbeddingModel) ModelID() string { return "scripted-embedding" }

func (m *EmbeddingModel) MaxBatch() int { return m.maxBatch }

func (m *EmbeddingModel) Calls() []ai.EmbedCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.EmbedCall(nil), m.calls...)
}

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	m.mu.Lock()
	m.calls = append(m.calls, *c)
	failure := m.failures.next()
	m.mu.Unlock()
	if failure != nil {
		return nil, failure
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, len(c.Inputs))
	for i, in := range c.Inputs {
		var text strings.Builder
		for _, p := range in {
			if t, ok := p.(ai.Text); ok {
				text.WriteString(t.Text)
			}
		}
		out[i] = m.embed(text.String())
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: ai.Usage{Input: len(c.Inputs)}}, nil
}

type RerankModel struct {
	mu       sync.Mutex
	calls    []ai.RerankCall
	failures failures
}

func NewRerankModel(opts ...RerankOption) *RerankModel {
	m := &RerankModel{}
	for _, opt := range opts {
		opt.applyRerank(m)
	}
	return m
}

func (m *RerankModel) Provider() string { return "aitest" }

func (m *RerankModel) ModelID() string { return "scripted-rerank" }

func (m *RerankModel) Calls() []ai.RerankCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.RerankCall(nil), m.calls...)
}

// Rerank scores each document by the share of query words it contains.
func (m *RerankModel) Rerank(_ context.Context, c *ai.RerankCall) (*ai.RerankResponse, error) {
	m.mu.Lock()
	m.calls = append(m.calls, *c)
	failure := m.failures.next()
	m.mu.Unlock()
	if failure != nil {
		return nil, failure
	}
	words := strings.Fields(strings.ToLower(c.Query))
	ranking := make([]ai.Ranked, len(c.Documents))
	for i, doc := range c.Documents {
		lower := strings.ToLower(doc)
		hits := 0
		for _, w := range words {
			if strings.Contains(lower, w) {
				hits++
			}
		}
		score := 0.0
		if len(words) > 0 {
			score = float64(hits) / float64(len(words))
		}
		ranking[i] = ai.Ranked{Index: i, Score: score}
	}
	return &ai.RerankResponse{Ranking: ranking}, nil
}

var (
	_ ai.EmbeddingModel = (*EmbeddingModel)(nil)
	_ ai.RerankModel    = (*RerankModel)(nil)
)
