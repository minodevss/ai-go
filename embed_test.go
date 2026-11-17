package ai_test

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func texts(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("doc-%d", i)
	}
	return out
}

func TestEmbed_chunking(t *testing.T) {
	tests := []struct {
		name      string
		maxBatch  int
		n         int
		wantSizes []int
	}{
		{name: "fits in one call", maxBatch: 10, n: 4, wantSizes: []int{4}},
		{name: "exact multiple", maxBatch: 3, n: 6, wantSizes: []int{3, 3}},
		{name: "remainder", maxBatch: 3, n: 7, wantSizes: []int{3, 3, 1}},
		{name: "unlimited", maxBatch: 0, n: 50, wantSizes: []int{50}},
		{name: "batch of one", maxBatch: 1, n: 3, wantSizes: []int{1, 1, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewEmbeddingModel(aitest.MaxBatch(tt.maxBatch))
			res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts(tt.n)})
			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			var sizes []int
			for _, c := range m.Calls() {
				sizes = append(sizes, len(c.Inputs))
			}
			if diff := cmp.Diff(tt.wantSizes, sizes); diff != "" {
				t.Errorf("batch sizes mismatch (-want +got):\n%s", diff)
			}
			if len(res.Embeddings) != tt.n {
				t.Fatalf("embeddings = %d, want %d", len(res.Embeddings), tt.n)
			}
			for i, v := range res.Embeddings {
				if want := float32(len(fmt.Sprintf("doc-%d", i))); v[0] != want {
					t.Errorf("Embeddings[%d] = %v, want [%v]: order not preserved", i, v, want)
				}
			}
			if res.Usage.Input != tt.n {
				t.Errorf("Usage.Input = %d, want %d", res.Usage.Input, tt.n)
			}
		})
	}
}

func TestEmbed_orderWithConcurrency(t *testing.T) {
	m := aitest.NewEmbeddingModel(aitest.MaxBatch(2), aitest.EmbedFunc(func(s string) []float32 { return []float32{float32(len(s))} }))
	in := []string{"a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"}
	res, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: in, Concurrency: 4})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	var got []float32
	for _, v := range res.Embeddings {
		got = append(got, v[0])
	}
	if diff := cmp.Diff([]float32{1, 2, 3, 4, 5, 6, 7}, got); diff != "" {
		t.Errorf("order mismatch (-want +got):\n%s", diff)
	}
}

type gatedEmbedder struct {
	running, peak atomic.Int32
}

func (*gatedEmbedder) Provider() string { return "test" }

func (*gatedEmbedder) ModelID() string { return "gated" }

func (*gatedEmbedder) MaxBatch() int { return 1 }

func (g *gatedEmbedder) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	cur := g.running.Add(1)
	defer g.running.Add(-1)
	for {
		old := g.peak.Load()
		if cur <= old || g.peak.CompareAndSwap(old, cur) {
			break
		}
	}
	select {
	case <-time.After(time.Second):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &ai.EmbedResponse{Embeddings: make([][]float32, len(c.Inputs))}, nil
}

func TestEmbed_concurrencyIsBounded(t *testing.T) {
	tests := []struct {
		name        string
		concurrency int
		wantPeak    int32
		wantElapsed time.Duration
	}{
		{name: "default is sequential", concurrency: 0, wantPeak: 1, wantElapsed: 8 * time.Second},
		{name: "two at a time", concurrency: 2, wantPeak: 2, wantElapsed: 4 * time.Second},
		{name: "more than batches", concurrency: 20, wantPeak: 8, wantElapsed: time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				g := &gatedEmbedder{}
				start := time.Now()
				if _, err := ai.Embed(t.Context(), g, ai.EmbedRequest{Texts: texts(8), Concurrency: tt.concurrency}); err != nil {
					t.Fatalf("Embed() error = %v", err)
				}
				if got := g.peak.Load(); got != tt.wantPeak {
					t.Errorf("peak = %d, want %d", got, tt.wantPeak)
				}
				if got := time.Since(start); got != tt.wantElapsed {
					t.Errorf("elapsed = %v, want %v", got, tt.wantElapsed)
				}
			})
		})
	}
}

func TestEmbed_inputValidation(t *testing.T) {
	tests := []struct {
		name string
		req  ai.EmbedRequest
	}{
		{name: "nothing set", req: ai.EmbedRequest{}},
		{name: "both set", req: ai.EmbedRequest{Texts: []string{"a"}, Inputs: []ai.Parts{{ai.Text{Text: "a"}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewEmbeddingModel()
			if _, err := ai.Embed(t.Context(), m, tt.req); err == nil {
				t.Error("Embed() error = nil, want an error")
			}
			if len(m.Calls()) != 0 {
				t.Error("model was called for an invalid request")
			}
		})
	}
}

func TestEmbed_multimodalInputsAndOptions(t *testing.T) {
	m := aitest.NewEmbeddingModel()
	inputs := []ai.Parts{{ai.Text{Text: "caption"}, &ai.File{MediaType: "image/png", Data: []byte{1}}}}
	_, err := ai.Embed(t.Context(), m, ai.EmbedRequest{
		Inputs:     inputs,
		InputType:  ai.InputQuery,
		Dimensions: 256,
		Headers:    map[string][]string{"X-K": {"v"}},
		Provider:   ai.ProviderData{}.WithRaw("voyage", rawJSON(`{"truncate":true}`)),
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	call := m.Calls()[0]
	if call.InputType != ai.InputQuery || call.Dimensions != 256 || call.Headers.Get("X-K") != "v" {
		t.Errorf("call options = %+v", call)
	}
	if _, ok := call.Inputs[0][1].(ai.File); !ok {
		t.Errorf("Inputs[0][1] = %T, want ai.File (pointers normalized)", call.Inputs[0][1])
	}
	if v, ok := ai.ProviderValue[map[string]bool](call.Provider, "voyage"); !ok || !v["truncate"] {
		t.Errorf("provider data lost: %v", call.Provider)
	}
}

func TestEmbed_errors(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("bad input")
		m := aitest.NewEmbeddingModel(aitest.MaxBatch(1), aitest.Fail(boom))
		_, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts(3)})
		if !errors.Is(err, boom) {
			t.Errorf("Embed() error = %v, want %v", err, boom)
		}
	})
}

func TestEmbed_retries(t *testing.T) {
	tests := []struct {
		name      string
		errs      []error
		retries   *int
		wantErr   bool
		wantCalls int
	}{
		{name: "retries a 503 once", errs: []error{apiErr(503)}, wantCalls: 2},
		{name: "gives up after the default", errs: []error{apiErr(503), apiErr(503), apiErr(503)}, wantErr: true, wantCalls: 3},
		{name: "MaxRetries 0", errs: []error{apiErr(503)}, retries: new(0), wantErr: true, wantCalls: 1},
		{name: "400 is not retried", errs: []error{apiErr(400)}, wantErr: true, wantCalls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewEmbeddingModel(aitest.Fail(tt.errs...))
				_, err := ai.Embed(t.Context(), m, ai.EmbedRequest{Texts: texts(2), MaxRetries: tt.retries})
				if (err != nil) != tt.wantErr {
					t.Errorf("error = %v, wantErr %v", err, tt.wantErr)
				}
				if got := len(m.Calls()); got != tt.wantCalls {
					t.Errorf("calls = %d, want %d", got, tt.wantCalls)
				}
			})
		})
	}
}

type shortEmbedder struct{ aitest.EmbeddingModel }

func (*shortEmbedder) Embed(context.Context, *ai.EmbedCall) (*ai.EmbedResponse, error) {
	return &ai.EmbedResponse{Embeddings: [][]float32{{1}}}, nil
}

func TestEmbed_countMismatch(t *testing.T) {
	if _, err := ai.Embed(t.Context(), &shortEmbedder{}, ai.EmbedRequest{Texts: texts(3)}); err == nil {
		t.Error("Embed() error = nil, want an error when the model returns too few vectors")
	}
}

func TestEmbed_canceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	m := aitest.NewEmbeddingModel(aitest.MaxBatch(1))
	if _, err := ai.Embed(ctx, m, ai.EmbedRequest{Texts: texts(3)}); !errors.Is(err, context.Canceled) {
		t.Errorf("Embed() error = %v, want context.Canceled", err)
	}
}

func TestRerank(t *testing.T) {
	docs := []string{"cats and dogs", "go concurrency patterns", "go generics tutorial", "baking bread"}
	tests := []struct {
		name string
		req  ai.RerankRequest
		want []int
	}{
		{name: "sorted by score", req: ai.RerankRequest{Query: "go generics", Documents: docs}, want: []int{2, 1, 0, 3}},
		{name: "top n", req: ai.RerankRequest{Query: "go generics", Documents: docs, TopN: 2}, want: []int{2, 1}},
		{name: "top n larger than docs", req: ai.RerankRequest{Query: "go", Documents: docs, TopN: 10}, want: []int{1, 2, 0, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := ai.Rerank(t.Context(), aitest.NewRerankModel(), tt.req)
			if err != nil {
				t.Fatalf("Rerank() error = %v", err)
			}
			var got []int
			for _, r := range res.Ranking {
				got = append(got, r.Index)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ranking mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRerank_noDocumentsSkipsModel(t *testing.T) {
	m := aitest.NewRerankModel()
	res, err := ai.Rerank(t.Context(), m, ai.RerankRequest{Query: "q"})
	if err != nil || len(res.Ranking) != 0 || len(m.Calls()) != 0 {
		t.Errorf("Rerank() = %+v, %v after %d calls; want an empty result without a call", res, err, len(m.Calls()))
	}
}

type badRanker struct{ ranking []ai.Ranked }

func (*badRanker) Provider() string { return "test" }

func (*badRanker) ModelID() string { return "bad" }

func (b *badRanker) Rerank(context.Context, *ai.RerankCall) (*ai.RerankResponse, error) {
	return &ai.RerankResponse{Ranking: b.ranking}, nil
}

func TestRerank_rejectsOutOfRangeIndex(t *testing.T) {
	for _, idx := range []int{-1, 2, 99} {
		t.Run(strconv.Itoa(idx), func(t *testing.T) {
			_, err := ai.Rerank(t.Context(), &badRanker{ranking: []ai.Ranked{{Index: idx, Score: 1}}}, ai.RerankRequest{Query: "q", Documents: []string{"a", "b"}})
			if err == nil {
				t.Error("Rerank() error = nil, want an error")
			}
		})
	}
}
