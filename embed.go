package ai

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
)

// EmbedRequest takes exactly one of Texts or Inputs. Inputs are split into batches of the model's MaxBatch and sent Concurrency batches at a time (0 = 1).
type EmbedRequest struct {
	Texts       []string
	Inputs      []Parts
	InputType   InputType
	Dimensions  int
	Concurrency int
	MaxRetries  *int
	Headers     http.Header
	Provider    ProviderData
}

type EmbedResult struct {
	Embeddings [][]float32
	Usage      Usage
	Warnings   []Warning
}

func Embed(ctx context.Context, m EmbeddingModel, req EmbedRequest) (*EmbedResult, error) {
	inputs, err := embedInputs(req)
	if err != nil {
		return nil, err
	}
	size := m.MaxBatch()
	if size <= 0 {
		size = len(inputs)
	}
	chunks := slices.Collect(slices.Chunk(inputs, size))
	responses := make([]*EmbedResponse, len(chunks))
	errs := make([]error, len(chunks))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	conc := max(req.Concurrency, 1)
	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	for i, chunk := range chunks {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			errs[i] = ctx.Err()
			continue
		}
		wg.Go(func() {
			defer func() { <-sem }()
			responses[i], errs[i] = embedChunk(ctx, m, req, chunk)
			if errs[i] != nil {
				cancel()
			}
		})
	}
	wg.Wait()
	if err := firstError(errs); err != nil {
		return nil, err
	}
	res := &EmbedResult{Embeddings: make([][]float32, 0, len(inputs))}
	for _, resp := range responses {
		res.Embeddings = append(res.Embeddings, resp.Embeddings...)
		res.Usage = res.Usage.add(resp.Usage)
		res.Warnings = append(res.Warnings, resp.Warnings...)
	}
	return res, nil
}

func embedInputs(req EmbedRequest) ([]Parts, error) {
	switch {
	case len(req.Texts) > 0 && len(req.Inputs) > 0:
		return nil, errors.New("ai: embed: set only one of Texts or Inputs")
	case len(req.Texts) > 0:
		inputs := make([]Parts, len(req.Texts))
		for i, t := range req.Texts {
			inputs[i] = Parts{Text{Text: t}}
		}
		return inputs, nil
	case len(req.Inputs) > 0:
		inputs := make([]Parts, len(req.Inputs))
		for i, in := range req.Inputs {
			inputs[i] = normalizeParts(in)
		}
		return inputs, nil
	default:
		return nil, errors.New("ai: embed: no inputs")
	}
}

func embedChunk(ctx context.Context, m EmbeddingModel, req EmbedRequest, chunk []Parts) (*EmbedResponse, error) {
	call := &EmbedCall{
		Inputs:     slices.Clone(chunk),
		InputType:  req.InputType,
		Dimensions: req.Dimensions,
		Headers:    req.Headers.Clone(),
		Provider:   req.Provider.clone(),
	}
	resp, err := withRetry(ctx, retriesOf(req.MaxRetries), isRetryable, func(ctx context.Context) (*EmbedResponse, error) {
		return m.Embed(ctx, call)
	})
	if err != nil {
		return nil, err
	}
	if resp == nil || len(resp.Embeddings) != len(chunk) {
		return nil, fmt.Errorf("ai: embed: model returned %d embeddings for %d inputs", embeddingCount(resp), len(chunk))
	}
	return resp, nil
}

func embeddingCount(resp *EmbedResponse) int {
	if resp == nil {
		return 0
	}
	return len(resp.Embeddings)
}

func firstError(errs []error) error {
	var canceled error
	for _, err := range errs {
		switch {
		case err == nil:
		case errors.Is(err, context.Canceled):
			canceled = cmp.Or(canceled, err)
		default:
			return err
		}
	}
	return canceled
}

func retriesOf(p *int) int {
	if p == nil {
		return defaultMaxRetries
	}
	return max(*p, 0)
}

type RerankRequest struct {
	Query      string
	Documents  []string
	TopN       int
	MaxRetries *int
	Headers    http.Header
	Provider   ProviderData
}

type RerankResult struct {
	Ranking  []Ranked
	Usage    Usage
	Warnings []Warning
}

// Rerank returns the ranking sorted by descending score, cut to TopN when TopN is set.
func Rerank(ctx context.Context, m RerankModel, req RerankRequest) (*RerankResult, error) {
	if len(req.Documents) == 0 {
		return &RerankResult{}, nil
	}
	call := &RerankCall{
		Query:     req.Query,
		Documents: slices.Clone(req.Documents),
		TopN:      req.TopN,
		Headers:   req.Headers.Clone(),
		Provider:  req.Provider.clone(),
	}
	resp, err := withRetry(ctx, retriesOf(req.MaxRetries), isRetryable, func(ctx context.Context) (*RerankResponse, error) {
		return m.Rerank(ctx, call)
	})
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("ai: rerank: model returned no response")
	}
	ranking := slices.Clone(resp.Ranking)
	for _, r := range ranking {
		if r.Index < 0 || r.Index >= len(req.Documents) {
			return nil, fmt.Errorf("ai: rerank: model returned index %d for %d documents", r.Index, len(req.Documents))
		}
	}
	slices.SortStableFunc(ranking, func(a, b Ranked) int { return cmp.Compare(b.Score, a.Score) })
	if req.TopN > 0 && len(ranking) > req.TopN {
		ranking = ranking[:req.TopN]
	}
	return &RerankResult{Ranking: ranking, Usage: resp.Usage, Warnings: resp.Warnings}, nil
}
