package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
)

type fallbackModel struct {
	models []LanguageModel
}

// Fallback tries each model in order and moves on only after a retryable or category error (rate limit, quota, auth, context length) before any output.
func Fallback(models ...LanguageModel) LanguageModel {
	return &fallbackModel{models: append([]LanguageModel(nil), models...)}
}

func (f *fallbackModel) Provider() string { return "fallback" }

func (f *fallbackModel) ModelID() string {
	ids := make([]string, len(f.models))
	for i, m := range f.models {
		ids[i] = m.Provider() + ":" + m.ModelID()
	}
	return strings.Join(ids, ",")
}

func (f *fallbackModel) Unwrap() []LanguageModel { return append([]LanguageModel(nil), f.models...) }

func (f *fallbackModel) Generate(ctx context.Context, c *Call) (*Response, error) {
	var errs []error
	for _, m := range f.models {
		resp, err := m.Generate(ctx, c.Clone())
		if err == nil {
			return resp, nil
		}
		if ctx.Err() != nil || !shouldFallback(err) {
			return nil, err
		}
		errs = append(errs, err)
	}
	return nil, exhausted(errs)
}

func (f *fallbackModel) Stream(ctx context.Context, c *Call) (*StreamResponse, error) {
	var errs []error
	for _, m := range f.models {
		resp, err := openStream(ctx, m, c.Clone())
		if err == nil {
			return resp, nil
		}
		if ctx.Err() != nil || !shouldFallback(err) {
			return nil, err
		}
		errs = append(errs, err)
	}
	return nil, exhausted(errs)
}

func exhausted(errs []error) error {
	if len(errs) == 0 {
		return errors.New("ai: fallback has no models")
	}
	return fmt.Errorf("ai: fallback: all models failed: %w", errors.Join(errs...))
}

func shouldFallback(err error) bool {
	if isRetryable(err) {
		return true
	}
	return errors.Is(err, ErrRateLimit) || errors.Is(err, ErrQuota) || errors.Is(err, ErrAuth) || errors.Is(err, ErrContextTooLong)
}

func openStream(ctx context.Context, m LanguageModel, c *Call) (*StreamResponse, error) {
	resp, err := m.Stream(ctx, c)
	if err != nil {
		return nil, err
	}
	if resp == nil || resp.Parts == nil {
		return nil, errors.New("ai: model returned no stream")
	}
	next, stop := iter.Pull2(resp.Parts)
	first, ferr, ok := next()
	if ferr != nil {
		stop()
		return nil, ferr
	}
	parts := func(yield func(StreamPart, error) bool) {
		defer stop()
		if !ok {
			return
		}
		if !yield(first, nil) {
			return
		}
		for {
			part, err, more := next()
			if !more || !yield(part, err) {
				return
			}
			if err != nil {
				return
			}
		}
	}
	return &StreamResponse{Parts: parts, Header: resp.Header}, nil
}
