package ai

import (
	"context"
	"errors"
	"time"

	"github.com/minodevss/ai-go/internal/retry"
)

const (
	defaultMaxRetries = 2
	defaultMaxSteps   = 20
	defaultToolLimit  = 8
)

type timeoutError struct {
	msg   string
	retry bool
}

func (e *timeoutError) Error() string { return e.msg }

func (e *timeoutError) Is(target error) bool { return target == context.DeadlineExceeded }

func (e *timeoutError) Timeout() bool { return true }

func isRetryable(err error) bool {
	if apiErr, ok := errors.AsType[*APIError](err); ok {
		return apiErr.Retryable()
	}
	if te, ok := errors.AsType[*timeoutError](err); ok {
		return te.retry
	}
	return false
}

func retryHint(err error) time.Duration {
	if apiErr, ok := errors.AsType[*APIError](err); ok {
		return apiErr.RetryAfter
	}
	return 0
}

func withRetry[T any](ctx context.Context, maxRetries int, canRetry func(error) bool, op func(context.Context) (T, error)) (T, error) {
	var errs []error
	for attempt := 0; ; attempt++ {
		v, err := op(ctx)
		if err == nil {
			return v, nil
		}
		errs = append(errs, err)
		if ctx.Err() != nil || attempt >= maxRetries || !canRetry(err) {
			return *new(T), joinAttempts(errs)
		}
		if err := retry.Sleep(ctx, retry.Backoff(attempt, retryHint(err))); err != nil {
			return *new(T), joinAttempts(append(errs, err))
		}
	}
}

func joinAttempts(errs []error) error {
	if len(errs) == 1 {
		return errs[0]
	}
	return &RetryError{Errs: errs}
}

func mapTimeout(ctx context.Context, err error) error {
	if te, ok := errors.AsType[*timeoutError](context.Cause(ctx)); ok {
		return te
	}
	return err
}
