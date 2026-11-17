package mediaio

import (
	"context"
	"errors"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/retry"
)

const (
	DefaultPollInterval = time.Second
	MaxPollInterval     = 10 * time.Second
	maxPollFailures     = 3
)

// Backoff is the wait between polls of an image job: Initial, growing by half each poll up to Max. Zero values use the package defaults.
type Backoff struct {
	Initial time.Duration
	Max     time.Duration
}

// Poll calls step until it reports done, an error that is not transient, or ctx ends. The job id stays in the returned *ai.JobError.
// Transient API errors are retried inside the loop and, when they persist, returned without their *ai.APIError so that ai.GenerateImage does not resubmit a job that already exists.
func (b Backoff) Poll(ctx context.Context, kind, id string, step func(context.Context) (bool, error)) error {
	interval, limit := b.Initial, b.Max
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	if limit <= 0 {
		limit = MaxPollInterval
	}
	failures := 0
	for {
		done, err := step(ctx)
		switch {
		case err == nil && done:
			return nil
		case err == nil:
			failures = 0
		case ctx.Err() != nil:
			return &ai.JobError{Kind: kind, ID: id, Err: ctx.Err()}
		case !transient(err):
			return &ai.JobError{Kind: kind, ID: id, Err: err}
		default:
			if failures++; failures >= maxPollFailures {
				return &ai.JobError{Kind: kind, ID: id, Err: errors.New("mediaio: polling kept failing: " + err.Error())}
			}
		}
		if err := retry.Sleep(ctx, interval); err != nil {
			return &ai.JobError{Kind: kind, ID: id, Err: err}
		}
		interval = min(interval+interval/2, limit)
	}
}

func transient(err error) bool {
	apiErr, ok := errors.AsType[*ai.APIError](err)
	return ok && apiErr.Retryable()
}
