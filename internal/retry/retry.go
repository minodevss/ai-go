package retry

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	MaxRetryAfter = 60 * time.Second
	baseDelay     = 500 * time.Millisecond
	maxBackoff    = 8 * time.Second
)

func Backoff(attempt int, hint time.Duration) time.Duration {
	if hint > 0 {
		return min(hint, MaxRetryAfter)
	}
	d := maxBackoff
	if attempt < 5 {
		d = min(baseDelay<<attempt, maxBackoff)
	}
	return d/2 + rand.N(d/2+1)
}

func RetryAfter(h http.Header, now time.Time) time.Duration {
	if v := strings.TrimSpace(h.Get("retry-after-ms")); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && ms > 0 {
			return min(time.Duration(ms*float64(time.Millisecond)), MaxRetryAfter)
		}
	}
	v := strings.TrimSpace(h.Get("retry-after"))
	if v == "" {
		return 0
	}
	if s, err := strconv.ParseFloat(v, 64); err == nil && s > 0 {
		return min(time.Duration(s*float64(time.Second)), MaxRetryAfter)
	}
	if t, err := http.ParseTime(v); err == nil && t.After(now) {
		return min(t.Sub(now), MaxRetryAfter)
	}
	return 0
}

func Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
