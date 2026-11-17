package retry_test

import (
	"net/http"
	"testing"
	"time"

	"github.com/minodevss/ai-go/internal/retry"
)

func TestBackoff(t *testing.T) {
	tests := []struct {
		name    string
		attempt int
		hint    time.Duration
		min     time.Duration
		max     time.Duration
	}{
		{name: "first attempt", attempt: 0, min: 250 * time.Millisecond, max: 500 * time.Millisecond},
		{name: "second attempt", attempt: 1, min: 500 * time.Millisecond, max: time.Second},
		{name: "grows to the cap", attempt: 10, min: 4 * time.Second, max: 8 * time.Second},
		{name: "huge attempt does not overflow", attempt: 200, min: 4 * time.Second, max: 8 * time.Second},
		{name: "hint wins", attempt: 0, hint: 3 * time.Second, min: 3 * time.Second, max: 3 * time.Second},
		{name: "hint capped", attempt: 0, hint: time.Hour, min: retry.MaxRetryAfter, max: retry.MaxRetryAfter},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for range 200 {
				if got := retry.Backoff(tt.attempt, tt.hint); got < tt.min || got > tt.max {
					t.Fatalf("Backoff(%d, %v) = %v, want within [%v, %v]", tt.attempt, tt.hint, got, tt.min, tt.max)
				}
			}
		})
	}
}

func TestRetryAfter(t *testing.T) {
	now := time.Date(2026, 11, 17, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name   string
		header http.Header
		want   time.Duration
	}{
		{name: "none", header: http.Header{}, want: 0},
		{name: "seconds", header: http.Header{"Retry-After": {"7"}}, want: 7 * time.Second},
		{name: "fractional seconds", header: http.Header{"Retry-After": {"1.5"}}, want: 1500 * time.Millisecond},
		{name: "milliseconds header wins", header: http.Header{"Retry-After-Ms": {"250"}, "Retry-After": {"9"}}, want: 250 * time.Millisecond},
		{name: "http date", header: http.Header{"Retry-After": {now.Add(20 * time.Second).UTC().Format(http.TimeFormat)}}, want: 20 * time.Second},
		{name: "date in the past", header: http.Header{"Retry-After": {now.Add(-time.Minute).UTC().Format(http.TimeFormat)}}, want: 0},
		{name: "capped seconds", header: http.Header{"Retry-After": {"3600"}}, want: retry.MaxRetryAfter},
		{name: "capped milliseconds", header: http.Header{"Retry-After-Ms": {"900000"}}, want: retry.MaxRetryAfter},
		{name: "garbage", header: http.Header{"Retry-After": {"soon"}}, want: 0},
		{name: "negative", header: http.Header{"Retry-After": {"-5"}}, want: 0},
		{name: "bad ms falls back to seconds", header: http.Header{"Retry-After-Ms": {"x"}, "Retry-After": {"2"}}, want: 2 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := retry.RetryAfter(tt.header, now); got != tt.want {
				t.Errorf("RetryAfter() = %v, want %v", got, tt.want)
			}
		})
	}
}
