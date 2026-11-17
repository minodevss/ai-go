package ai_test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func apiErr(status int) *ai.APIError {
	return &ai.APIError{Provider: "test", StatusCode: status, Message: "failed", Body: "secret-body"}
}

func failures(n int, err error) []aitest.ModelOption {
	turns := make([]aitest.ModelOption, n)
	for i := range turns {
		turns[i] = aitest.Turn(aitest.Err(err))
	}
	return turns
}

func TestRetry_policy(t *testing.T) {
	tests := []struct {
		name         string
		err          error
		maxRetries   *int
		wantAttempts int
	}{
		{name: "503 retries twice by default", err: apiErr(503), wantAttempts: 3},
		{name: "529 overloaded retries", err: apiErr(529), wantAttempts: 3},
		{name: "500 retries", err: apiErr(500), wantAttempts: 3},
		{name: "408 retries", err: apiErr(408), wantAttempts: 3},
		{name: "409 retries", err: apiErr(409), wantAttempts: 3},
		{name: "429 retries", err: apiErr(429), wantAttempts: 3},
		{name: "429 quota does not retry", err: &ai.APIError{StatusCode: 429, Code: "insufficient_quota"}, wantAttempts: 1},
		{name: "400 does not retry", err: apiErr(400), wantAttempts: 1},
		{name: "401 does not retry", err: apiErr(401), wantAttempts: 1},
		{name: "404 does not retry", err: apiErr(404), wantAttempts: 1},
		{name: "plain error does not retry", err: errors.New("boom"), wantAttempts: 1},
		{name: "MaxRetries 0", err: apiErr(503), maxRetries: new(0), wantAttempts: 1},
		{name: "MaxRetries 4", err: apiErr(503), maxRetries: new(4), wantAttempts: 5},
		{name: "negative MaxRetries is 0", err: apiErr(503), maxRetries: new(-3), wantAttempts: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewModel(failures(10, tt.err)...)
				_, err := ai.GenerateText(t.Context(), m, ai.TextRequest{MaxRetries: tt.maxRetries})
				if err == nil {
					t.Fatal("GenerateText() error = nil, want an error")
				}
				if got := len(m.Calls()); got != tt.wantAttempts {
					t.Errorf("attempts = %d, want %d", got, tt.wantAttempts)
				}
				retryErr, isRetry := errors.AsType[*ai.RetryError](err)
				if tt.wantAttempts > 1 {
					if !isRetry || len(retryErr.Errs) != tt.wantAttempts {
						t.Errorf("error = %v, want a RetryError with %d errors", err, tt.wantAttempts)
					}
				} else if isRetry {
					t.Errorf("error = %v, want the original error unwrapped", err)
				}
				if !errors.Is(err, tt.err) {
					t.Errorf("error does not wrap %v", tt.err)
				}
			})
		})
	}
}

func TestRetry_recovers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewModel(aitest.Turn(aitest.Err(apiErr(503))), aitest.Turn(aitest.Err(apiErr(429))), aitest.Turn(aitest.Text("finally")))
		res := mustGenerate(t, m, ai.TextRequest{})
		if res.Text != "finally" || len(m.Calls()) != 3 {
			t.Errorf("Text = %q after %d calls, want %q after 3", res.Text, len(m.Calls()), "finally")
		}
	})
}

func TestRetry_backoffBounds(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewModel(failures(3, apiErr(503))...)
		start := time.Now()
		_, _ = ai.GenerateText(t.Context(), m, ai.TextRequest{})
		elapsed := time.Since(start)
		if elapsed < 750*time.Millisecond || elapsed > 1500*time.Millisecond {
			t.Errorf("total backoff = %v, want within [750ms, 1.5s]", elapsed)
		}
	})
}

func TestRetry_retryAfter(t *testing.T) {
	tests := []struct {
		name       string
		retryAfter time.Duration
		want       time.Duration
	}{
		{name: "honors the hint", retryAfter: 2 * time.Second, want: 2 * time.Second},
		{name: "small hint", retryAfter: 10 * time.Millisecond, want: 10 * time.Millisecond},
		{name: "capped at 60s", retryAfter: 10 * time.Minute, want: 60 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				err := &ai.APIError{StatusCode: 429, RetryAfter: tt.retryAfter}
				m := aitest.NewModel(aitest.Turn(aitest.Err(err)), aitest.Turn(aitest.Text("ok")))
				start := time.Now()
				mustGenerate(t, m, ai.TextRequest{})
				if got := time.Since(start); got != tt.want {
					t.Errorf("waited %v, want %v", got, tt.want)
				}
			})
		})
	}
}

func TestRetry_cancelDuringBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		time.AfterFunc(50*time.Millisecond, cancel)
		m := aitest.NewModel(failures(3, apiErr(503))...)
		_, err := ai.GenerateText(ctx, m, ai.TextRequest{})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want context.Canceled", err)
		}
		if got := len(m.Calls()); got != 1 {
			t.Errorf("attempts = %d, want 1", got)
		}
	})
}

func TestRetry_streamOnlyBeforeFirstPart(t *testing.T) {
	tests := []struct {
		name         string
		turns        []aitest.ModelOption
		wantAttempts int
		wantText     string
		wantErr      bool
	}{
		{
			name:         "error opening the stream is retried",
			turns:        []aitest.ModelOption{aitest.Turn(aitest.Err(apiErr(503))), aitest.Turn(aitest.Text("ok"))},
			wantAttempts: 2,
			wantText:     "ok",
		},
		{
			name:         "error after the first part is not retried",
			turns:        []aitest.ModelOption{aitest.Turn(aitest.Text("partial"), aitest.Err(apiErr(503))), aitest.Turn(aitest.Text("never"))},
			wantAttempts: 1,
			wantErr:      true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewModel(tt.turns...)
				res, err := ai.StreamText(t.Context(), m, ai.TextRequest{}).Result()
				if (err != nil) != tt.wantErr {
					t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
				}
				if got := len(m.Calls()); got != tt.wantAttempts {
					t.Errorf("attempts = %d, want %d", got, tt.wantAttempts)
				}
				if res.Text != tt.wantText {
					t.Errorf("Text = %q, want %q", res.Text, tt.wantText)
				}
			})
		})
	}
}

func TestRetry_firstChunkTimeoutIsRetried(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewModel(aitest.Turn(aitest.Delay(time.Minute), aitest.Text("slow")), aitest.Turn(aitest.Text("fast")))
		res, err := ai.StreamText(t.Context(), m, ai.TextRequest{Timeouts: ai.Timeouts{FirstChunk: time.Second}}).Result()
		if err != nil || res.Text != "fast" {
			t.Errorf("Result() = %+v, %v; want Text %q", res, err, "fast")
		}
	})
}

func TestAPIError_categories(t *testing.T) {
	tests := []struct {
		name          string
		err           *ai.APIError
		wantIs        []error
		wantNot       []error
		wantRetryable bool
	}{
		{name: "rate limit", err: &ai.APIError{StatusCode: http.StatusTooManyRequests, Code: "rate_limit_exceeded"}, wantIs: []error{ai.ErrRateLimit}, wantNot: []error{ai.ErrQuota}, wantRetryable: true},
		{name: "quota", err: &ai.APIError{StatusCode: http.StatusTooManyRequests, Code: "insufficient_quota"}, wantIs: []error{ai.ErrQuota}, wantNot: []error{ai.ErrRateLimit}},
		{name: "unauthorized", err: &ai.APIError{StatusCode: http.StatusUnauthorized}, wantIs: []error{ai.ErrAuth}},
		{name: "forbidden", err: &ai.APIError{StatusCode: http.StatusForbidden}, wantIs: []error{ai.ErrAuth}},
		{name: "context code", err: &ai.APIError{StatusCode: http.StatusBadRequest, Code: "context_length_exceeded"}, wantIs: []error{ai.ErrContextTooLong}},
		{name: "context message", err: &ai.APIError{StatusCode: http.StatusBadRequest, Message: "prompt is too long: 300000 tokens"}, wantIs: []error{ai.ErrContextTooLong}},
		{name: "provider classified context", err: &ai.APIError{StatusCode: http.StatusBadRequest, ContextTooLong: true}, wantIs: []error{ai.ErrContextTooLong}},
		{name: "should retry true", err: &ai.APIError{StatusCode: http.StatusBadRequest, ShouldRetry: new(true)}, wantRetryable: true},
		{name: "should retry false", err: &ai.APIError{StatusCode: http.StatusInternalServerError, ShouldRetry: new(false)}},
		{name: "server error", err: &ai.APIError{StatusCode: http.StatusInternalServerError}, wantNot: []error{ai.ErrRateLimit, ai.ErrAuth}, wantRetryable: true},
		{name: "payload too large", err: &ai.APIError{StatusCode: http.StatusRequestEntityTooLarge}, wantNot: []error{ai.ErrContextTooLong}},
		{name: "bad request", err: &ai.APIError{StatusCode: http.StatusBadRequest}, wantNot: []error{ai.ErrContextTooLong}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, target := range tt.wantIs {
				if !errors.Is(tt.err, target) {
					t.Errorf("errors.Is(%v, %v) = false, want true", tt.err, target)
				}
			}
			for _, target := range tt.wantNot {
				if errors.Is(tt.err, target) {
					t.Errorf("errors.Is(%v, %v) = true, want false", tt.err, target)
				}
			}
			if got := tt.err.Retryable(); got != tt.wantRetryable {
				t.Errorf("Retryable() = %v, want %v", got, tt.wantRetryable)
			}
		})
	}
}

func TestAPIError_bodyNeverInMessage(t *testing.T) {
	err := &ai.APIError{Provider: "openai", StatusCode: 500, Code: "server_error", Message: "oops", RequestID: "req_1", Body: "SECRET-BODY"}
	wrapped := &ai.RetryError{Errs: []error{err, err}}
	for _, e := range []error{err, wrapped} {
		if containsStr(e.Error(), "SECRET-BODY") {
			t.Errorf("Error() = %q leaks the response body", e.Error())
		}
	}
	if got, want := err.Error(), "ai: openai: status 500 (server_error): oops"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestRetryError_unwrapsAll(t *testing.T) {
	first, last := errors.New("first"), errors.New("last")
	err := &ai.RetryError{Errs: []error{first, last}}
	if !errors.Is(err, first) || !errors.Is(err, last) {
		t.Errorf("RetryError does not unwrap to both errors")
	}
	if diff := cmp.Diff("ai: failed after 2 attempts: last", err.Error()); diff != "" {
		t.Errorf("Error() mismatch (-want +got):\n%s", diff)
	}
}
