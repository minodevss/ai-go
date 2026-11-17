package ai

import (
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"
)

var (
	ErrRateLimit       = errors.New("ai: rate limited")
	ErrQuota           = errors.New("ai: quota exceeded")
	ErrAuth            = errors.New("ai: authentication failed")
	ErrContextTooLong  = errors.New("ai: context too long")
	ErrStreamConsumed  = errors.New("ai: stream already consumed")
	ErrStreamActive    = errors.New("ai: stream is being consumed")
	ErrDuplicateTool   = errors.New("ai: duplicate tool name")
	ErrInvalidApproval = errors.New("ai: invalid tool approval")
)

// APIError.Body holds the raw provider response and is never part of Error(). ShouldRetry, when non-nil, carries the provider's explicit retry instruction and overrides the status-based default.
// ContextTooLong is set by the provider adapter when it recognizes a context-length failure that the generic checks in Is miss.
type APIError struct {
	Provider       string
	StatusCode     int
	Code           string
	Message        string
	RequestID      string
	RetryAfter     time.Duration
	Body           string
	ShouldRetry    *bool
	ContextTooLong bool
}

func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString("ai: ")
	if e.Provider != "" {
		b.WriteString(e.Provider)
		b.WriteString(": ")
	}
	if e.StatusCode != 0 {
		fmt.Fprintf(&b, "status %d", e.StatusCode)
	} else {
		b.WriteString("api error")
	}
	if e.Code != "" {
		fmt.Fprintf(&b, " (%s)", e.Code)
	}
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

func (e *APIError) Retryable() bool {
	if e.ShouldRetry != nil {
		return *e.ShouldRetry
	}
	if e.Is(ErrQuota) {
		return false
	}
	switch e.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	default:
		return e.StatusCode >= http.StatusInternalServerError
	}
}

func (e *APIError) Is(target error) bool {
	switch target {
	case ErrQuota:
		return e.quota()
	case ErrRateLimit:
		return e.StatusCode == http.StatusTooManyRequests && !e.quota()
	case ErrAuth:
		return e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden
	case ErrContextTooLong:
		return e.contextTooLong()
	default:
		return false
	}
}

func (e *APIError) quota() bool {
	code := strings.ToLower(e.Code)
	return strings.Contains(code, "quota") || strings.Contains(code, "billing") || code == "insufficient_funds"
}

func (e *APIError) contextTooLong() bool {
	msg := strings.ToLower(e.Message)
	return e.ContextTooLong || e.Code == "context_length_exceeded" ||
		strings.Contains(msg, "maximum context length") || strings.Contains(msg, "prompt is too long")
}

type RetryError struct {
	Errs []error
}

func (e *RetryError) Error() string {
	last := "unknown"
	if len(e.Errs) > 0 {
		last = e.Errs[len(e.Errs)-1].Error()
	}
	return fmt.Sprintf("ai: failed after %d attempts: %s", len(e.Errs), last)
}

func (e *RetryError) Unwrap() []error { return slices.Clone(e.Errs) }

// OutputError.Text is the model output that failed to decode or validate.
type OutputError struct {
	Text string
	Err  error
}

func (e *OutputError) Error() string { return "ai: invalid structured output: " + e.Err.Error() }

func (e *OutputError) Unwrap() error { return e.Err }

// ToolError messages are shown to the model; all other tool errors are replaced by a generic message.
type ToolError struct {
	Message string
	Err     error
}

func (e *ToolError) Error() string { return "ai: tool error: " + e.Message }

func (e *ToolError) Unwrap() error { return e.Err }

func ToolErrorf(format string, args ...any) error {
	err := fmt.Errorf(format, args...)
	return &ToolError{Message: err.Error(), Err: errors.Unwrap(err)}
}

type NoSuchToolError struct {
	Name      string
	Available []string
}

func (e *NoSuchToolError) Error() string {
	return fmt.Sprintf("ai: no such tool %q", e.Name)
}

type NoSuchModelError struct {
	Spec string
	Kind string
}

func (e *NoSuchModelError) Error() string {
	return fmt.Sprintf("ai: no such %s model %q", e.Kind, e.Spec)
}

var ErrUnsupported = errors.New("ai: operation not supported by this model")

// JobError reports a failed or abandoned asynchronous job. ID is the provider's job id, so the job can still be found after ctx ends.
type JobError struct {
	Kind string
	ID   string
	Err  error
}

func (e *JobError) Error() string { return fmt.Sprintf("ai: %s job %s: %v", e.Kind, e.ID, e.Err) }

func (e *JobError) Unwrap() error { return e.Err }
