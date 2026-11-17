package mediaio

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/minodevss/ai-go"
)

const (
	CodeContentPolicy = "content_policy_violation"
	CodeBilling       = "billing_payment_required"
	CodeJobFailed     = "job_failed"
	CodeJobCanceled   = "job_canceled"
)

// ParseError reads the error shapes of the hosted generation APIs: a string or object under error, a detail string or a FastAPI validation list, or title and message.
func ParseError(body []byte) (code, message string) {
	var env struct {
		Error   json.RawMessage `json:"error"`
		Detail  json.RawMessage `json:"detail"`
		Title   string          `json:"title"`
		Message string          `json:"message"`
		Code    json.RawMessage `json:"code"`
		Type    string          `json:"type"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return "", ""
	}
	code = text(env.Code)
	if code == "" {
		code = env.Type
	}
	message = text(env.Detail)
	if message == "" {
		message = text(env.Error)
	}
	if message == "" {
		message = env.Message
	}
	if message == "" {
		message = env.Title
	}
	if code == "" && env.Title != "" && env.Title != message {
		code = env.Title
	}
	var obj struct {
		Code    json.RawMessage `json:"code"`
		Type    string          `json:"type"`
		Message string          `json:"message"`
	}
	if json.Unmarshal(env.Error, &obj) == nil && obj.Message != "" {
		message = obj.Message
		if c := text(obj.Code); c != "" {
			code = c
		} else if obj.Type != "" {
			code = obj.Type
		}
	}
	if len(message) > 300 {
		message = strings.ToValidUTF8(message[:300], "") + "..."
	}
	return code, message
}

func text(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var list []struct {
		Msg     string `json:"msg"`
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &list) == nil {
		parts := make([]string, 0, len(list))
		for _, e := range list {
			if m := e.Msg + e.Message; m != "" {
				parts = append(parts, m)
			}
		}
		return strings.Join(parts, "; ")
	}
	var n json.Number
	if json.Unmarshal(raw, &n) == nil {
		return n.String()
	}
	return ""
}

// FixError gives 402 responses a billing code, so errors.Is(err, ai.ErrQuota) holds and the error is not retried.
func FixError(err error) error {
	if apiErr, ok := errors.AsType[*ai.APIError](err); ok && apiErr.StatusCode == http.StatusPaymentRequired && !apiErr.Is(ai.ErrQuota) {
		apiErr.Code = CodeBilling
	}
	return err
}

// Blocked is the error of a request or result that the provider's moderation rejected. It is not retryable.
func Blocked(provider, message string) *ai.APIError {
	return &ai.APIError{Provider: provider, StatusCode: http.StatusUnprocessableEntity, Code: CodeContentPolicy, Message: message}
}

// Failed is the error of a job that ended unsuccessfully. It is not retryable.
func Failed(provider, code, message string) *ai.APIError {
	return &ai.APIError{Provider: provider, Code: code, Message: message}
}

// LooksModerated reports whether a free-text failure message names a safety or moderation block.
func LooksModerated(message string) bool {
	m := strings.ToLower(message)
	for _, w := range []string{"nsfw", "moderat", "safety", "sensitive", "content policy", "blacklisted", "violat"} {
		if strings.Contains(m, w) {
			return true
		}
	}
	return false
}
