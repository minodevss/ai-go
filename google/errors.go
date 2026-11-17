package google

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/retry"
	"github.com/minodevss/ai-go/internal/transport"
)

type errorEnvelope struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Status  string            `json:"status"`
	Details []json.RawMessage `json:"details"`
}

func parseError(body []byte) (string, string) {
	var env struct {
		Error *errorEnvelope `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil || env.Error == nil {
		return "", ""
	}
	code := env.Error.Status
	if env.Error.Status == "INVALID_ARGUMENT" && tooLong(env.Error.Message) {
		code = "context_length_exceeded"
	}
	return code, transport.Truncate(env.Error.Message)
}

func tooLong(message string) bool {
	msg := strings.ToLower(message)
	return strings.Contains(msg, "token count") && strings.Contains(msg, "exceeds the maximum")
}

func (e errorEnvelope) apiError(provider string, body []byte) *ai.APIError {
	code := e.Status
	if e.Status == "INVALID_ARGUMENT" && tooLong(e.Message) {
		code = "context_length_exceeded"
	}
	return &ai.APIError{
		Provider:   provider,
		StatusCode: e.Code,
		Code:       code,
		Message:    transport.Truncate(e.Message),
		RetryAfter: min(retryDelay(e.Details), retry.MaxRetryAfter),
		Body:       string(body),
	}
}

func fail(err error) error {
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok {
		return err
	}
	if apiErr.RetryAfter == 0 {
		var env struct {
			Error errorEnvelope `json:"error"`
		}
		if json.Unmarshal([]byte(apiErr.Body), &env) == nil {
			apiErr.RetryAfter = min(retryDelay(env.Error.Details), retry.MaxRetryAfter)
		}
	}
	if apiErr.StatusCode == http.StatusPaymentRequired && apiErr.Code == "RESOURCE_EXHAUSTED" {
		apiErr.Code = "RESOURCE_EXHAUSTED_QUOTA"
	}
	return err
}

func retryDelay(details []json.RawMessage) time.Duration {
	for _, raw := range details {
		var d struct {
			Type       string `json:"@type"`
			RetryDelay string `json:"retryDelay"`
		}
		if json.Unmarshal(raw, &d) != nil || !strings.HasSuffix(d.Type, "RetryInfo") {
			continue
		}
		if delay, err := time.ParseDuration(d.RetryDelay); err == nil && delay > 0 {
			return delay
		}
	}
	return 0
}
