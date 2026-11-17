package anthropic

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	spendLimitCode = "enforced_spend_limit_reached"
	statusOverload = 529
)

func parseError(body []byte) (code, message string) {
	code, message = transport.ParseError(body)
	var env struct {
		Error struct {
			Details struct {
				ErrorCode string `json:"error_code"`
			} `json:"details"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err == nil && env.Error.Details.ErrorCode == spendLimitCode {
		code = "billing_" + spendLimitCode
	}
	return code, message
}

func contextTooLong(status int, _, message string) bool {
	if status != http.StatusBadRequest {
		return false
	}
	message = strings.ToLower(message)
	return slices.ContainsFunc([]string{"prompt is too long", "exceed context limit", "exceed the context window", "exceeds the context window"}, func(m string) bool { return strings.Contains(message, m) })
}

func eventError(typ, message string) *ai.APIError {
	err := transport.ErrorFromEvent(providerName, typ, message)
	err.StatusCode = statusForType(typ)
	err.ContextTooLong = contextTooLong(err.StatusCode, typ, message)
	return err
}

func statusForType(typ string) int {
	switch typ {
	case "invalid_request_error":
		return http.StatusBadRequest
	case "authentication_error":
		return http.StatusUnauthorized
	case "billing_error":
		return http.StatusPaymentRequired
	case "permission_error":
		return http.StatusForbidden
	case "not_found_error":
		return http.StatusNotFound
	case "conflict_error":
		return http.StatusConflict
	case "request_too_large":
		return http.StatusRequestEntityTooLarge
	case "rate_limit_error":
		return http.StatusTooManyRequests
	case "api_error":
		return http.StatusInternalServerError
	case "timeout_error":
		return http.StatusGatewayTimeout
	case "overloaded_error":
		return statusOverload
	default:
		return 0
	}
}
