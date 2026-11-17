package bedrock

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

func exceptionStatus(code string) int {
	// Status codes of the exceptions documented for Converse, ConverseStream and InvokeModel: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
	switch code {
	case "AccessDeniedException":
		return http.StatusForbidden
	case "InternalServerException":
		return http.StatusInternalServerError
	case "ModelErrorException", "ModelStreamErrorException":
		return http.StatusFailedDependency
	case "ModelNotReadyException", "ThrottlingException":
		return http.StatusTooManyRequests
	case "ModelTimeoutException":
		return http.StatusRequestTimeout
	case "ResourceNotFoundException":
		return http.StatusNotFound
	case "ServiceUnavailableException":
		return http.StatusServiceUnavailable
	case "ValidationException":
		return http.StatusBadRequest
	default:
		return 0
	}
}

func parseError(body []byte) (code, message string) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return "", ""
	}
	for name, raw := range fields {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			continue
		}
		switch strings.ToLower(name) {
		case typeField, "code":
			if code == "" || strings.EqualFold(name, typeField) {
				code = sanitizeErrorType(text)
			}
		case "message", "errormessage":
			message = transport.Truncate(text)
		}
	}
	return code, message
}

func contextTooLong(_ int, _, message string) bool {
	message = strings.ToLower(message)
	return slices.ContainsFunc([]string{"input is too long", "prompt is too long", "too many input tokens", "exceed context limit"}, func(m string) bool { return strings.Contains(message, m) })
}

func modelErrorRetry(code string, payload []byte) *bool {
	if code != "ModelErrorException" && code != "ModelStreamErrorException" {
		return nil
	}
	var body struct {
		OriginalStatusCode int `json:"originalStatusCode"`
	}
	if err := json.Unmarshal(payload, &body); err != nil || body.OriginalStatusCode == 0 {
		return nil
	}
	switch body.OriginalStatusCode {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return new(true)
	default:
		return new(body.OriginalStatusCode >= http.StatusInternalServerError)
	}
}

func streamError(exceptionType string, payload []byte) *ai.APIError {
	code := capitalize(exceptionType)
	_, message := parseError(payload)
	status := exceptionStatus(code)
	return &ai.APIError{Provider: providerName, StatusCode: status, Code: code, Message: message, Body: string(payload), ShouldRetry: modelErrorRetry(code, payload), ContextTooLong: contextTooLong(status, code, message)}
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
