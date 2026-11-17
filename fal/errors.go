package fal

import (
	"cmp"
	"encoding/json"
	"strings"

	"github.com/minodevss/ai-go/internal/transport"
)

func parseError(body []byte) (code, message string) {
	var env struct {
		Detail    json.RawMessage `json:"detail"`
		Type      string          `json:"type"`
		ErrorType string          `json:"error_type"`
		Message   string          `json:"message"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return "", ""
	}
	var items []struct {
		Msg  string `json:"msg"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(env.Detail, &items); err == nil && len(items) > 0 {
		msgs := make([]string, 0, len(items))
		for _, it := range items {
			msgs = append(msgs, it.Msg)
		}
		return items[0].Type, transport.Truncate(strings.Join(msgs, "; "))
	}
	var text string
	if err := json.Unmarshal(env.Detail, &text); err == nil {
		return cmp.Or(env.ErrorType, env.Type), transport.Truncate(text)
	}
	return cmp.Or(env.ErrorType, env.Type), transport.Truncate(env.Message)
}
