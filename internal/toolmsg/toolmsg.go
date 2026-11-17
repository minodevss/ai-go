// Package toolmsg holds the tool-call texts and input handling shared by the core loop and realtime sessions.
package toolmsg

import (
	"bytes"
	"encoding/json"
)

const (
	Failure  = "tool failed"
	TimedOut = "tool timed out"
)

func Denied(reason string) string {
	if reason == "" {
		return "tool call was denied"
	}
	return "tool call was denied: " + reason
}

func NormalizeInput(input json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(input)
	if len(trimmed) == 0 {
		return json.RawMessage("{}")
	}
	return trimmed
}
