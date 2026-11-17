package toolmsg_test

import (
	"encoding/json"
	"testing"

	"github.com/minodevss/ai-go/internal/toolmsg"
)

func TestDenied(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   string
	}{
		{name: "no reason", want: "tool call was denied"},
		{name: "reason", reason: "too risky", want: "tool call was denied: too risky"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := toolmsg.Denied(tt.reason); got != tt.want {
				t.Errorf("Denied(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestNormalizeInput(t *testing.T) {
	tests := []struct {
		name  string
		input json.RawMessage
		want  string
	}{
		{name: "nil", want: "{}"},
		{name: "blank", input: json.RawMessage(" \n"), want: "{}"},
		{name: "trimmed", input: json.RawMessage(` {"a":1} `), want: `{"a":1}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(toolmsg.NormalizeInput(tt.input)); got != tt.want {
				t.Errorf("NormalizeInput(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
