package chatcompletions

import (
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
)

func TestModel_requestWarnsWhenStrictCannotApply(t *testing.T) {
	m := New(nil, "m", &Dialect{Provider: "openai", ToolStrict: true})
	call := &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Tools:    []ai.Tool{{Name: "f", InputSchema: []byte(`{not json`), Strict: true}},
	}
	_, warnings, err := m.request(call, false)
	if err != nil {
		t.Fatalf("request() error = %v", err)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0].Message, `tool "f" cannot use strict mode`) {
		t.Errorf("warnings = %v, want a strict fallback warning for f", warnings)
	}
}
