package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
)

type Celsius float64

func (Celsius) JSONSchema() json.RawMessage {
	return json.RawMessage(`{"type":"number","minimum":-273.15,"description":"degrees Celsius"}`)
}

type weatherIn struct {
	City  string    `json:"city" jsonschema:"city name"`
	Units string    `json:"units,omitempty"`
	Temp  Celsius   `json:"temp"`
	Temps []Celsius `json:"temps,omitempty"`
}

func schemaOf(t *testing.T, tool ai.Tool) map[string]any {
	t.Helper()
	var s map[string]any
	mustUnmarshal(t, tool.InputSchema, &s)
	return s
}

func TestNewTool_schema(t *testing.T) {
	tool := ai.NewTool("weather", "get weather", func(context.Context, weatherIn) (string, error) { return "", nil })
	s := schemaOf(t, tool)
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %v", s)
	}

	tests := []struct {
		name string
		got  any
		want any
	}{
		{name: "type", got: s["type"], want: "object"},
		{name: "city description", got: at(props, "city", "description"), want: "city name"},
		{name: "required", got: s["required"], want: []any{"city", "temp"}},
		{name: "override on field", got: props["temp"], want: map[string]any{"type": "number", "minimum": -273.15, "description": "degrees Celsius"}},
		{name: "override on slice element", got: at(props, "temps", "items"), want: map[string]any{"type": "number", "minimum": -273.15, "description": "degrees Celsius"}},
		{name: "no extra properties", got: s["additionalProperties"], want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, tt.got); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
	if tool.Name != "weather" || tool.Description != "get weather" || tool.Execute == nil {
		t.Errorf("tool = %+v", tool)
	}
}

func TestNewTool_overrideIsEnforcedByValidation(t *testing.T) {
	var got float64
	tool := ai.NewTool("w", "d", func(_ context.Context, in weatherIn) (string, error) {
		got = float64(in.Temp)
		return "ok", nil
	})
	m := newScripted(t, `{"city":"Oz","temp":-500}`, `{"city":"Oz","temp":20.5}`)
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{tool}})

	results := resultsOf(res.Messages)
	if len(results) != 2 || !results[0].IsError || results[1].IsError || got != 20.5 {
		t.Errorf("results = %+v, temp = %v; want first rejected by the override schema, second executed", results, got)
	}
}

func TestNewTool_executeOutputs(t *testing.T) {
	type out struct {
		N int `json:"n"`
	}
	tests := []struct {
		name string
		tool ai.Tool
		want ai.ToolOutput
	}{
		{
			name: "struct becomes JSON",
			tool: ai.NewTool("t", "d", func(context.Context, struct{}) (out, error) { return out{N: 3}, nil }),
			want: ai.ToolOutput{JSON: json.RawMessage(`{"n":3}`)},
		},
		{
			name: "string becomes a JSON string",
			tool: ai.NewTool("t", "d", func(context.Context, struct{}) (string, error) { return "hi", nil }),
			want: ai.ToolOutput{JSON: json.RawMessage(`"hi"`)},
		},
		{
			name: "ToolOutput passes through",
			tool: ai.NewTool("t", "d", func(context.Context, struct{}) (ai.ToolOutput, error) { return ai.ToolOutput{Text: "raw"}, nil }),
			want: ai.ToolOutput{Text: "raw"},
		},
		{
			name: "nil slice becomes null",
			tool: ai.NewTool("t", "d", func(context.Context, struct{}) ([]string, error) { return nil, nil }),
			want: ai.ToolOutput{JSON: json.RawMessage(`null`)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.tool.Execute(t.Context(), nil)
			if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Execute() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewTool_inputDecoding(t *testing.T) {
	var seen searchIn
	tool := ai.NewTool("t", "d", func(_ context.Context, in searchIn) (int, error) { seen = in; return 0, nil })
	tests := []struct {
		name    string
		input   string
		want    searchIn
		wantErr bool
	}{
		{name: "object", input: `{"query":"x"}`, want: searchIn{Query: "x"}},
		{name: "empty input is an empty object", input: ``, want: searchIn{}},
		{name: "whitespace", input: "  \n", want: searchIn{}},
		{name: "malformed", input: `{`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seen = searchIn{}
			_, err := tool.Execute(t.Context(), json.RawMessage(tt.input))
			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				if _, ok := errors.AsType[*ai.ToolError](err); !ok {
					t.Errorf("error = %v, want a *ToolError", err)
				}
			}
			if seen != tt.want {
				t.Errorf("input = %+v, want %+v", seen, tt.want)
			}
		})
	}
}

func TestNewTool_panicsOnUnrepresentableInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("NewTool with a chan field did not panic")
		}
	}()
	ai.NewTool("bad", "d", func(context.Context, struct{ C chan int }) (int, error) { return 0, nil })
}

func TestNewClientTool(t *testing.T) {
	tool := ai.NewClientTool[searchIn]("ask", "ask the user")
	if tool.Execute != nil {
		t.Error("client tool has an Execute function")
	}
	if got := schemaOf(t, tool)["type"]; got != "object" {
		t.Errorf("schema type = %v, want object", got)
	}
}

func TestTool_withersCopy(t *testing.T) {
	base := searchTool()
	approved := base.WithApproval(ai.AlwaysAsk)
	timed := base.WithTimeout(time.Second)

	if base.Approval != nil || base.Timeout != 0 {
		t.Error("With* mutated the receiver")
	}
	if approved.Approval == nil || timed.Timeout != time.Second {
		t.Error("With* did not set the field")
	}
}

func TestToolChoice(t *testing.T) {
	tests := []struct {
		name     string
		choice   ai.ToolChoice
		wantMode ai.ToolChoiceMode
		wantName string
	}{
		{name: "zero is auto", choice: ai.ToolChoice{}, wantMode: ai.ToolChoiceAuto},
		{name: "required", choice: ai.ToolRequired(), wantMode: ai.ToolChoiceRequired},
		{name: "none", choice: ai.ToolNone(), wantMode: ai.ToolChoiceNone},
		{name: "named", choice: ai.ToolNamed("search"), wantMode: ai.ToolChoiceNamed, wantName: "search"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.choice.Mode(); got != tt.wantMode {
				t.Errorf("Mode() = %q, want %q", got, tt.wantMode)
			}
			if got := tt.choice.Name(); got != tt.wantName {
				t.Errorf("Name() = %q, want %q", got, tt.wantName)
			}
		})
	}
}

func TestToolErrorf(t *testing.T) {
	cause := errors.New("root cause")
	tests := []struct {
		name        string
		err         error
		wantMessage string
		wantCause   bool
	}{
		{name: "formats", err: ai.ToolErrorf("bad %s %d", "thing", 3), wantMessage: "bad thing 3"},
		{name: "wraps with %w", err: ai.ToolErrorf("lookup: %w", cause), wantMessage: "lookup: root cause", wantCause: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toolErr, ok := errors.AsType[*ai.ToolError](tt.err)
			if !ok || toolErr.Message != tt.wantMessage {
				t.Fatalf("error = %#v, want a ToolError with message %q", tt.err, tt.wantMessage)
			}
			if got := errors.Is(tt.err, cause); got != tt.wantCause {
				t.Errorf("errors.Is(cause) = %v, want %v", got, tt.wantCause)
			}
		})
	}
}

func TestNoSuchErrors(t *testing.T) {
	if got, want := (&ai.NoSuchToolError{Name: "x"}).Error(), `ai: no such tool "x"`; got != want {
		t.Errorf("NoSuchToolError = %q, want %q", got, want)
	}
	if got, want := (&ai.NoSuchModelError{Spec: "a:b", Kind: "language"}).Error(), `ai: no such language model "a:b"`; got != want {
		t.Errorf("NoSuchModelError = %q, want %q", got, want)
	}
}

func at(v any, keys ...string) any {
	for _, k := range keys {
		m, ok := v.(map[string]any)
		if !ok {
			return nil
		}
		v = m[k]
	}
	return v
}
