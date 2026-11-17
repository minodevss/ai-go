package schema_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/internal/schema"
)

type Money string

func (Money) JSONSchema() json.RawMessage {
	return json.RawMessage(`{"type":"string","pattern":"^[0-9]+\\.[0-9]{2}$"}`)
}

type invoice struct {
	ID     string  `json:"id" jsonschema:"invoice id"`
	Total  Money   `json:"total"`
	Extras []Money `json:"extras,omitempty"`
	Note   *string `json:"note,omitempty"`
}

func TestFor(t *testing.T) {
	raw, err := schema.For[invoice]()
	if err != nil {
		t.Fatalf("For() error = %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("schema is not JSON: %v", err)
	}
	props, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("no properties in %s", raw)
	}
	tests := []struct {
		name string
		got  any
		want any
	}{
		{name: "root type", got: got["type"], want: "object"},
		{name: "required", got: got["required"], want: []any{"id", "total"}},
		{name: "description", got: at(props, "id", "description"), want: "invoice id"},
		{name: "override", got: at(props, "total", "pattern"), want: `^[0-9]+\.[0-9]{2}$`},
		{name: "override in slice", got: at(props, "extras", "items", "type"), want: "string"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, tt.got); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFor_rootOverrideAndErrors(t *testing.T) {
	raw, err := schema.For[Money]()
	if err != nil {
		t.Fatalf("For[Money]() error = %v", err)
	}
	if v := mustCompile(t, raw); v.Validate([]byte(`"12.50"`)) != nil || v.Validate([]byte(`"12.5"`)) == nil {
		t.Errorf("root override not enforced for %s", raw)
	}
	if _, err := schema.For[struct{ C chan int }](); err == nil {
		t.Error("For(chan) error = nil, want an error")
	}
}

func mustCompile(t *testing.T, raw json.RawMessage) *schema.Validator {
	t.Helper()
	v, err := schema.Compile(raw)
	if err != nil {
		t.Fatalf("Compile(%s) error = %v", raw, err)
	}
	return v
}

func TestValidator(t *testing.T) {
	raw, err := schema.For[invoice]()
	if err != nil {
		t.Fatalf("For() error = %v", err)
	}
	v := mustCompile(t, raw)
	tests := []struct {
		name    string
		doc     string
		wantErr bool
	}{
		{name: "valid", doc: `{"id":"a","total":"10.00"}`},
		{name: "valid with optional", doc: `{"id":"a","total":"10.00","note":"n","extras":["1.00"]}`},
		{name: "missing required", doc: `{"id":"a"}`, wantErr: true},
		{name: "pattern violated", doc: `{"id":"a","total":"ten"}`, wantErr: true},
		{name: "wrong type", doc: `{"id":1,"total":"10.00"}`, wantErr: true},
		{name: "unknown property", doc: `{"id":"a","total":"10.00","x":1}`, wantErr: true},
		{name: "not JSON", doc: `{`, wantErr: true},
		{name: "not an object", doc: `[]`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := v.Validate([]byte(tt.doc)); (err != nil) != tt.wantErr {
				t.Errorf("Validate(%s) error = %v, wantErr %v", tt.doc, err, tt.wantErr)
			}
		})
	}
}

func TestCompile_rejectsBadSchemas(t *testing.T) {
	for _, raw := range []string{`{`, `{"properties":5}`, `{"$ref":"#/missing"}`} {
		t.Run(raw, func(t *testing.T) {
			if _, err := schema.Compile(json.RawMessage(raw)); err == nil {
				t.Errorf("Compile(%s) error = nil, want an error", raw)
			}
		})
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

func TestCompile_Unsafe(t *testing.T) {
	deep := func(n int) string {
		defs := map[string]any{}
		for i := range n {
			next := fmt.Sprintf("#/$defs/d%d", i+1)
			defs[fmt.Sprintf("d%d", i)] = map[string]any{"allOf": []any{map[string]any{"$ref": next}, map[string]any{"$ref": next}}}
		}
		defs[fmt.Sprintf("d%d", n)] = map[string]any{"type": "object"}
		raw, _ := json.Marshal(map[string]any{"$defs": defs, "$ref": "#/$defs/d0"})
		return string(raw)
	}
	tests := []struct {
		name   string
		schema string
	}{
		{"self ref", `{"$ref":"#"}`},
		{"self ref via allOf", `{"allOf":[{"$ref":"#"}]}`},
		{"mutual refs", `{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}`},
		{"escaped pointer cycle", `{"$defs":{"a/b":{"$ref":"#/$defs/a~1b"}},"$ref":"#/$defs/a~1b"}`},
		{"remote ref", `{"$ref":"http://127.0.0.1:1/x"}`},
		{"exponential fan-out", deep(30)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := schema.Compile(json.RawMessage(tt.schema))
			if !errors.Is(err, schema.ErrUnsafe) {
				t.Fatalf("Compile() error = %v, want ErrUnsafe", err)
			}
		})
	}
}

func TestCompile_BoundedRecursion(t *testing.T) {
	v, err := schema.Compile(json.RawMessage(`{"type":"object","properties":{"child":{"$ref":"#"},"n":{"type":"integer"}}}`))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if err := v.Validate([]byte(`{"child":{"child":{"n":1}}}`)); err != nil {
		t.Errorf("Validate() error = %v", err)
	}
	if err := v.Validate([]byte(`{"child":{"child":{"n":"x"}}}`)); err == nil {
		t.Error("Validate() accepted invalid nested value")
	}
}

func TestCompile_ModerateFanOut(t *testing.T) {
	raw := `{"$defs":{"a":{"allOf":[{"$ref":"#/$defs/b"},{"$ref":"#/$defs/b"}]},"b":{"type":"object"}},"$ref":"#/$defs/a"}`
	v, err := schema.Compile(json.RawMessage(raw))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if err := v.Validate([]byte(`{}`)); err != nil {
		t.Errorf("Validate() error = %v", err)
	}
}
