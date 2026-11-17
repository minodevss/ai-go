package schemafit_test

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
)

var rules = schemafit.Rules{
	Provider:    "p",
	Unsupported: []string{"minLength", "allOf", "default"},
	Renamed:     map[string]string{"oneOf": "anyOf"},
	Formats:     []string{"date-time"},
	Strict:      true,
}

func TestFit(t *testing.T) {
	tests := []struct {
		name         string
		schema       string
		want         string
		wantWarnings []ai.Warning
		wantStrict   bool
	}{
		{
			name:       "already strict",
			schema:     `{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`,
			want:       `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"required":["a"],"type":"object"}`,
			wantStrict: true,
		},
		{
			name:         "drops keywords at any depth",
			schema:       `{"type":"object","properties":{"a":{"type":"string","minLength":2,"default":"x"}},"required":["a"],"additionalProperties":false}`,
			want:         `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"required":["a"],"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: schema keyword "default" at /properties/a is not supported and was removed`}, {Message: `p: schema keyword "minLength" at /properties/a is not supported and was removed`}},
			wantStrict:   true,
		},
		{
			name:         "renames oneOf",
			schema:       `{"type":"object","properties":{"a":{"oneOf":[{"type":"string"},{"type":"integer"}]}},"required":["a"],"additionalProperties":false}`,
			want:         `{"additionalProperties":false,"properties":{"a":{"anyOf":[{"type":"string"},{"type":"integer"}]}},"required":["a"],"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: schema keyword "oneOf" at /properties/a is not supported and was replaced by "anyOf"`}},
			wantStrict:   true,
		},
		{
			name:         "drops unknown format",
			schema:       `{"type":"object","properties":{"a":{"type":"string","format":"uri"},"b":{"type":"string","format":"date-time"}},"required":["a","b"],"additionalProperties":false}`,
			want:         `{"additionalProperties":false,"properties":{"a":{"type":"string"},"b":{"format":"date-time","type":"string"}},"required":["a","b"],"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: schema format "uri" at /properties/a is not supported and was removed`}},
			wantStrict:   true,
		},
		{
			name:         "optional property turns strict off",
			schema:       `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}`,
			want:         `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: property "a" at the root is optional, so strict mode is off`}},
		},
		{
			name:         "missing additionalProperties is set",
			schema:       `{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`,
			want:         `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"required":["a"],"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: additionalProperties at the root was set to false`}},
			wantStrict:   true,
		},
		{
			name:         "open map turns strict off",
			schema:       `{"type":"object","additionalProperties":{"type":"string"}}`,
			want:         `{"additionalProperties":{"type":"string"},"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: object at the root has a schema for additional properties, so strict mode is off`}},
		},
		{
			name:         "nested in defs and arrays",
			schema:       `{"type":"object","properties":{"l":{"type":"array","items":{"$ref":"#/$defs/x"}}},"required":["l"],"additionalProperties":false,"$defs":{"x":{"type":"object","properties":{"n":{"type":"integer","allOf":[]}},"required":["n"],"additionalProperties":false}}}`,
			want:         `{"$defs":{"x":{"additionalProperties":false,"properties":{"n":{"type":"integer"}},"required":["n"],"type":"object"}},"additionalProperties":false,"properties":{"l":{"items":{"$ref":"#/$defs/x"},"type":"array"}},"required":["l"],"type":"object"}`,
			wantWarnings: []ai.Warning{{Message: `p: schema keyword "allOf" at /$defs/x/properties/n is not supported and was removed`}},
			wantStrict:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := schemafit.Fit(json.RawMessage(tt.schema), rules)
			if err != nil {
				t.Fatalf("Fit() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, string(got.Schema)); diff != "" {
				t.Errorf("Fit() schema mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantWarnings, got.Warnings); diff != "" {
				t.Errorf("Fit() warnings mismatch (-want +got):\n%s", diff)
			}
			if got.Strict != tt.wantStrict {
				t.Errorf("Fit() strict = %v, want %v", got.Strict, tt.wantStrict)
			}
		})
	}
}

func TestFit_invalidJSON(t *testing.T) {
	if _, err := schemafit.Fit(json.RawMessage(`{`), rules); err == nil {
		t.Error("Fit() error = nil, want error")
	}
}
