package schemafit_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
)

var closedRules = schemafit.Rules{Provider: "p"}

func TestFitClosed(t *testing.T) {
	tests := []struct {
		name         string
		schema       string
		limits       schemafit.Limits
		want         string
		wantWarnings []ai.Warning
	}{
		{
			name:   "closes open objects at any depth",
			schema: `{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"string"}},"additionalProperties":true},"c":{"type":"array","items":{"type":"object","properties":{},"additionalProperties":{"type":"string"}}}},"required":["a","c"]}`,
			want:   `{"additionalProperties":false,"properties":{"a":{"additionalProperties":false,"properties":{"b":{"type":"string"}},"type":"object"},"c":{"items":{"additionalProperties":false,"properties":{},"type":"object"},"type":"array"}},"required":["a","c"],"type":"object"}`,
			wantWarnings: []ai.Warning{
				{Message: "p: additionalProperties at the root was set to false"},
				{Message: "p: additionalProperties at /properties/a was changed from true to false"},
				{Message: "p: additionalProperties at /properties/c/items was changed from a schema to false"},
			},
		},
		{
			name:   "limits array bounds",
			schema: `{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":5}`,
			want:   `{"items":{"type":"string"},"type":"array"}`,
			wantWarnings: []ai.Warning{
				{Message: `p: schema keyword "maxItems" at the root is not supported and was removed`},
				{Message: `p: schema keyword "minItems" at the root supports only 0 and 1 and was removed`},
			},
		},
		{
			name:   "keeps minItems of one",
			schema: `{"type":"array","items":{"type":"string"},"minItems":1}`,
			want:   `{"items":{"type":"string"},"minItems":1,"type":"array"}`,
		},
		{
			name:   "counts optional properties and unions over the limits",
			schema: `{"type":"object","properties":{"a":{"type":["string","null"]},"b":{"anyOf":[{"type":"string"},{"type":"integer"}]}},"additionalProperties":false}`,
			limits: schemafit.Limits{MaxOptional: 1, MaxUnion: 1},
			want:   `{"additionalProperties":false,"properties":{"a":{"type":["string","null"]},"b":{"anyOf":[{"type":"string"},{"type":"integer"}]}},"type":"object"}`,
			wantWarnings: []ai.Warning{
				{Message: "p: schema has 2 optional properties and the API allows 1, so it may be rejected as too complex: list properties in required and make them nullable"},
				{Message: "p: schema has 2 properties with a type array or anyOf and the API allows 1"},
			},
		},
		{
			name:   "no limits means no counting warnings",
			schema: `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}`,
			want:   `{"additionalProperties":false,"properties":{"a":{"type":"string"}},"type":"object"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := schemafit.FitClosed(json.RawMessage(tt.schema), closedRules, tt.limits)
			if err != nil {
				t.Fatalf("FitClosed() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, string(got.Schema)); diff != "" {
				t.Errorf("FitClosed() schema mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantWarnings, got.Warnings); diff != "" {
				t.Errorf("FitClosed() warnings mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFitClosed_invalidJSON(t *testing.T) {
	_, err := schemafit.FitClosed(json.RawMessage(`{`), closedRules, schemafit.Limits{})
	if err == nil || !strings.HasPrefix(err.Error(), "p: ") {
		t.Errorf("FitClosed() error = %v, want a p: prefixed error", err)
	}
}
