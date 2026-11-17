package anthropic

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
)

func schemaRules() schemafit.Rules {
	return schemafit.Rules{
		Provider: providerName,
		Unsupported: []string{
			"$schema", "not", "if", "then", "else", "dependentRequired", "dependentSchemas",
			"contains", "minContains", "maxContains", "propertyNames", "patternProperties",
			"unevaluatedProperties", "unevaluatedItems", "prefixItems", "uniqueItems",
			"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
			"minLength", "maxLength", "minProperties", "maxProperties", "examples",
		},
		Renamed: map[string]string{"oneOf": "anyOf"},
		Formats: []string{"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"},
	}
}

func fitSchema(raw json.RawMessage) (json.RawMessage, []ai.Warning, error) {
	fit, err := schemafit.FitClosed(raw, schemaRules(), schemafit.Limits{})
	if err != nil {
		return nil, nil, err
	}
	return fit.Schema, fit.Warnings, nil
}
