package bedrock

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
)

const (
	maxOptionalProperties = 12
	maxUnionProperties    = 16
)

func schemaRules() schemafit.Rules {
	// Structured output accepts a subset of JSON Schema 2020-12: https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
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

func fitSchema(raw json.RawMessage) (string, []ai.Warning, error) {
	fit, err := schemafit.FitClosed(raw, schemaRules(), schemafit.Limits{MaxOptional: maxOptionalProperties, MaxUnion: maxUnionProperties})
	if err != nil {
		return "", nil, err
	}
	return string(fit.Schema), fit.Warnings, nil
}
