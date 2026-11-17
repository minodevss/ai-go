package schemafit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/minodevss/ai-go"
)

type Limits struct {
	MaxOptional int
	MaxUnion    int
}

type closer struct {
	provider string
	warnings []ai.Warning
	optional int
	unions   int
}

func FitClosed(raw json.RawMessage, rules Rules, limits Limits) (Result, error) {
	fit, err := Fit(raw, rules)
	if err != nil {
		return Result{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(fit.Schema))
	dec.UseNumber()
	var root any
	if err := dec.Decode(&root); err != nil {
		return Result{}, fmt.Errorf("%s: parse schema: %w", rules.Provider, err)
	}
	c := &closer{provider: rules.Provider}
	root = c.node(root, "")
	out, err := json.Marshal(root)
	if err != nil {
		return Result{}, fmt.Errorf("%s: encode schema: %w", rules.Provider, err)
	}
	if limits.MaxOptional > 0 && c.optional > limits.MaxOptional {
		c.warn("schema has %d optional properties and the API allows %d, so it may be rejected as too complex: list properties in required and make them nullable", c.optional, limits.MaxOptional)
	}
	if limits.MaxUnion > 0 && c.unions > limits.MaxUnion {
		c.warn("schema has %d properties with a type array or anyOf and the API allows %d", c.unions, limits.MaxUnion)
	}
	return Result{Schema: out, Warnings: slices.Concat(fit.Warnings, c.warnings), Strict: fit.Strict}, nil
}

func (c *closer) warn(format string, args ...any) {
	c.warnings = append(c.warnings, ai.Warning{Message: c.provider + ": " + fmt.Sprintf(format, args...)})
}

func (c *closer) node(v any, path string) any {
	obj, ok := v.(map[string]any)
	if !ok {
		return v
	}
	c.limits(obj, path)
	c.object(obj, path)
	for _, key := range []string{"properties", "$defs", "definitions"} {
		if m, ok := obj[key].(map[string]any); ok {
			for _, name := range sortedKeys(m) {
				m[name] = c.node(m[name], path+"/"+key+"/"+name)
			}
		}
	}
	for _, key := range []string{"anyOf", "allOf"} {
		if list, ok := obj[key].([]any); ok {
			for i := range list {
				list[i] = c.node(list[i], fmt.Sprintf("%s/%s/%d", path, key, i))
			}
		}
	}
	if child, ok := obj["items"]; ok {
		obj["items"] = c.node(child, path+"/items")
	}
	return obj
}

func (c *closer) limits(obj map[string]any, path string) {
	if _, ok := obj["maxItems"]; ok {
		delete(obj, "maxItems")
		c.warn("schema keyword %q at %s is not supported and was removed", "maxItems", at(path))
	}
	if n, ok := obj["minItems"].(json.Number); ok && n.String() != "0" && n.String() != "1" {
		delete(obj, "minItems")
		c.warn("schema keyword %q at %s supports only 0 and 1 and was removed", "minItems", at(path))
	}
	if _, ok := obj["anyOf"]; ok || isUnion(obj["type"]) {
		c.unions++
	}
}

func (c *closer) object(obj map[string]any, path string) {
	props, hasProps := obj["properties"].(map[string]any)
	if !hasProps && !isType(obj["type"], "object") {
		return
	}
	required := toStrings(obj["required"])
	for name := range props {
		if !slices.Contains(required, name) {
			c.optional++
		}
	}
	switch ap := obj["additionalProperties"].(type) {
	case nil:
		obj["additionalProperties"] = false
		c.warn("additionalProperties at %s was set to false", at(path))
	case bool:
		if ap {
			obj["additionalProperties"] = false
			c.warn("additionalProperties at %s was changed from true to false", at(path))
		}
	default:
		obj["additionalProperties"] = false
		c.warn("additionalProperties at %s was changed from a schema to false", at(path))
	}
}

func isUnion(t any) bool {
	list, ok := t.([]any)
	return ok && len(list) > 1
}
