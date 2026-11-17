// Package schemafit rewrites JSON Schemas for providers that accept only a subset and reports every change as a warning.
package schemafit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
)

type Rules struct {
	Provider    string
	Unsupported []string
	Renamed     map[string]string
	Formats     []string
	Strict      bool
}

type Result struct {
	Schema   json.RawMessage
	Warnings []ai.Warning
	Strict   bool
}

type fitter struct {
	rules    Rules
	warnings []ai.Warning
	strict   bool
}

func schemaKeywords() []string {
	return []string{"items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames", "unevaluatedItems", "unevaluatedProperties"}
}

func schemaMapKeys() []string {
	return []string{"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
}

func schemaListKeys() []string {
	return []string{"anyOf", "oneOf", "allOf", "prefixItems"}
}

func Fit(raw json.RawMessage, rules Rules) (Result, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var root any
	if err := dec.Decode(&root); err != nil {
		return Result{}, fmt.Errorf("%s: parse schema: %w", rules.Provider, err)
	}
	f := &fitter{rules: rules, strict: rules.Strict}
	root = f.node(root, "")
	out, err := json.Marshal(root)
	if err != nil {
		return Result{}, fmt.Errorf("%s: encode schema: %w", rules.Provider, err)
	}
	return Result{Schema: out, Warnings: f.warnings, Strict: f.strict}, nil
}

func (f *fitter) warn(format string, args ...any) {
	f.warnings = append(f.warnings, ai.Warning{Message: f.rules.Provider + ": " + fmt.Sprintf(format, args...)})
}

func (f *fitter) node(v any, path string) any {
	obj, ok := v.(map[string]any)
	if !ok {
		return v
	}
	for _, key := range sortedKeys(obj) {
		if slices.Contains(f.rules.Unsupported, key) {
			delete(obj, key)
			f.warn("schema keyword %q at %s is not supported and was removed", key, at(path))
			continue
		}
		if to, ok := f.rules.Renamed[key]; ok {
			if _, exists := obj[to]; !exists {
				obj[to] = obj[key]
				delete(obj, key)
				f.warn("schema keyword %q at %s is not supported and was replaced by %q", key, at(path), to)
			}
		}
	}
	f.checkFormat(obj, path)
	f.children(obj, path)
	if f.rules.Strict {
		f.strictObject(obj, path)
	}
	return obj
}

func (f *fitter) checkFormat(obj map[string]any, path string) {
	if len(f.rules.Formats) == 0 {
		return
	}
	format, ok := obj["format"].(string)
	if !ok || slices.Contains(f.rules.Formats, format) {
		return
	}
	delete(obj, "format")
	f.warn("schema format %q at %s is not supported and was removed", format, at(path))
}

func (f *fitter) children(obj map[string]any, path string) {
	for _, key := range schemaMapKeys() {
		if m, ok := obj[key].(map[string]any); ok {
			for _, name := range sortedKeys(m) {
				m[name] = f.node(m[name], path+"/"+key+"/"+name)
			}
		}
	}
	for _, key := range schemaListKeys() {
		if list, ok := obj[key].([]any); ok {
			for i := range list {
				list[i] = f.node(list[i], fmt.Sprintf("%s/%s/%d", path, key, i))
			}
		}
	}
	for _, key := range schemaKeywords() {
		if child, ok := obj[key].(map[string]any); ok {
			obj[key] = f.node(child, path+"/"+key)
		}
	}
}

func (f *fitter) strictObject(obj map[string]any, path string) {
	props, hasProps := obj["properties"].(map[string]any)
	if !hasProps && !isType(obj["type"], "object") {
		return
	}
	switch ap := obj["additionalProperties"].(type) {
	case nil:
		if hasProps {
			obj["additionalProperties"] = false
			f.warn("additionalProperties at %s was set to false", at(path))
		} else {
			f.strict = false
			f.warn("object at %s has no properties, so strict mode is off", at(path))
		}
	case bool:
		if ap {
			f.strict = false
			f.warn("object at %s allows additional properties, so strict mode is off", at(path))
		}
	default:
		f.strict = false
		f.warn("object at %s has a schema for additional properties, so strict mode is off", at(path))
	}
	required := toStrings(obj["required"])
	for _, name := range sortedKeys(props) {
		if !slices.Contains(required, name) {
			f.strict = false
			f.warn("property %q at %s is optional, so strict mode is off", name, at(path))
		}
	}
}

func isType(t any, want string) bool {
	switch v := t.(type) {
	case string:
		return v == want
	case []any:
		return slices.Contains(toStrings(v), want)
	default:
		return false
	}
}

func toStrings(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, e := range list {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

func at(path string) string {
	if path == "" {
		return "the root"
	}
	return strings.ReplaceAll(path, "~", "~0")
}
