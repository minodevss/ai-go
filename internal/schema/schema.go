package schema

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/google/jsonschema-go/jsonschema"
)

type Validator struct {
	resolved *jsonschema.Resolved
}

type customSchema interface {
	JSONSchema() json.RawMessage
}

func For[T any]() (json.RawMessage, error) {
	t := reflect.TypeFor[T]()
	overrides := map[reflect.Type]*jsonschema.Schema{}
	if err := collectOverrides(t, map[reflect.Type]bool{}, overrides); err != nil {
		return nil, err
	}
	s, err := jsonschema.For[T](&jsonschema.ForOptions{TypeSchemas: overrides})
	if err != nil {
		return nil, err
	}
	return json.Marshal(s)
}

func Compile(raw json.RawMessage) (*Validator, error) {
	var doc any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("schema: parse: %w", err)
	}
	if err := checkSafe(doc); err != nil {
		return nil, err
	}
	var s jsonschema.Schema
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("schema: parse: %w", err)
	}
	resolved, err := s.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("schema: resolve: %w", err)
	}
	return &Validator{resolved: resolved}, nil
}

func (v *Validator) Validate(doc []byte) error {
	var instance any
	if err := json.Unmarshal(doc, &instance); err != nil {
		return fmt.Errorf("schema: invalid JSON: %w", err)
	}
	return v.resolved.Validate(instance)
}

func collectOverrides(t reflect.Type, seen map[reflect.Type]bool, out map[reflect.Type]*jsonschema.Schema) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if seen[t] {
		return nil
	}
	seen[t] = true
	if reflect.PointerTo(t).Implements(reflect.TypeFor[customSchema]()) {
		c, ok := reflect.TypeAssert[customSchema](reflect.New(t))
		if !ok {
			return nil
		}
		var s jsonschema.Schema
		if err := json.Unmarshal(c.JSONSchema(), &s); err != nil {
			return fmt.Errorf("schema: override for %s: %w", t, err)
		}
		out[t] = &s
		return nil
	}
	switch t.Kind() {
	case reflect.Struct:
		for field := range t.Fields() {
			if !field.IsExported() {
				continue
			}
			if err := collectOverrides(field.Type, seen, out); err != nil {
				return err
			}
		}
	case reflect.Slice, reflect.Array, reflect.Map:
		return collectOverrides(t.Elem(), seen, out)
	default:
	}
	return nil
}
