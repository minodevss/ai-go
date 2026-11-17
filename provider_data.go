package ai

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

type ProviderData map[string]json.RawMessage

// ProviderOptions is implemented by a provider package's option structs; ProviderName is the key they are stored under in ProviderData and must equal the provider's name.
type ProviderOptions interface {
	ProviderName() string
}

const unknownFieldsKey = "unknown"

// WithRaw returns a copy of d with key set to the already encoded JSON raw.
func (d ProviderData) WithRaw(key string, raw json.RawMessage) ProviderData {
	out := maps.Clone(d)
	if out == nil {
		out = ProviderData{}
	}
	out[key] = slices.Clone(raw)
	return out
}

// WithValue returns a copy of d with key set to the JSON encoding of v, or the encoding error.
func (d ProviderData) WithValue(key string, v any) (ProviderData, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return d, fmt.Errorf("ai: ProviderData.WithValue %q: %w", key, err)
	}
	return d.WithRaw(key, raw), nil
}

func ProviderValue[T any](d ProviderData, key string) (T, bool) {
	var v T
	raw, ok := d[key]
	if !ok {
		return v, false
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		return *new(T), false
	}
	return v, true
}

func Providers(opts ...ProviderOptions) ProviderData {
	if len(opts) == 0 {
		return nil
	}
	out := make(ProviderData, len(opts))
	for _, o := range opts {
		if next, err := out.WithValue(o.ProviderName(), o); err == nil {
			out = next
		}
	}
	return out
}

func (d ProviderData) clone() ProviderData {
	if len(d) == 0 {
		return nil
	}
	out := make(ProviderData, len(d))
	for k, v := range d {
		out[k] = append(json.RawMessage(nil), v...)
	}
	return out
}

func (d ProviderData) merged(defaults ProviderData) ProviderData {
	if len(defaults) == 0 {
		return d
	}
	out := d.clone()
	if out == nil {
		out = make(ProviderData, len(defaults))
	}
	for k, v := range defaults {
		if _, ok := out[k]; !ok {
			out[k] = append(json.RawMessage(nil), v...)
		}
	}
	return out
}
