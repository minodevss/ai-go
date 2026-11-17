// Package pdata encodes and decodes the per-provider entries of ai.ProviderData.
package pdata

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

func JSON(provider string, v any) json.RawMessage {
	raw, err := transport.EncodeJSON(v)
	if err != nil {
		panic(provider + ": encode value: " + err.Error())
	}
	return raw
}

func Encode(provider string, v any) ai.ProviderData {
	return ai.ProviderData{provider: JSON(provider, v)}
}

func Decode[T any](provider string, d ai.ProviderData) T {
	var out T
	raw, ok := d[provider]
	if !ok {
		return out
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		var zero T
		return zero
	}
	return out
}
