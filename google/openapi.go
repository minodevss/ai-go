package google

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go/internal/schemafit"
)

func openAPISchema(raw json.RawMessage) (schemafit.Result, error) {
	fit, err := schemafit.Fit(raw, openAPIRules)
	if err != nil {
		return schemafit.Result{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(fit.Schema))
	dec.UseNumber()
	var root any
	if err := dec.Decode(&root); err != nil {
		return schemafit.Result{}, fmt.Errorf("google: parse schema: %w", err)
	}
	out, err := json.Marshal(upperTypes(root))
	if err != nil {
		return schemafit.Result{}, fmt.Errorf("google: encode schema: %w", err)
	}
	fit.Schema = out
	return fit, nil
}

func upperTypes(v any) any {
	switch n := v.(type) {
	case map[string]any:
		for k, child := range n {
			n[k] = upperTypes(child)
		}
		switch t := n["type"].(type) {
		case string:
			n["type"] = strings.ToUpper(t)
		case []any:
			var kept []any
			for _, e := range t {
				if s, ok := e.(string); ok && s == "null" {
					n["nullable"] = true
					continue
				}
				if s, ok := e.(string); ok {
					kept = append(kept, strings.ToUpper(s))
				}
			}
			if len(kept) == 1 {
				n["type"] = kept[0]
			} else {
				n["type"] = kept
			}
		default:
		}
		return n
	case []any:
		for i, child := range n {
			n[i] = upperTypes(child)
		}
		return n
	default:
		return v
	}
}
