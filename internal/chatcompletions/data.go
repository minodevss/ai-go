package chatcompletions

import "encoding/json"

const packageName = "chatcompletions"

type partData struct {
	Derived bool                       `json:"derived,omitzero"`
	Refusal bool                       `json:"refusal,omitzero"`
	Field   string                     `json:"field,omitzero"`
	Fields  map[string]json.RawMessage `json:"fields,omitzero"`
	Chunk   json.RawMessage            `json:"chunk,omitzero"`
}
