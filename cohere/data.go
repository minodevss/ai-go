package cohere

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

type partData struct {
	Thinking bool            `json:"thinking,omitzero"`
	ToolPlan bool            `json:"toolPlan,omitzero"`
	Citation json.RawMessage `json:"citation,omitzero"`
}

func responseProvider(data ResponseData) ai.ProviderData {
	if data.ID == "" && len(data.Citations) == 0 {
		return nil
	}
	return pdata.Encode(providerName, data)
}
