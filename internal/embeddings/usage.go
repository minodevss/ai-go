package embeddings

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
)

func TotalTokensUsage(raw json.RawMessage) ai.Usage {
	var u struct {
		TotalTokens int `json:"total_tokens"`
	}
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{Input: u.TotalTokens, Raw: raw}
}
