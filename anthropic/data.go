package anthropic

import (
	"encoding/json"
	"regexp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

type partData struct {
	Block        json.RawMessage `json:"block,omitzero"`
	ForcedTool   bool            `json:"forcedTool,omitzero"`
	CacheControl *CacheControl   `json:"cacheControl,omitzero"`
	FileID       string          `json:"fileId,omitzero"`
}

var validCallID = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

func dataOf(d ai.ProviderData) partData {
	return pdata.Decode[partData](providerName, d)
}

func blockData(raw json.RawMessage) ai.ProviderData {
	return pdata.Encode(providerName, partData{Block: raw})
}
