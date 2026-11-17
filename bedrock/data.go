package bedrock

import (
	"encoding/json"
	"regexp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

type cachePointData struct {
	TTL CacheTTL `json:"ttl,omitzero"`
}

type partData struct {
	Block      json.RawMessage `json:"block,omitzero"`
	Model      string          `json:"model,omitzero"`
	ForcedTool bool            `json:"forcedTool,omitzero"`
	CachePoint *cachePointData `json:"cachePoint,omitzero"`
	SystemTool string          `json:"systemTool,omitzero"`
}

var validCallID = regexp.MustCompile(`^[A-Za-z0-9_.:-]+$`)

func dataOf(d ai.ProviderData) partData {
	return pdata.Decode[partData](providerName, d)
}

func blockData(raw json.RawMessage, model string) ai.ProviderData {
	return pdata.Encode(providerName, partData{Block: raw, Model: model})
}
