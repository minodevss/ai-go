package openai

import (
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
)

type partData struct {
	ID      string          `json:"id,omitzero"`
	Item    json.RawMessage `json:"item,omitzero"`
	Index   int             `json:"index,omitzero"`
	Derived bool            `json:"derived,omitzero"`
	Refusal bool            `json:"refusal,omitzero"`
}

// FileID references a file uploaded to OpenAI instead of sending its bytes; pass it as File.Provider.
func FileID(id string) ai.ProviderData {
	return ai.ProviderData{providerName: openaiwire.EncodeFileRef(openaiwire.FileRef{FileID: id})}
}

// FileIDOf returns the file id held by a File that UploadFile, GetFile, ListFiles or FileID produced.
func FileIDOf(f ai.File) (string, bool) {
	id := openaiwire.FileRefOf(providerName, f.Provider).FileID
	return id, id != ""
}
