package google

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
)

const dummySignature = "skip_thought_signature_validator"

type partData struct {
	Part        json.RawMessage `json:"part,omitzero"`
	GeneratedID bool            `json:"generatedId,omitzero"`
}

// ResponseData is stored in Response.Provider under "google" (AI Studio) or "google-vertex".
// The metadata fields hold the API objects as received; streamed fragments are merged.
type ResponseData struct {
	ResponseID         string          `json:"responseId,omitzero"`
	ModelVersion       string          `json:"modelVersion,omitzero"`
	FinishMessage      string          `json:"finishMessage,omitzero"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata,omitzero"`
	URLContextMetadata json.RawMessage `json:"urlContextMetadata,omitzero"`
	CitationMetadata   json.RawMessage `json:"citationMetadata,omitzero"`
	SafetyRatings      json.RawMessage `json:"safetyRatings,omitzero"`
	PromptFeedback     json.RawMessage `json:"promptFeedback,omitzero"`
}

type citationData struct {
	ChunkIndex int       `json:"groundingChunkIndex"`
	Confidence []float64 `json:"confidenceScores,omitzero"`
}

var validCallID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func (b *backend) encode(d partData) ai.ProviderData {
	return pdata.Encode(b.name, d)
}

func (b *backend) decode(d ai.ProviderData) partData {
	return pdata.Decode[partData](b.name, d)
}

func (b *backend) response(d ResponseData) ai.ProviderData {
	raw := pdata.JSON(b.name, d)
	if len(raw) == len("{}") {
		return nil
	}
	return ai.ProviderData{b.name: raw}
}

func sendableID(id string) string {
	if validCallID.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return "call_" + hex.EncodeToString(sum[:])[:16]
}

func generatedID(prefix, responseID, name string, args json.RawMessage, index int) string {
	h := sha256.New()
	for _, s := range []string{responseID, name, string(args)} {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	h.Write([]byte{byte(index), byte(index >> 8)})
	return prefix + hex.EncodeToString(h.Sum(nil))[:12]
}
