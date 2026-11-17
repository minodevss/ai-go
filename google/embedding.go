package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	studioMaxBatch       = 100
	vertexMaxBatch       = 5
	nativeEmbeddingDims  = 3072
	embeddingTwoMarker   = "gemini-embedding-2"
	embeddingOneMarker   = "gemini-embedding-001"
	retrievalQueryPrefix = "task: search result | query: "
)

// EmbeddingModel embeds text, and for gemini-embedding-2 on AI Studio also images, audio, video and PDF.
// Its MaxBatch is 100 on AI Studio, 5 on Vertex and 1 for gemini-embedding-001 on Vertex.
type EmbeddingModel struct {
	backend *backend
	id      string
}

type studioEmbedRequest struct {
	Requests []studioEmbedItem `json:"requests"`
}

type studioEmbedItem struct {
	Model   string        `json:"model"`
	Content content       `json:"content"`
	Config  *studioConfig `json:"embedContentConfig,omitzero"`
}

type studioConfig struct {
	TaskType             string `json:"taskType,omitzero"`
	OutputDimensionality int    `json:"outputDimensionality,omitzero"`
}

type studioEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
	UsageMetadata json.RawMessage `json:"usageMetadata"`
}

type vertexEmbedRequest struct {
	Instances  []vertexInstance `json:"instances"`
	Parameters *vertexParams    `json:"parameters,omitzero"`
}

type vertexInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitzero"`
}

type vertexParams struct {
	OutputDimensionality int `json:"outputDimensionality,omitzero"`
}

type vertexEmbedResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values     []float32 `json:"values"`
			Statistics struct {
				TokenCount int `json:"token_count"`
			} `json:"statistics"`
		} `json:"embeddings"`
	} `json:"predictions"`
}

func (m *EmbeddingModel) Provider() string { return m.backend.name }

func (m *EmbeddingModel) ModelID() string { return m.id }

func (m *EmbeddingModel) MaxBatch() int {
	switch {
	case !m.backend.vertex:
		return studioMaxBatch
	case strings.Contains(m.id, embeddingOneMarker):
		return 1
	default:
		return vertexMaxBatch
	}
}

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	if len(c.Inputs) > m.MaxBatch() {
		return nil, fmt.Errorf("google: embed: %d inputs exceed the limit of %d per request", len(c.Inputs), m.MaxBatch())
	}
	if m.backend.vertex {
		return m.embedVertex(ctx, c)
	}
	return m.embedStudio(ctx, c)
}

func (m *EmbeddingModel) embedStudio(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	modelTwo := strings.Contains(m.id, embeddingTwoMarker)
	name := "models/" + strings.TrimPrefix(m.id, "models/")
	req := studioEmbedRequest{Requests: make([]studioEmbedItem, len(c.Inputs))}
	for i, in := range c.Inputs {
		parts, textOnly, err := embeddingParts(in, modelTwo)
		if err != nil {
			return nil, fmt.Errorf("google: embed input %d: %w", i, err)
		}
		item := studioEmbedItem{Model: name, Content: content{Parts: parts}}
		switch {
		case modelTwo && textOnly:
			item.Content = content{Parts: []json.RawMessage{pdata.JSON(providerName, partOut{Text: new(embeddingTwoPrompt(in, c.InputType))})}}
		case !modelTwo:
			item.Config = &studioConfig{TaskType: taskType(c.InputType)}
		default:
		}
		if c.Dimensions > 0 {
			if item.Config == nil {
				item.Config = &studioConfig{}
			}
			item.Config.OutputDimensionality = c.Dimensions
		}
		req.Requests[i] = item
	}
	var body studioEmbedResponse
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":batchEmbedContents", Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, fail(err)
	}
	if len(body.Embeddings) != len(c.Inputs) {
		return nil, fmt.Errorf("google: embed: got %d embeddings for %d inputs", len(body.Embeddings), len(c.Inputs))
	}
	out := make([][]float32, len(body.Embeddings))
	for i, e := range body.Embeddings {
		out[i] = e.Values
	}
	var warnings []ai.Warning
	if !modelTwo && c.Dimensions > 0 && c.Dimensions < nativeEmbeddingDims {
		for _, v := range out {
			normalize(v)
		}
		warnings = append(warnings, ai.Warning{Message: "google: embeddings below 3072 dimensions were L2-normalized because the API returns them unnormalized for this model"})
	}
	var usage struct {
		PromptTokenCount int `json:"promptTokenCount"`
	}
	if len(body.UsageMetadata) > 0 {
		if err := json.Unmarshal(body.UsageMetadata, &usage); err != nil {
			usage.PromptTokenCount = 0
		}
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: ai.Usage{Input: usage.PromptTokenCount, Raw: body.UsageMetadata}, Warnings: warnings, Header: header}, nil
}

func (m *EmbeddingModel) embedVertex(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	req := vertexEmbedRequest{Instances: make([]vertexInstance, len(c.Inputs))}
	for i, in := range c.Inputs {
		text, err := embeddingText(in)
		if err != nil {
			return nil, fmt.Errorf("google: embed input %d: %w", i, err)
		}
		req.Instances[i] = vertexInstance{Content: text, TaskType: taskType(c.InputType)}
	}
	if c.Dimensions > 0 {
		req.Parameters = &vertexParams{OutputDimensionality: c.Dimensions}
	}
	var body vertexEmbedResponse
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":predict", Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, fail(err)
	}
	if len(body.Predictions) != len(c.Inputs) {
		return nil, fmt.Errorf("google: embed: got %d embeddings for %d inputs", len(body.Predictions), len(c.Inputs))
	}
	out := make([][]float32, len(body.Predictions))
	tokens := 0
	for i, p := range body.Predictions {
		out[i] = p.Embeddings.Values
		tokens += p.Embeddings.Statistics.TokenCount
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: ai.Usage{Input: tokens}, Header: header}, nil
}

func taskType(t ai.InputType) string {
	switch t {
	case ai.InputQuery:
		return "RETRIEVAL_QUERY"
	case ai.InputDocument:
		return "RETRIEVAL_DOCUMENT"
	default:
		return ""
	}
}

func embeddingTwoPrompt(parts ai.Parts, t ai.InputType) string {
	text, err := embeddingText(parts)
	if err != nil {
		return ""
	}
	switch t {
	case ai.InputQuery:
		return retrievalQueryPrefix + text
	case ai.InputDocument:
		return "title: none | text: " + text
	default:
		return text
	}
}

func embeddingParts(parts ai.Parts, media bool) ([]json.RawMessage, bool, error) {
	out := make([]json.RawMessage, 0, len(parts))
	textOnly := true
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, pdata.JSON(providerName, partOut{Text: &v.Text}))
		case ai.File:
			if !media {
				return nil, false, errors.New("google: only text inputs are supported by this model")
			}
			raw, err := filePart(v)
			if err != nil {
				return nil, false, err
			}
			out = append(out, raw)
			textOnly = false
		default:
		}
	}
	if len(out) == 0 {
		return nil, false, errors.New("google: input has no text or file")
	}
	return out, textOnly, nil
}

func embeddingText(parts ai.Parts) (string, error) {
	var b strings.Builder
	for _, p := range parts {
		t, ok := p.(ai.Text)
		if !ok {
			return "", errors.New("google: only text inputs are supported")
		}
		b.WriteString(t.Text)
	}
	return b.String(), nil
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	norm := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= norm
	}
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
