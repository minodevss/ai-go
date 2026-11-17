package cohere

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	embedPath     = "/v2/embed"
	maxEmbedBatch = 96
)

// EmbeddingModel embeds text and images as float vectors. Its MaxBatch is the API limit of 96 inputs per request, which ai.Embed uses to split larger requests.
type EmbeddingModel struct {
	client *transport.Client
	id     string
}

type embedRequest struct {
	Model          string       `json:"model"`
	InputType      string       `json:"input_type"`
	EmbeddingTypes []string     `json:"embedding_types"`
	Texts          []string     `json:"texts,omitzero"`
	Inputs         []embedInput `json:"inputs,omitzero"`
	MaxTokens      int          `json:"max_tokens,omitzero"`
	OutputDim      int          `json:"output_dimension,omitzero"`
	Truncate       string       `json:"truncate,omitzero"`
	Priority       int          `json:"priority,omitzero"`
}

type embedInput struct {
	Content []any `json:"content"`
}

type embedResponse struct {
	Embeddings struct {
		Float [][]float32 `json:"float"`
	} `json:"embeddings"`
	Meta metaBody `json:"meta"`
}

type metaBody struct {
	BilledUnits struct {
		InputTokens int `json:"input_tokens"`
		ImageTokens int `json:"image_tokens"`
		TextTokens  int `json:"text_tokens"`
	} `json:"billed_units"`
	Tokens struct {
		InputTokens int `json:"input_tokens"`
	} `json:"tokens"`
	CachedTokens float64  `json:"cached_tokens"`
	Warnings     []string `json:"warnings"`
}

func (m *EmbeddingModel) Provider() string { return providerName }

func (m *EmbeddingModel) ModelID() string { return m.id }

func (m *EmbeddingModel) MaxBatch() int { return maxEmbedBatch }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	if len(c.Inputs) > maxEmbedBatch {
		return nil, fmt.Errorf("cohere: embed: %d inputs exceed the limit of %d per request", len(c.Inputs), maxEmbedBatch)
	}
	opts, err := optionsOf[EmbeddingOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	req := &embedRequest{
		Model:          m.id,
		InputType:      embedInputType(c.InputType, opts),
		EmbeddingTypes: []string{"float"},
		MaxTokens:      opts.MaxTokens,
		OutputDim:      c.Dimensions,
		Truncate:       opts.Truncate,
		Priority:       opts.Priority,
	}
	if err := fillInputs(req, c.Inputs); err != nil {
		return nil, err
	}
	var body embedResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: embedPath, Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	if len(body.Embeddings.Float) != len(c.Inputs) {
		return nil, fmt.Errorf("cohere: embed: got %d embeddings for %d inputs", len(body.Embeddings.Float), len(c.Inputs))
	}
	return &ai.EmbedResponse{Embeddings: body.Embeddings.Float, Usage: body.Meta.usage(), Warnings: body.Meta.warnings(), Header: header}, nil
}

func embedInputType(t ai.InputType, opts EmbeddingOptions) string {
	switch {
	case opts.InputType != "":
		return opts.InputType
	case t == ai.InputQuery:
		return "search_query"
	default:
		return "search_document"
	}
}

func fillInputs(req *embedRequest, inputs []ai.Parts) error {
	textOnly := true
	for _, in := range inputs {
		for _, p := range in {
			if _, ok := p.(ai.Text); !ok {
				textOnly = false
			}
		}
	}
	for i, in := range inputs {
		if textOnly {
			var text strings.Builder
			for _, p := range in {
				if t, ok := p.(ai.Text); ok {
					text.WriteString(t.Text)
				}
			}
			req.Texts = append(req.Texts, text.String())
			continue
		}
		content, err := embedContent(in)
		if err != nil {
			return fmt.Errorf("%w (input %d)", err, i)
		}
		req.Inputs = append(req.Inputs, embedInput{Content: content})
	}
	return nil
}

func embedContent(parts ai.Parts) ([]any, error) {
	out := make([]any, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, textBlock{Type: "text", Text: v.Text})
		case ai.File:
			if len(v.Data) == 0 {
				return nil, errors.New("cohere: embed: an image input needs Data")
			}
			out = append(out, imageBlock{Type: "image_url", ImageURL: imageURL{URL: fileURL(v)}})
		default:
			return nil, fmt.Errorf("cohere: embed: input part %T is not supported", p)
		}
	}
	return out, nil
}

func (m metaBody) usage() ai.Usage {
	in := m.BilledUnits.InputTokens
	if in == 0 {
		in = m.BilledUnits.TextTokens + m.BilledUnits.ImageTokens
	}
	if in == 0 {
		in = m.Tokens.InputTokens
	}
	return ai.Usage{Input: in, CacheRead: min(int(m.CachedTokens), in)}
}

func (m metaBody) warnings() []ai.Warning {
	var out []ai.Warning
	for _, w := range m.Warnings {
		out = append(out, ai.Warning{Message: "cohere: " + w})
	}
	return out
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
