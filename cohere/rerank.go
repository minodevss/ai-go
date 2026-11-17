package cohere

import (
	"context"
	"errors"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const rerankPath = "/v2/rerank"

// RerankModel scores documents against a query. Scores are relevance values in [0, 1] and are not proportional.
type RerankModel struct {
	client *transport.Client
	id     string
}

type rerankRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            int      `json:"top_n,omitzero"`
	MaxTokensPerDoc int      `json:"max_tokens_per_doc,omitzero"`
	Priority        int      `json:"priority,omitzero"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	Meta metaBody `json:"meta"`
}

func (m *RerankModel) Provider() string { return providerName }

func (m *RerankModel) ModelID() string { return m.id }

func (m *RerankModel) Rerank(ctx context.Context, c *ai.RerankCall) (*ai.RerankResponse, error) {
	if len(c.Documents) == 0 {
		return nil, errors.New("cohere: rerank: no documents")
	}
	opts, err := optionsOf[RerankOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	req := &rerankRequest{
		Model:           m.id,
		Query:           c.Query,
		Documents:       c.Documents,
		TopN:            c.TopN,
		MaxTokensPerDoc: opts.MaxTokensPerDoc,
		Priority:        opts.Priority,
	}
	var body rerankResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: rerankPath, Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	ranking := make([]ai.Ranked, len(body.Results))
	for i, r := range body.Results {
		if r.Index < 0 || r.Index >= len(c.Documents) {
			return nil, fmt.Errorf("cohere: rerank: index %d out of range for %d documents", r.Index, len(c.Documents))
		}
		ranking[i] = ai.Ranked{Index: r.Index, Score: r.RelevanceScore}
	}
	return &ai.RerankResponse{Ranking: ranking, Usage: body.Meta.usage(), Warnings: body.Meta.warnings(), Header: header}, nil
}

var _ ai.RerankModel = (*RerankModel)(nil)
