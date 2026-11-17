package jina

import (
	"context"
	"encoding/json"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/embeddings"
	"github.com/minodevss/ai-go/internal/transport"
)

type RerankModel struct {
	client *transport.Client
	id     string
}

type rerankRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            int      `json:"top_n,omitzero"`
	ReturnDocuments bool     `json:"return_documents"`
	MaxDocLength    int      `json:"max_doc_length,omitzero"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

type rerankResponse struct {
	Results []rerankResult  `json:"results"`
	Usage   json.RawMessage `json:"usage"`
}

func (m *RerankModel) Provider() string { return providerName }

func (m *RerankModel) ModelID() string { return m.id }

func (m *RerankModel) Rerank(ctx context.Context, c *ai.RerankCall) (*ai.RerankResponse, error) {
	opts, err := decodeOptions[RerankOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	body := rerankRequest{Model: m.id, Query: c.Query, Documents: c.Documents, TopN: c.TopN, MaxDocLength: opts.MaxDocLength}
	var resp rerankResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: "/v1/rerank", Body: body, Header: c.Headers}, &resp)
	if err != nil {
		return nil, err
	}
	ranking := make([]ai.Ranked, len(resp.Results))
	for i, r := range resp.Results {
		ranking[i] = ai.Ranked{Index: r.Index, Score: r.RelevanceScore}
	}
	return &ai.RerankResponse{Ranking: ranking, Usage: embeddings.TotalTokensUsage(resp.Usage), Header: header}, nil
}

var _ ai.RerankModel = (*RerankModel)(nil)
