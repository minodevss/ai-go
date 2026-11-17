package voyage

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/embeddings"
	"github.com/minodevss/ai-go/internal/transport"
)

const maxRerankDocuments = 1000

// RerankModel scores up to 1000 documents per call against a query.
type RerankModel struct {
	client *transport.Client
	id     string
}

type rerankRequest struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	Model      string   `json:"model"`
	TopK       int      `json:"top_k,omitzero"`
	Truncation *bool    `json:"truncation,omitzero"`
}

type rerankResponse struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"data"`
	Usage json.RawMessage `json:"usage"`
}

func (m *RerankModel) Provider() string { return providerName }

func (m *RerankModel) ModelID() string { return m.id }

func (m *RerankModel) Rerank(ctx context.Context, c *ai.RerankCall) (*ai.RerankResponse, error) {
	if len(c.Documents) > maxRerankDocuments {
		return nil, fmt.Errorf("voyage: rerank: %d documents exceed the limit of %d per request", len(c.Documents), maxRerankDocuments)
	}
	opts, err := optionsOf[RerankOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	body := rerankRequest{Query: c.Query, Documents: c.Documents, Model: m.id, TopK: c.TopN, Truncation: opts.Truncation}
	var resp rerankResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: "/v1/rerank", Body: body, Header: c.Headers}, &resp)
	if err != nil {
		return nil, err
	}
	ranking := make([]ai.Ranked, len(resp.Data))
	for i, d := range resp.Data {
		ranking[i] = ai.Ranked{Index: d.Index, Score: d.RelevanceScore}
	}
	return &ai.RerankResponse{Ranking: ranking, Usage: embeddings.TotalTokensUsage(resp.Usage), Header: header}, nil
}

var _ ai.RerankModel = (*RerankModel)(nil)
