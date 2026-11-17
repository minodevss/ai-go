package openai

import (
	"context"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/embeddings"
)

const maxEmbeddingBatch = 2048

// EmbeddingModel splits requests over the API limit of 2048 inputs through MaxBatch.
type EmbeddingModel struct {
	model *embeddings.Model
}

func embeddingConfig() *embeddings.Config {
	return &embeddings.Config{
		Provider: providerName,
		MaxBatch: maxEmbeddingBatch,
		Request: func(c *ai.EmbedCall, r *embeddings.Request) ([]ai.Warning, error) {
			opts, err := optionsOf(c.Provider)
			if err != nil {
				return nil, err
			}
			if opts.User != "" {
				r.Extra = map[string]any{"user": opts.User}
			}
			return nil, nil
		},
	}
}

func (m *EmbeddingModel) Provider() string { return m.model.Provider() }

func (m *EmbeddingModel) ModelID() string { return m.model.ModelID() }

func (m *EmbeddingModel) MaxBatch() int { return m.model.MaxBatch() }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	return m.model.Embed(ctx, c)
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
