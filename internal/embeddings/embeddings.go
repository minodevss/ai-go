// Package embeddings implements the OpenAI-style /embeddings endpoint for OpenAI and compatible providers.
package embeddings

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/transport"
)

// Config describes one provider's embeddings variant. MaxBatch is the provider's limit of inputs per request.
type Config struct {
	Provider        string
	MaxBatch        int
	DimensionsField string
	NoEncoding      bool
	InputType       bool
	Request         func(c *ai.EmbedCall, r *Request) ([]ai.Warning, error)
}

// Request is the embeddings request body; Extra holds provider-specific fields.
type Request struct {
	Model          string
	Input          []string
	Dimensions     int
	EncodingFormat string
	Extra          map[string]any
}

type Model struct {
	client *transport.Client
	id     string
	cfg    *Config
}

type data struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type response struct {
	Data  []data          `json:"data"`
	Usage json.RawMessage `json:"usage"`
}

func New(client *transport.Client, id string, cfg *Config) *Model {
	return &Model{client: client, id: id, cfg: cfg}
}

func (m *Model) Provider() string { return m.cfg.Provider }

func (m *Model) ModelID() string { return m.id }

func (m *Model) MaxBatch() int { return m.cfg.MaxBatch }

func (m *Model) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	if m.cfg.MaxBatch > 0 && len(c.Inputs) > m.cfg.MaxBatch {
		return nil, fmt.Errorf("%s: embed: %d inputs exceed the limit of %d per request", m.cfg.Provider, len(c.Inputs), m.cfg.MaxBatch)
	}
	texts := make([]string, len(c.Inputs))
	for i, in := range c.Inputs {
		text, err := inputText(in)
		if err != nil {
			return nil, fmt.Errorf("%s: embed input %d: %w", m.cfg.Provider, i, err)
		}
		texts[i] = text
	}
	req := &Request{Model: m.id, Input: texts, Dimensions: c.Dimensions}
	if !m.cfg.NoEncoding {
		req.EncodingFormat = "float"
	}
	var warnings []ai.Warning
	if c.InputType != "" && !m.cfg.InputType {
		warnings = append(warnings, ai.Warning{Message: m.cfg.Provider + ": embedding input type is not supported and was ignored"})
	}
	if m.cfg.Request != nil {
		w, err := m.cfg.Request(c, req)
		if err != nil {
			return nil, err
		}
		warnings = append(warnings, w...)
	}
	var body response
	header, err := m.client.JSON(ctx, transport.Request{Path: "/embeddings", Body: req.body(m.cfg), Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	if len(body.Data) != len(texts) {
		return nil, fmt.Errorf("%s: embed: got %d embeddings for %d inputs", m.cfg.Provider, len(body.Data), len(texts))
	}
	slices.SortStableFunc(body.Data, func(a, b data) int { return a.Index - b.Index })
	out := make([][]float32, len(body.Data))
	for i, d := range body.Data {
		out[i] = d.Embedding
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: openaiwire.Usage(body.Usage), Warnings: warnings, Header: header}, nil
}

func (r *Request) body(cfg *Config) map[string]any {
	out := map[string]any{"model": r.Model, "input": r.Input}
	if r.Dimensions != 0 {
		out[cmp.Or(cfg.DimensionsField, "dimensions")] = r.Dimensions
	}
	if r.EncodingFormat != "" {
		out["encoding_format"] = r.EncodingFormat
	}
	maps.Copy(out, r.Extra)
	return out
}

func inputText(parts ai.Parts) (string, error) {
	var text strings.Builder
	for _, p := range parts {
		t, ok := p.(ai.Text)
		if !ok {
			return "", errors.New("only text inputs are supported")
		}
		text.WriteString(t.Text)
	}
	return text.String(), nil
}

var _ ai.EmbeddingModel = (*Model)(nil)
