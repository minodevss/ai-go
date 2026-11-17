package ollama

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const embedPath = "/api/embed"

// EmbeddingModel embeds text, and images or audio on multimodal embedding models. The API documents no batch limit, so MaxBatch is 0.
type EmbeddingModel struct {
	client *transport.Client
	id     string
}

type embedRequest struct {
	Model      string         `json:"model"`
	Input      []any          `json:"input"`
	Truncate   *bool          `json:"truncate,omitzero"`
	Options    map[string]any `json:"options,omitzero"`
	KeepAlive  string         `json:"keep_alive,omitzero"`
	Dimensions int            `json:"dimensions,omitzero"`
}

type embedInput struct {
	Text  string `json:"text,omitzero"`
	Image string `json:"image,omitzero"`
	Audio string `json:"audio,omitzero"`
}

type embedBody struct {
	Embeddings      [][]float32 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

func (m *EmbeddingModel) Provider() string { return providerName }

func (m *EmbeddingModel) ModelID() string { return m.id }

func (m *EmbeddingModel) MaxBatch() int { return 0 }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	opts, err := optionsOf[EmbeddingOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	req := &embedRequest{Model: m.id, Truncate: opts.Truncate, KeepAlive: opts.KeepAlive, Dimensions: c.Dimensions}
	for i, in := range c.Inputs {
		v, err := embedInputOf(in)
		if err != nil {
			return nil, fmt.Errorf("%w (input %d)", err, i)
		}
		req.Input = append(req.Input, v)
	}
	if o := embedOptions(opts); len(o) > 0 {
		req.Options = o
	}
	var warnings []ai.Warning
	if c.InputType != "" {
		warnings = append(warnings, ai.Warning{Message: "ollama: embedding input type is not supported and was ignored"})
	}
	var body embedBody
	header, err := m.client.JSON(ctx, transport.Request{Path: embedPath, Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	if len(body.Embeddings) != len(c.Inputs) {
		return nil, fmt.Errorf("ollama: embed: got %d embeddings for %d inputs", len(body.Embeddings), len(c.Inputs))
	}
	return &ai.EmbedResponse{
		Embeddings: body.Embeddings,
		Usage:      ai.Usage{Input: body.PromptEvalCount},
		Warnings:   warnings,
		Header:     header,
	}, nil
}

func embedOptions(o EmbeddingOptions) map[string]any {
	out := map[string]any{}
	maps.Copy(out, o.Parameters)
	if o.NumCtx != 0 {
		out["num_ctx"] = o.NumCtx
	}
	return out
}

func embedInputOf(parts ai.Parts) (any, error) {
	var in embedInput
	var text strings.Builder
	media := false
	for _, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			text.WriteString(v.Text)
		case ai.File:
			if len(v.Data) == 0 {
				return nil, errors.New("ollama: embed: file has no data: URLs are not supported")
			}
			data := base64.StdEncoding.EncodeToString(v.Data)
			switch {
			case strings.HasPrefix(v.MediaType, "image/") && in.Image == "":
				in.Image = data
			case strings.HasPrefix(v.MediaType, "audio/") && in.Audio == "":
				in.Audio = data
			default:
				return nil, fmt.Errorf("ollama: embed: unsupported or repeated file type %q: an input takes at most one image and one audio file", v.MediaType)
			}
			media = true
		default:
			return nil, fmt.Errorf("ollama: embed: unsupported part %T", p)
		}
	}
	if !media {
		return text.String(), nil
	}
	in.Text = text.String()
	return in, nil
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
