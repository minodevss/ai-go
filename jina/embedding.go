package jina

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/embeddings"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	taskQuery    = "retrieval.query"
	taskPassage  = "retrieval.passage"
	pdfMediaType = "application/pdf"
)

// EmbeddingModel embeds text, and images, audio, video and PDFs for multimodal models. The API has no limit on inputs per request, so MaxBatch is 0 and ai.Embed sends one request.
type EmbeddingModel struct {
	client *transport.Client
	id     string
}

type embeddingRequest struct {
	Model        string `json:"model"`
	Input        []any  `json:"input"`
	Task         string `json:"task,omitzero"`
	Dimensions   int    `json:"dimensions,omitzero"`
	Normalized   *bool  `json:"normalized,omitzero"`
	Truncate     *bool  `json:"truncate,omitzero"`
	LateChunking *bool  `json:"late_chunking,omitzero"`
}

type mediaDoc struct {
	Image string `json:"image,omitzero"`
	Audio string `json:"audio,omitzero"`
	Video string `json:"video,omitzero"`
	PDF   string `json:"pdf,omitzero"`
}

type textDoc struct {
	Text string `json:"text"`
}

type contentGroup struct {
	Content []any `json:"content"`
}

type embeddingData struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type embeddingResponse struct {
	Data  []embeddingData `json:"data"`
	Usage json.RawMessage `json:"usage"`
}

func (m *EmbeddingModel) Provider() string { return providerName }

func (m *EmbeddingModel) ModelID() string { return m.id }

func (m *EmbeddingModel) MaxBatch() int { return 0 }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	opts, err := decodeOptions[EmbeddingOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	inputs, err := inputsOf(c.Inputs)
	if err != nil {
		return nil, err
	}
	task, warnings, err := m.taskOf(c.InputType, opts.Task)
	if err != nil {
		return nil, err
	}
	body := embeddingRequest{
		Model:        m.id,
		Input:        inputs,
		Task:         task,
		Dimensions:   c.Dimensions,
		Normalized:   opts.Normalized,
		Truncate:     opts.Truncate,
		LateChunking: opts.LateChunking,
	}
	var resp embeddingResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: "/v1/embeddings", Body: body, Header: c.Headers}, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(c.Inputs) {
		return nil, fmt.Errorf("jina: embed: got %d embeddings for %d inputs", len(resp.Data), len(c.Inputs))
	}
	slices.SortStableFunc(resp.Data, func(a, b embeddingData) int { return a.Index - b.Index })
	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: embeddings.TotalTokensUsage(resp.Usage), Warnings: warnings, Header: header}, nil
}

func taskModelPrefixes() []string {
	return []string{"jina-embeddings-v3", "jina-embeddings-v4", "jina-embeddings-v5"}
}

func (m *EmbeddingModel) taskOf(t ai.InputType, override string) (string, []ai.Warning, error) {
	if override != "" {
		return override, nil, nil
	}
	if t == "" {
		return "", nil, nil
	}
	if t != ai.InputQuery && t != ai.InputDocument {
		return "", nil, fmt.Errorf("jina: embed: unknown input type %q", t)
	}
	if !slices.ContainsFunc(taskModelPrefixes(), func(p string) bool { return strings.HasPrefix(m.id, p) }) {
		return "", []ai.Warning{{Message: "jina: input type is not mapped to a task for " + m.id + " and was ignored; set EmbeddingOptions.Task"}}, nil
	}
	if t == ai.InputQuery {
		return taskQuery, nil, nil
	}
	return taskPassage, nil, nil
}

func inputsOf(inputs []ai.Parts) ([]any, error) {
	out := make([]any, len(inputs))
	hasPDF := false
	for i, in := range inputs {
		item, isPDF, err := inputOf(in)
		if err != nil {
			return nil, fmt.Errorf("%w (input %d)", err, i)
		}
		hasPDF = hasPDF || isPDF
		out[i] = item
	}
	if hasPDF && len(inputs) > 1 {
		return nil, errors.New("jina: embed: a PDF input must be sent on its own")
	}
	return out, nil
}

func inputOf(in ai.Parts) (item any, isPDF bool, err error) {
	if len(in) == 0 {
		return nil, false, errors.New("jina: embed: input has no parts")
	}
	if text, ok := onlyText(in); ok {
		return text, false, nil
	}
	if len(in) == 1 {
		doc, err := docOf(in[0])
		return doc, isPDFPart(in[0]), err
	}
	group := contentGroup{Content: make([]any, len(in))}
	for i, p := range in {
		if isPDFPart(p) {
			return nil, false, errors.New("jina: embed: a PDF cannot be part of a multi-part input")
		}
		doc, err := docOf(p)
		if err != nil {
			return nil, false, err
		}
		group.Content[i] = doc
	}
	return group, false, nil
}

func onlyText(in ai.Parts) (string, bool) {
	var text strings.Builder
	for _, p := range in {
		t, ok := p.(ai.Text)
		if !ok {
			return "", false
		}
		text.WriteString(t.Text)
	}
	return text.String(), true
}

func isPDFPart(p ai.Part) bool {
	f, ok := p.(ai.File)
	return ok && f.MediaType == pdfMediaType
}

func docOf(p ai.Part) (any, error) {
	switch v := p.(type) {
	case ai.Text:
		return textDoc{Text: v.Text}, nil
	case ai.File:
		return fileDoc(v)
	default:
		return nil, fmt.Errorf("jina: embed: unsupported part %T", p)
	}
}

func fileDoc(f ai.File) (mediaDoc, error) {
	value := f.URL
	if value == "" {
		if len(f.Data) == 0 {
			return mediaDoc{}, errors.New("jina: embed: file has neither URL nor data")
		}
		value = base64.StdEncoding.EncodeToString(f.Data)
	}
	kind, _, _ := strings.Cut(f.MediaType, "/")
	switch {
	case f.MediaType == pdfMediaType:
		return mediaDoc{PDF: value}, nil
	case kind == "image":
		return mediaDoc{Image: value}, nil
	case kind == "audio":
		return mediaDoc{Audio: value}, nil
	case kind == "video":
		return mediaDoc{Video: value}, nil
	default:
		return mediaDoc{}, fmt.Errorf("jina: embed: media type %q is not supported, use an image, audio, video or %s", f.MediaType, pdfMediaType)
	}
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
