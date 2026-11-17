package voyage

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
	maxEmbeddingBatch  = 1000
	multimodalPrefix   = "voyage-multimodal"
	textPath           = "/v1/embeddings"
	multimodalPath     = "/v1/multimodalembeddings"
	contentText        = "text"
	contentImageURL    = "image_url"
	contentImageBase64 = "image_base64"
	contentVideoURL    = "video_url"
	contentVideoBase64 = "video_base64"
	videoMediaType     = "video/mp4"
	imageMediaPrefix   = "image/"
)

// EmbeddingModel embeds text, or text with images and video for voyage-multimodal models. Its MaxBatch is the API limit of 1000 inputs per request.
type EmbeddingModel struct {
	client *transport.Client
	id     string
}

type textRequest struct {
	Input           []string `json:"input"`
	Model           string   `json:"model"`
	InputType       string   `json:"input_type,omitzero"`
	Truncation      *bool    `json:"truncation,omitzero"`
	OutputDimension int      `json:"output_dimension,omitzero"`
}

type multimodalRequest struct {
	Inputs     []multimodalInput `json:"inputs"`
	Model      string            `json:"model"`
	InputType  string            `json:"input_type,omitzero"`
	Truncation *bool             `json:"truncation,omitzero"`
}

type multimodalInput struct {
	Content []contentPiece `json:"content"`
}

type contentPiece struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitzero"`
	ImageURL    string `json:"image_url,omitzero"`
	ImageBase64 string `json:"image_base64,omitzero"`
	VideoURL    string `json:"video_url,omitzero"`
	VideoBase64 string `json:"video_base64,omitzero"`
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

func (m *EmbeddingModel) MaxBatch() int { return maxEmbeddingBatch }

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	if len(c.Inputs) > maxEmbeddingBatch {
		return nil, fmt.Errorf("voyage: embed: %d inputs exceed the limit of %d per request", len(c.Inputs), maxEmbeddingBatch)
	}
	opts, err := optionsOf[EmbeddingOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	inputType, err := inputTypeOf(c.InputType)
	if err != nil {
		return nil, err
	}
	var warnings []ai.Warning
	var body any
	path := textPath
	if strings.HasPrefix(m.id, multimodalPrefix) {
		path = multimodalPath
		inputs, err := multimodalInputs(c.Inputs)
		if err != nil {
			return nil, err
		}
		if c.Dimensions != 0 {
			warnings = append(warnings, ai.Warning{Message: "voyage: dimensions is not supported by the multimodal embeddings endpoint and was ignored"})
		}
		body = multimodalRequest{Inputs: inputs, Model: m.id, InputType: inputType, Truncation: opts.Truncation}
	} else {
		texts, err := textInputs(c.Inputs)
		if err != nil {
			return nil, err
		}
		body = textRequest{Input: texts, Model: m.id, InputType: inputType, Truncation: opts.Truncation, OutputDimension: c.Dimensions}
	}
	var resp embeddingResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: path, Body: body, Header: c.Headers}, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(c.Inputs) {
		return nil, fmt.Errorf("voyage: embed: got %d embeddings for %d inputs", len(resp.Data), len(c.Inputs))
	}
	slices.SortStableFunc(resp.Data, func(a, b embeddingData) int { return a.Index - b.Index })
	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return &ai.EmbedResponse{Embeddings: out, Usage: embeddings.TotalTokensUsage(resp.Usage), Warnings: warnings, Header: header}, nil
}

func inputTypeOf(t ai.InputType) (string, error) {
	switch t {
	case "":
		return "", nil
	case ai.InputQuery:
		return "query", nil
	case ai.InputDocument:
		return "document", nil
	default:
		return "", fmt.Errorf("voyage: embed: unknown input type %q", t)
	}
}

func textInputs(inputs []ai.Parts) ([]string, error) {
	out := make([]string, len(inputs))
	for i, in := range inputs {
		var text strings.Builder
		for _, p := range in {
			t, ok := p.(ai.Text)
			if !ok {
				return nil, fmt.Errorf("voyage: embed input %d: only text inputs are supported by text embedding models, use a %s model for images and video", i, multimodalPrefix)
			}
			text.WriteString(t.Text)
		}
		out[i] = text.String()
	}
	return out, nil
}

func multimodalInputs(inputs []ai.Parts) ([]multimodalInput, error) {
	out := make([]multimodalInput, len(inputs))
	var inline, linked bool
	for i, in := range inputs {
		for _, p := range in {
			piece, isInline, err := contentOf(p)
			if err != nil {
				return nil, fmt.Errorf("%w (input %d)", err, i)
			}
			switch {
			case piece.Type == contentText:
			case isInline:
				inline = true
			default:
				linked = true
			}
			out[i].Content = append(out[i].Content, piece)
		}
	}
	if inline && linked {
		return nil, errors.New("voyage: embed: one request cannot mix media URLs and inline media data")
	}
	return out, nil
}

func contentOf(p ai.Part) (piece contentPiece, inline bool, err error) {
	switch v := p.(type) {
	case ai.Text:
		return contentPiece{Type: contentText, Text: v.Text}, false, nil
	case ai.File:
		return fileContent(v)
	default:
		return contentPiece{}, false, fmt.Errorf("voyage: embed: unsupported part %T", p)
	}
}

func fileContent(f ai.File) (contentPiece, bool, error) {
	isVideo := f.MediaType == videoMediaType
	if !isVideo && !strings.HasPrefix(f.MediaType, imageMediaPrefix) {
		return contentPiece{}, false, fmt.Errorf("voyage: embed: media type %q is not supported, use an image or %s", f.MediaType, videoMediaType)
	}
	switch {
	case f.URL != "" && isVideo:
		return contentPiece{Type: contentVideoURL, VideoURL: f.URL}, false, nil
	case f.URL != "":
		return contentPiece{Type: contentImageURL, ImageURL: f.URL}, false, nil
	case len(f.Data) == 0:
		return contentPiece{}, false, errors.New("voyage: embed: file has neither URL nor data")
	}
	dataURL := "data:" + f.MediaType + ";base64," + base64.StdEncoding.EncodeToString(f.Data)
	if isVideo {
		return contentPiece{Type: contentVideoBase64, VideoBase64: dataURL}, true, nil
	}
	return contentPiece{Type: contentImageBase64, ImageBase64: dataURL}, true, nil
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
