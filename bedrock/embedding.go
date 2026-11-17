package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	cohereMaxBatch  = 96
	titanMaxBatch   = 1
	cohereDocument  = "search_document"
	cohereQuery     = "search_query"
	cohereImageType = "image"
)

type embeddingFamily int

const (
	familyUnknown embeddingFamily = iota
	familyTitanText
	familyTitanTextV2
	familyTitanImage
	familyCohereV3
	familyCohereV4
)

func embeddingFamilyOf(id string) embeddingFamily {
	id = strings.ToLower(id)
	switch {
	case strings.Contains(id, "titan-embed-text-v2"):
		return familyTitanTextV2
	case strings.Contains(id, "titan-embed-text"):
		return familyTitanText
	case strings.Contains(id, "titan-embed-image"):
		return familyTitanImage
	case strings.Contains(id, "cohere.embed-v4"):
		return familyCohereV4
	case strings.Contains(id, "cohere.embed"):
		return familyCohereV3
	default:
		return familyUnknown
	}
}

// EmbeddingModel calls InvokeModel for Amazon Titan text and multimodal embeddings and for Cohere Embed v3 and v4. Titan models embed one input per call, so MaxBatch is 1.
type EmbeddingModel struct {
	client *transport.Client
	id     string
	family embeddingFamily
}

type titanRequest struct {
	InputText       string           `json:"inputText,omitzero"`
	InputImage      string           `json:"inputImage,omitzero"`
	Dimensions      int              `json:"dimensions,omitzero"`
	Normalize       *bool            `json:"normalize,omitzero"`
	EmbeddingConfig *titanImageShape `json:"embeddingConfig,omitzero"`
}

type titanImageShape struct {
	OutputEmbeddingLength int `json:"outputEmbeddingLength"`
}

type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
	Message             string    `json:"message"`
	ByType              struct {
		Float []float32 `json:"float"`
	} `json:"embeddingsByType"`
}

type cohereRequest struct {
	InputType       string        `json:"input_type"`
	Texts           []string      `json:"texts,omitzero"`
	Images          []string      `json:"images,omitzero"`
	Inputs          []cohereInput `json:"inputs,omitzero"`
	OutputDimension int           `json:"output_dimension,omitzero"`
	Truncate        string        `json:"truncate,omitzero"`
}

type cohereInput struct {
	Content []cohereContent `json:"content"`
}

type cohereContent struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitzero"`
	ImageURL *cohereImageURL `json:"image_url,omitzero"`
}

type cohereImageURL struct {
	URL string `json:"url"`
}

type cohereResponse struct {
	Embeddings json.RawMessage `json:"embeddings"`
}

func (m *EmbeddingModel) Provider() string { return providerName }

func (m *EmbeddingModel) ModelID() string { return m.id }

func (m *EmbeddingModel) MaxBatch() int {
	switch m.family {
	case familyCohereV3, familyCohereV4:
		return cohereMaxBatch
	case familyTitanText, familyTitanTextV2, familyTitanImage, familyUnknown:
		return titanMaxBatch
	default:
		return titanMaxBatch
	}
}

func (m *EmbeddingModel) Embed(ctx context.Context, c *ai.EmbedCall) (*ai.EmbedResponse, error) {
	opts, err := embeddingOptionsOf(c.Provider)
	if err != nil {
		return nil, err
	}
	header := http.Header{}
	for k, v := range c.Headers {
		header[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	switch m.family {
	case familyTitanText, familyTitanTextV2, familyTitanImage:
		return m.embedTitan(ctx, c, opts, header)
	case familyCohereV3, familyCohereV4:
		return m.embedCohere(ctx, c, opts, header)
	case familyUnknown:
		return nil, fmt.Errorf("bedrock: unsupported embedding model %q: supported are Amazon Titan text and multimodal embeddings and Cohere Embed v3 and v4", m.id)
	default:
		return nil, fmt.Errorf("bedrock: unsupported embedding model %q", m.id)
	}
}

func (m *EmbeddingModel) invoke(ctx context.Context, header http.Header, body, out any) (http.Header, error) {
	return m.client.JSON(ctx, transport.Request{Path: modelPath(m.id, "invoke"), Body: body, Header: header}, out)
}

func (m *EmbeddingModel) embedTitan(ctx context.Context, c *ai.EmbedCall, opts EmbeddingOptions, header http.Header) (*ai.EmbedResponse, error) {
	res := &ai.EmbedResponse{}
	for _, in := range c.Inputs {
		req, warnings, err := m.titanRequest(in, c.Dimensions, opts)
		if err != nil {
			return nil, err
		}
		var body titanResponse
		h, err := m.invoke(ctx, header, req, &body)
		if err != nil {
			return nil, err
		}
		vec := body.Embedding
		if len(vec) == 0 {
			vec = body.ByType.Float
		}
		if len(vec) == 0 {
			return nil, fmt.Errorf("bedrock: embedding response has no vector%s", messageSuffix(body.Message))
		}
		res.Embeddings = append(res.Embeddings, vec)
		res.Usage.Input += body.InputTextTokenCount
		res.Warnings = append(res.Warnings, warnings...)
		res.Header = h
	}
	return res, nil
}

func messageSuffix(message string) string {
	if message == "" {
		return ""
	}
	return ": " + message
}

func (m *EmbeddingModel) titanRequest(in ai.Parts, dims int, opts EmbeddingOptions) (titanRequest, []ai.Warning, error) {
	text, files, err := splitParts(in)
	if err != nil {
		return titanRequest{}, nil, err
	}
	var req titanRequest
	var warnings []ai.Warning
	req.InputText = text
	switch m.family {
	case familyTitanImage:
		if len(files) > 1 {
			return titanRequest{}, nil, errors.New("bedrock: Titan multimodal embeddings take one image per input")
		}
		if len(files) == 1 {
			req.InputImage = base64.StdEncoding.EncodeToString(files[0].Data)
		}
		if dims != 0 {
			req.EmbeddingConfig = &titanImageShape{OutputEmbeddingLength: dims}
		}
	case familyTitanTextV2:
		req.Dimensions, req.Normalize = dims, opts.Normalize
		if len(files) > 0 {
			return titanRequest{}, nil, errors.New("bedrock: Titan text embeddings take text only")
		}
	default:
		if len(files) > 0 {
			return titanRequest{}, nil, errors.New("bedrock: Titan text embeddings take text only")
		}
		if dims != 0 {
			warnings = append(warnings, ai.Warning{Message: "bedrock: dimensions is not supported by Titan Embeddings G1 and was ignored"})
		}
	}
	if req.InputText == "" && req.InputImage == "" {
		return titanRequest{}, nil, errors.New("bedrock: embedding input is empty")
	}
	return req, warnings, nil
}

func splitParts(in ai.Parts) (string, []ai.File, error) {
	var text strings.Builder
	var files []ai.File
	for _, p := range in {
		switch v := p.(type) {
		case ai.Text:
			text.WriteString(v.Text)
		case ai.File:
			if len(v.Data) == 0 {
				return "", nil, errors.New("bedrock: embedding file has no data")
			}
			files = append(files, v)
		default:
			return "", nil, fmt.Errorf("bedrock: embedding inputs take text and files, not %T", p)
		}
	}
	return text.String(), files, nil
}

func (m *EmbeddingModel) embedCohere(ctx context.Context, c *ai.EmbedCall, opts EmbeddingOptions, header http.Header) (*ai.EmbedResponse, error) {
	req, warnings, err := m.cohereRequest(c, opts)
	if err != nil {
		return nil, err
	}
	var body cohereResponse
	h, err := m.invoke(ctx, header, req, &body)
	if err != nil {
		return nil, err
	}
	vectors, err := cohereVectors(body.Embeddings)
	if err != nil {
		return nil, err
	}
	if len(vectors) != len(c.Inputs) {
		return nil, fmt.Errorf("bedrock: embedding response has %d vectors for %d inputs", len(vectors), len(c.Inputs))
	}
	return &ai.EmbedResponse{Embeddings: vectors, Warnings: warnings, Header: h}, nil
}

func (m *EmbeddingModel) cohereRequest(c *ai.EmbedCall, opts EmbeddingOptions) (cohereRequest, []ai.Warning, error) {
	req := cohereRequest{InputType: cohereDocument, Truncate: opts.Truncate}
	if c.InputType == ai.InputQuery {
		req.InputType = cohereQuery
	}
	var warnings []ai.Warning
	switch {
	case c.Dimensions == 0:
	case m.family == familyCohereV4:
		req.OutputDimension = c.Dimensions
	default:
		warnings = append(warnings, ai.Warning{Message: "bedrock: dimensions is not supported by Cohere Embed v3 and was ignored"})
	}
	texts := make([]string, 0, len(c.Inputs))
	multimodal := false
	for _, in := range c.Inputs {
		text, files, err := splitParts(in)
		if err != nil {
			return cohereRequest{}, nil, err
		}
		if len(files) > 0 {
			multimodal = true
		}
		texts = append(texts, text)
	}
	if !multimodal {
		req.Texts = texts
		return req, warnings, nil
	}
	if m.family == familyCohereV3 {
		return m.cohereV3Image(req, c, warnings)
	}
	for _, in := range c.Inputs {
		content, err := cohereContents(in)
		if err != nil {
			return cohereRequest{}, nil, err
		}
		req.Inputs = append(req.Inputs, cohereInput{Content: content})
	}
	return req, warnings, nil
}

func (m *EmbeddingModel) cohereV3Image(req cohereRequest, c *ai.EmbedCall, warnings []ai.Warning) (cohereRequest, []ai.Warning, error) {
	if len(c.Inputs) != 1 {
		return cohereRequest{}, nil, errors.New("bedrock: Cohere Embed v3 takes one image per call and no mix of texts and images")
	}
	text, files, err := splitParts(c.Inputs[0])
	if err != nil {
		return cohereRequest{}, nil, err
	}
	if text != "" || len(files) != 1 {
		return cohereRequest{}, nil, errors.New("bedrock: Cohere Embed v3 takes one image per call and no mix of texts and images")
	}
	uri, err := dataURI(files[0])
	if err != nil {
		return cohereRequest{}, nil, err
	}
	req.InputType = cohereImageType
	req.Images = []string{uri}
	return req, warnings, nil
}

func cohereContents(in ai.Parts) ([]cohereContent, error) {
	var out []cohereContent
	for _, p := range in {
		switch v := p.(type) {
		case ai.Text:
			out = append(out, cohereContent{Type: "text", Text: v.Text})
		case ai.File:
			uri, err := dataURI(v)
			if err != nil {
				return nil, err
			}
			out = append(out, cohereContent{Type: "image_url", ImageURL: &cohereImageURL{URL: uri}})
		default:
			return nil, fmt.Errorf("bedrock: embedding inputs take text and files, not %T", p)
		}
	}
	return out, nil
}

func dataURI(f ai.File) (string, error) {
	if _, ok := imageFormat(f.MediaType); !ok {
		return "", fmt.Errorf("bedrock: embedding file media type %q is not an image", f.MediaType)
	}
	return "data:" + f.MediaType + ";base64," + base64.StdEncoding.EncodeToString(f.Data), nil
}

func cohereVectors(raw json.RawMessage) ([][]float32, error) {
	var plain [][]float32
	if err := json.Unmarshal(raw, &plain); err == nil {
		return plain, nil
	}
	var byType struct {
		Float [][]float32 `json:"float"`
	}
	if err := json.Unmarshal(raw, &byType); err != nil || len(byType.Float) == 0 {
		return nil, errors.New("bedrock: embedding response has no float vectors")
	}
	return byType.Float, nil
}

var _ ai.EmbeddingModel = (*EmbeddingModel)(nil)
