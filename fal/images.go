package fal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

// ImageModel runs any fal image endpoint. The model id is the endpoint id, for example "fal-ai/flux/dev" or "fal-ai/flux-pro/kontext". Requests go through the queue unless ImageOptions.Sync is set.
type ImageModel struct {
	p  *Provider
	id string
}

type imageOutput struct {
	Images          []outputFile `json:"images"`
	Image           *outputFile  `json:"image"`
	HasNSFWConcepts []bool       `json:"has_nsfw_concepts"`
}

func (p *Provider) ImageModel(id string) *ImageModel { return &ImageModel{p: p, id: id} }

func NewImageModel(id string) *ImageModel { return New().ImageModel(id) }

func (m *ImageModel) Provider() string { return providerName }

func (m *ImageModel) ModelID() string { return m.id }

func (m *ImageModel) GenerateImage(ctx context.Context, c *ai.ImageCall) (*ai.ImageResponse, error) {
	opts, err := mediaio.DecodeOptions[ImageOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	body, err := m.input(c, opts)
	if err != nil {
		return nil, err
	}
	headers := requestHeaders{Priority: opts.Priority, StartTimeout: opts.StartTimeoutSeconds, NoRetry: opts.NoRetry}.header(c.Headers)
	var raw json.RawMessage
	var respHeader http.Header
	if opts.Sync {
		raw, respHeader, err = m.runSync(ctx, body, headers)
	} else {
		var t ticket
		if t, err = m.p.submit(ctx, m.id, body, headers); err == nil {
			raw, err = m.p.await(ctx, t)
		}
	}
	if err != nil {
		return nil, err
	}
	var out imageOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("fal: decode image result: %w", err)
	}
	images, warnings, err := m.files(ctx, out, opts.KeepURL)
	if err != nil {
		return nil, err
	}
	return &ai.ImageResponse{Images: images, Warnings: warnings, Header: respHeader}, nil
}

func (m *ImageModel) runSync(ctx context.Context, body map[string]any, headers http.Header) (json.RawMessage, http.Header, error) {
	var raw json.RawMessage
	header, err := m.p.sync.JSON(ctx, transport.Request{Path: "/" + escapeModel(m.id), Body: body, Header: headers}, &raw)
	return raw, header, err
}

func (m *ImageModel) input(c *ai.ImageCall, opts ImageOptions) (map[string]any, error) {
	body := map[string]any{"prompt": c.Prompt}
	if c.N > 0 {
		body["num_images"] = c.N
	}
	if c.Size != "" {
		sizeInput(c.Size, body)
	}
	if c.AspectRatio != "" {
		body["aspect_ratio"] = c.AspectRatio
	}
	if c.Seed != nil {
		body["seed"] = *c.Seed
	}
	if opts.OutputFormat != "" {
		body["output_format"] = opts.OutputFormat
	}
	if opts.EnableSafetyChecker != nil {
		body["enable_safety_checker"] = *opts.EnableSafetyChecker
	}
	if opts.SafetyTolerance != "" {
		body["safety_tolerance"] = opts.SafetyTolerance
	}
	files := inputFiles{files: c.Files, mask: c.Mask, imageField: opts.ImageField, imagesField: opts.ImagesField, maskField: opts.MaskField}
	if err := files.apply(body); err != nil {
		return nil, err
	}
	mergeExtra(body, opts.Extra)
	return body, nil
}

func (m *ImageModel) files(ctx context.Context, out imageOutput, keepURL bool) ([]ai.File, []ai.Warning, error) {
	list := out.Images
	if out.Image != nil {
		list = append(list, *out.Image)
	}
	if len(list) == 0 {
		return nil, nil, errors.New("fal: result has no images")
	}
	dl := m.p.downloader()
	files := make([]ai.File, 0, len(list))
	var warnings []ai.Warning
	for i, img := range list {
		if i < len(out.HasNSFWConcepts) && out.HasNSFWConcepts[i] {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("fal: image %d was flagged by the safety checker and may be blank", i)})
		}
		f, err := dl.Result(ctx, img.URL, imageType(img.ContentType), mediaio.MaxImageBytes, keepURL)
		if err != nil {
			return nil, nil, err
		}
		files = append(files, f)
	}
	return files, warnings, nil
}

func imageType(contentType string) string {
	if contentType != "" {
		return contentType
	}
	return mediaio.ImageType("png")
}

var _ ai.ImageModel = (*ImageModel)(nil)
