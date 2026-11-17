package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const maxImageInputs = 14

// ImageOptions are image settings passed with ai.Providers under "google", also on Vertex. Unset fields keep the API default.
type ImageOptions struct {
	SafetySettings []SafetySetting `json:"safetySettings,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

// ImageModel generates and edits images with a Nano Banana model. N above 1 sends N sequential requests.
// Input Files are sent as up to 14 reference images; masks are not supported.
type ImageModel struct {
	backend *backend
	id      string
}

type imageRequest struct {
	Contents         []content       `json:"contents"`
	SafetySettings   []SafetySetting `json:"safetySettings,omitzero"`
	GenerationConfig imageGeneration `json:"generationConfig"`
}

type imageGeneration struct {
	ResponseModalities []string     `json:"responseModalities"`
	ImageConfig        *imageConfig `json:"imageConfig,omitzero"`
	Seed               *int         `json:"seed,omitzero"`
}

type imageConfig struct {
	AspectRatio string `json:"aspectRatio,omitzero"`
	ImageSize   string `json:"imageSize,omitzero"`
}

func (p *Provider) ImageModel(id string) *ImageModel {
	return &ImageModel{backend: p.backend, id: id}
}

func (m *ImageModel) Provider() string { return m.backend.name }

func (m *ImageModel) ModelID() string { return m.id }

func (m *ImageModel) GenerateImage(ctx context.Context, c *ai.ImageCall) (*ai.ImageResponse, error) {
	opts, err := decodeOptions[ImageOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	req, warnings, err := m.request(c, opts)
	if err != nil {
		return nil, err
	}
	count := max(c.N, 1)
	out := &ai.ImageResponse{Warnings: warnings}
	for range count {
		var body responseBody
		header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":generateContent", Body: req, Header: c.Headers}, &body)
		if err != nil {
			return nil, fail(err)
		}
		images, err := m.images(body)
		if err != nil {
			return nil, err
		}
		out.Images = append(out.Images, images...)
		out.Header = header
		out.Usage = addUsage(out.Usage, usageOf(body.UsageMetadata))
	}
	return out, nil
}

func (m *ImageModel) request(c *ai.ImageCall, opts ImageOptions) (*imageRequest, []ai.Warning, error) {
	if c.Mask != nil {
		return nil, nil, errors.New("google: image masks are not supported; describe the edit in the prompt")
	}
	if len(c.Files) > maxImageInputs {
		return nil, nil, fmt.Errorf("google: an image request takes at most %d input images, got %d", maxImageInputs, len(c.Files))
	}
	var warnings []ai.Warning
	parts := make([]json.RawMessage, 0, len(c.Files)+1)
	for i, f := range c.Files {
		raw, err := filePart(f)
		if err != nil {
			return nil, nil, fmt.Errorf("google: image input %d: %w", i, err)
		}
		parts = append(parts, raw)
	}
	parts = append(parts, pdata.JSON(providerName, partOut{Text: &c.Prompt}))
	cfg := imageGeneration{ResponseModalities: []string{"TEXT", "IMAGE"}, Seed: c.Seed}
	if c.AspectRatio != "" {
		cfg.ImageConfig = &imageConfig{AspectRatio: c.AspectRatio}
	}
	if c.Size != "" {
		size := strings.ToUpper(c.Size)
		if !validImageSize(size) {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("google: size %q is not one of 512, 1K, 2K, 4K and was ignored", c.Size)})
		} else {
			if cfg.ImageConfig == nil {
				cfg.ImageConfig = &imageConfig{}
			}
			cfg.ImageConfig.ImageSize = size
		}
	}
	return &imageRequest{Contents: []content{{Role: "user", Parts: parts}}, SafetySettings: opts.SafetySettings, GenerationConfig: cfg}, warnings, nil
}

func (m *ImageModel) images(body responseBody) ([]ai.File, error) {
	if err := body.failure(m.backend); err != nil {
		return nil, err
	}
	cand := body.first()
	var images []ai.File
	for _, raw := range cand.Content.Parts {
		var p partProbe
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, fmt.Errorf("google: decode image part: %w", err)
		}
		if p.Thought || p.InlineData == nil || !strings.HasPrefix(p.InlineData.MimeType, "image/") {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return nil, fmt.Errorf("google: decode image data: %w", err)
		}
		images = append(images, ai.File{MediaType: p.InlineData.MimeType, Data: data})
	}
	if len(images) > 0 {
		return images, nil
	}
	return nil, m.noImage(body, cand)
}

func (m *ImageModel) noImage(body responseBody, cand candidate) error {
	reason := cmp.Or(cand.FinishReason, blockReason(body.PromptFeedback))
	if reason == "" {
		return errors.New("google: the response holds no image")
	}
	return fmt.Errorf("google: the response holds no image (%s)", strings.ToLower(reason))
}

func addUsage(a, b ai.Usage) ai.Usage {
	a.Input += b.Input
	a.Output += b.Output
	a.CacheRead += b.CacheRead
	a.Reasoning += b.Reasoning
	if len(b.Raw) > 0 {
		a.Raw = b.Raw
	}
	return a
}

func validImageSize(size string) bool {
	switch size {
	case "512", "1K", "2K", "4K":
		return true
	default:
		return false
	}
}

func decodeOptions[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return o, fmt.Errorf("google: decode provider options: %w", err)
	}
	return o, nil
}

var _ ai.ImageModel = (*ImageModel)(nil)
