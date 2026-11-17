package luma

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
)

const maxImageRefs = 4

// ImageOptions are image settings passed with ai.Providers. Extra is merged into the request body and wins over the fields mapped from the call.
// FileRole says how input Files guide the result: "image_ref" (default, up to 4), "style_ref" (up to 4), "character_ref" (up to 4 images of one person) or "modify_image_ref" (one image to edit). FileWeight sets their influence.
type ImageOptions struct {
	Format     string         `json:"format,omitzero"`
	FileRole   string         `json:"fileRole,omitzero"`
	FileWeight *float64       `json:"fileWeight,omitzero"`
	KeepURL    bool           `json:"keepUrl,omitzero"`
	Extra      map[string]any `json:"extra,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

// ImageModel generates images with a Photon model. Input Files must be URLs.
type ImageModel struct {
	p  *Provider
	id string
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
	body, warnings, err := m.body(c, opts)
	if err != nil {
		return nil, err
	}
	gen, err := m.p.create(ctx, "/generations/image", body, c.Headers)
	if err != nil {
		return nil, err
	}
	id := gen.ID
	err = m.p.backoff.Poll(ctx, "image", id, func(ctx context.Context) (bool, error) {
		next, err := m.p.get(ctx, id)
		if err != nil {
			return false, err
		}
		gen = next
		return gen.terminal(), nil
	})
	if err != nil {
		return nil, err
	}
	if err := gen.failure(); err != nil {
		return nil, &ai.JobError{Kind: "image", ID: id, Err: err}
	}
	file, err := m.p.file(ctx, gen.Assets.Image, mediaio.ImageType(cmp.Or(opts.Format, "jpg")), mediaio.MaxImageBytes, opts.KeepURL)
	if err != nil {
		return nil, &ai.JobError{Kind: "image", ID: id, Err: err}
	}
	return &ai.ImageResponse{Images: []ai.File{file}, Warnings: warnings}, nil
}

func (m *ImageModel) body(c *ai.ImageCall, opts ImageOptions) (map[string]any, []ai.Warning, error) {
	body := map[string]any{"model": m.id, "prompt": c.Prompt}
	var warnings []ai.Warning
	warn := func(format string, args ...any) {
		warnings = append(warnings, ai.Warning{Message: "luma: " + fmt.Sprintf(format, args...)})
	}
	if c.AspectRatio != "" {
		body["aspect_ratio"] = c.AspectRatio
	}
	if opts.Format != "" {
		body["format"] = opts.Format
	}
	if c.N > 1 {
		warn("n=%d is not supported; each request returns one image", c.N)
	}
	if c.Seed != nil {
		warn("seed is not supported by the Dream Machine API and was ignored")
	}
	if c.Size != "" {
		warn("size is not supported by the Dream Machine API and was ignored; set AspectRatio")
	}
	if c.Mask != nil {
		warn("masks are not supported by the Dream Machine API and were ignored; use FileRole modify_image_ref to edit by prompt")
	}
	if err := fileRefs(body, c.Files, opts); err != nil {
		return nil, nil, err
	}
	maps.Copy(body, opts.Extra)
	return body, warnings, nil
}

type imageRef struct {
	URL    string   `json:"url"`
	Weight *float64 `json:"weight,omitzero"`
}

func fileRefs(body map[string]any, files []ai.File, opts ImageOptions) error {
	if len(files) == 0 {
		return nil
	}
	urls := make([]string, 0, len(files))
	for _, f := range files {
		u, err := inputURL(f)
		if err != nil {
			return err
		}
		urls = append(urls, u)
	}
	role := cmp.Or(opts.FileRole, "image_ref")
	limit := maxImageRefs
	if role == "modify_image_ref" {
		limit = 1
	}
	if len(urls) > limit {
		return fmt.Errorf("luma: %s takes at most %d input files, got %d", role, limit, len(urls))
	}
	switch role {
	case "image_ref", "style_ref":
		refs := make([]imageRef, 0, len(urls))
		for _, u := range urls {
			refs = append(refs, imageRef{URL: u, Weight: opts.FileWeight})
		}
		body[role] = refs
	case "modify_image_ref":
		body[role] = imageRef{URL: urls[0], Weight: opts.FileWeight}
	case "character_ref":
		body[role] = map[string]any{"identity0": map[string]any{"images": urls}}
	default:
		return errors.New("luma: FileRole must be image_ref, style_ref, character_ref or modify_image_ref")
	}
	return nil
}

var _ ai.ImageModel = (*ImageModel)(nil)
