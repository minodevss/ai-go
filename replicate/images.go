package replicate

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
)

// ImageOptions are image settings passed with ai.Providers. Extra is merged into the model's input and wins over the fields mapped from the call.
// Wait is how long the create request is held open, 60s when nil and none when zero. ImageField and MaskField name the model's input fields for the first input file and the mask.
type ImageOptions struct {
	Wait        *time.Duration `json:"wait,omitzero"`
	CancelAfter time.Duration  `json:"cancelAfter,omitzero"`
	ImageField  string         `json:"imageField,omitzero"`
	MaskField   string         `json:"maskField,omitzero"`
	KeepURL     bool           `json:"keepUrl,omitzero"`
	Extra       map[string]any `json:"extra,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

// ImageModel runs a Replicate model that returns image files. The id is owner/name for official models, or owner/name:version, or a version id.
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
	input, warnings, err := imageInput(c, opts)
	if err != nil {
		return nil, err
	}
	pred, err := m.p.create(ctx, m.id, input, createOptions{wait: opts.Wait, cancelAfter: opts.CancelAfter, waitDefault: defaultWait}, c.Headers)
	if err != nil {
		return nil, err
	}
	if !pred.terminal() {
		pred, err = m.p.await(ctx, pred)
		if err != nil {
			return nil, err
		}
	}
	if err := pred.failure(); err != nil {
		return nil, &ai.JobError{Kind: "image", ID: pred.ID, Err: err}
	}
	images, err := m.p.files(ctx, pred, "image/png", mediaio.MaxImageBytes, opts.KeepURL)
	if err != nil {
		return nil, &ai.JobError{Kind: "image", ID: pred.ID, Err: err}
	}
	return &ai.ImageResponse{Images: images, Warnings: warnings}, nil
}

func (p *Provider) await(ctx context.Context, pred *prediction) (*prediction, error) {
	err := p.backoff.Poll(ctx, "image", pred.ID, func(ctx context.Context) (bool, error) {
		next, err := p.get(ctx, pred.ID)
		if err != nil {
			return false, err
		}
		pred = next
		return pred.terminal(), nil
	})
	if err != nil {
		return nil, p.cancelAfterEnd(ctx, pred.ID, err)
	}
	return pred, nil
}

func imageInput(c *ai.ImageCall, opts ImageOptions) (map[string]any, []ai.Warning, error) {
	input := map[string]any{"prompt": c.Prompt}
	var warnings []ai.Warning
	if c.Seed != nil {
		input["seed"] = *c.Seed
	}
	if c.AspectRatio != "" {
		input["aspect_ratio"] = c.AspectRatio
	}
	if c.N > 1 {
		input["num_outputs"] = c.N
	}
	if c.Size != "" {
		w, h, err := parseSize(c.Size)
		if err != nil {
			return nil, nil, err
		}
		input["width"], input["height"] = w, h
	}
	if len(c.Files) > 0 {
		ref, err := mediaio.Input(providerName, c.Files[0])
		if err != nil {
			return nil, nil, err
		}
		input[cmp.Or(opts.ImageField, "image")] = ref
		if len(c.Files) > 1 {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("replicate: only the first of %d input files was sent as %q; pass further inputs in ImageOptions.Extra", len(c.Files), cmp.Or(opts.ImageField, "image"))})
		}
	}
	if c.Mask != nil {
		if len(c.Files) == 0 {
			return nil, nil, errors.New("replicate: an image mask needs input Files to edit")
		}
		ref, err := mediaio.Input(providerName, *c.Mask)
		if err != nil {
			return nil, nil, err
		}
		input[cmp.Or(opts.MaskField, "mask")] = ref
	}
	maps.Copy(input, opts.Extra)
	return input, warnings, nil
}

var _ ai.ImageModel = (*ImageModel)(nil)
