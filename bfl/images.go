package bfl

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const resultExpiryWarning = "bfl: result urls expire after 10 minutes; download the image before then"

// ImageOptions are image settings passed with ai.Providers. Unset fields keep the API default. Extra is merged into the request body and wins over the fields mapped from the call.
type ImageOptions struct {
	OutputFormat     string         `json:"outputFormat,omitzero"`
	SafetyTolerance  *int           `json:"safetyTolerance,omitzero"`
	PromptUpsampling *bool          `json:"promptUpsampling,omitzero"`
	KeepURL          bool           `json:"keepUrl,omitzero"`
	Extra            map[string]any `json:"extra,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

// ImageModel generates and edits images with a FLUX model: text to image, or editing when the call has input Files (Kontext, FLUX.2, FLUX.3 Image and Fill with a Mask).
// The request body follows the model family of the id, because the API names the size, aspect ratio and reference fields per model.
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
	var sub submitResponse
	header, err := m.p.client.JSON(ctx, transport.Request{Path: "/v1/" + m.id, Body: body, Header: c.Headers}, &sub)
	if err != nil {
		return nil, mediaio.FixError(err)
	}
	if sub.ID == "" {
		return nil, errors.New("bfl: submit response has no id")
	}
	target := m.p.pollTarget(sub)
	if err := m.p.trustedPollingURL(target); err != nil {
		return nil, &ai.JobError{Kind: "image", ID: sub.ID, Err: err}
	}
	var final pollResponse
	err = m.p.backoff.Poll(ctx, "image", sub.ID, func(ctx context.Context) (bool, error) {
		var next pollResponse
		if err := m.p.getURL(ctx, target, &next); err != nil {
			return false, err
		}
		if next.pending() {
			return false, nil
		}
		final = next
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	if err := final.failure(); err != nil {
		return nil, &ai.JobError{Kind: "image", ID: sub.ID, Err: err}
	}
	sample := final.sample()
	if sample == "" {
		return nil, &ai.JobError{Kind: "image", ID: sub.ID, Err: errors.New("bfl: ready result has no sample url")}
	}
	file, err := m.p.dl.Result(ctx, sample, mediaio.ImageType(cmp.Or(opts.OutputFormat, "jpeg")), mediaio.MaxImageBytes, opts.KeepURL)
	if err != nil {
		return nil, &ai.JobError{Kind: "image", ID: sub.ID, Err: err}
	}
	if opts.KeepURL {
		warnings = append(warnings, ai.Warning{Message: resultExpiryWarning})
	}
	return &ai.ImageResponse{Images: []ai.File{file}, Warnings: warnings, Header: header}, nil
}

type family struct {
	size     bool
	aspect   bool
	seed     bool
	maxFiles int
	files    func(values []string) map[string]any
	mask     bool
}

func inputImages(values []string) map[string]any {
	out := make(map[string]any, len(values))
	for i, v := range values {
		key := "input_image"
		if i > 0 {
			key += "_" + strconv.Itoa(i+1)
		}
		out[key] = v
	}
	return out
}

func familyOf(id string) family {
	switch {
	case strings.HasPrefix(id, "flux-3-image"):
		return family{size: true, aspect: true, maxFiles: 10, files: func(v []string) map[string]any { return map[string]any{"images": v} }}
	case strings.HasPrefix(id, "flux-kontext"):
		return family{aspect: true, seed: true, maxFiles: 4, files: inputImages}
	case strings.HasPrefix(id, "flux-2-"):
		return family{size: true, seed: true, maxFiles: 8, files: inputImages}
	case id == "flux-pro-1.0-fill":
		return family{seed: true, maxFiles: 1, mask: true, files: func(v []string) map[string]any { return map[string]any{"image": v[0]} }}
	case id == "flux-pro-1.1-ultra":
		return family{aspect: true, seed: true, maxFiles: 1, files: func(v []string) map[string]any { return map[string]any{"image_prompt": v[0]} }}
	default:
		return family{size: true, seed: true, maxFiles: 1, files: func(v []string) map[string]any { return map[string]any{"image_prompt": v[0]} }}
	}
}

func (m *ImageModel) body(c *ai.ImageCall, opts ImageOptions) (map[string]any, []ai.Warning, error) {
	fam := familyOf(m.id)
	body := map[string]any{"prompt": c.Prompt}
	var warnings []ai.Warning
	warn := func(format string, args ...any) {
		warnings = append(warnings, ai.Warning{Message: "bfl: " + fmt.Sprintf(format, args...)})
	}
	if c.N > 1 {
		warn("n=%d is not supported; each request returns one image", c.N)
	}
	switch {
	case c.Seed != nil && fam.seed:
		body["seed"] = *c.Seed
	case c.Seed != nil:
		warn("seed is not supported by %s and was ignored", m.id)
	}
	if c.AspectRatio != "" {
		if !fam.aspect {
			warn("aspectRatio is not supported by %s and was ignored; set Size to WIDTHxHEIGHT", m.id)
		} else {
			body["aspect_ratio"] = c.AspectRatio
		}
	}
	if c.Size != "" {
		w, h, ok := strings.Cut(strings.ToLower(c.Size), "x")
		width, wErr := strconv.Atoi(w)
		height, hErr := strconv.Atoi(h)
		switch {
		case !ok || wErr != nil || hErr != nil || width <= 0 || height <= 0:
			return nil, nil, fmt.Errorf("bfl: size %q must be WIDTHxHEIGHT", c.Size)
		case !fam.size:
			warn("size is not supported by %s and was ignored; set AspectRatio", m.id)
		default:
			body["width"], body["height"] = width, height
		}
	}
	if err := m.addFiles(body, c, fam, &warnings); err != nil {
		return nil, nil, err
	}
	if opts.OutputFormat != "" {
		body["output_format"] = opts.OutputFormat
	}
	if opts.SafetyTolerance != nil {
		body["safety_tolerance"] = *opts.SafetyTolerance
	}
	if opts.PromptUpsampling != nil {
		body["prompt_upsampling"] = *opts.PromptUpsampling
	}
	maps.Copy(body, opts.Extra)
	return body, warnings, nil
}

func (m *ImageModel) addFiles(body map[string]any, c *ai.ImageCall, fam family, warnings *[]ai.Warning) error {
	if c.Mask != nil && !fam.mask {
		*warnings = append(*warnings, ai.Warning{Message: fmt.Sprintf("bfl: masks are only supported by flux-pro-1.0-fill; the mask was ignored by %s", m.id)})
	}
	if len(c.Files) == 0 {
		if fam.mask {
			return errors.New("bfl: flux-pro-1.0-fill needs an input File and a Mask")
		}
		return nil
	}
	files := c.Files
	if len(files) > fam.maxFiles {
		*warnings = append(*warnings, ai.Warning{Message: fmt.Sprintf("bfl: %s takes at most %d input files; the last %d were ignored", m.id, fam.maxFiles, len(files)-fam.maxFiles)})
		files = files[:fam.maxFiles]
	}
	values := make([]string, 0, len(files))
	for _, f := range files {
		v, err := fileValue(f, fam.mask)
		if err != nil {
			return err
		}
		values = append(values, v)
	}
	maps.Copy(body, fam.files(values))
	if fam.mask {
		if c.Mask == nil {
			return errors.New("bfl: flux-pro-1.0-fill needs a Mask")
		}
		mask, err := mediaio.Base64(providerName, *c.Mask)
		if err != nil {
			return err
		}
		body["mask"] = mask
	}
	return nil
}

func fileValue(f ai.File, base64Only bool) (string, error) {
	if f.URL != "" && !base64Only {
		return f.URL, nil
	}
	return mediaio.Base64(providerName, f)
}

var _ ai.ImageModel = (*ImageModel)(nil)
