package openai

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"strconv"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/transport"
)

const maxImageInputBytes = 50 << 20

// ImageOptions are image settings passed with ai.Providers. Unset fields keep the API default.
type ImageOptions struct {
	Quality           string `json:"quality,omitzero"`
	OutputFormat      string `json:"outputFormat,omitzero"`
	OutputCompression *int   `json:"outputCompression,omitzero"`
	Background        string `json:"background,omitzero"`
	Moderation        string `json:"moderation,omitzero"`
	InputFidelity     string `json:"inputFidelity,omitzero"`
	SafetyIdentifier  string `json:"safetyIdentifier,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

// ImageModel runs Images API edits when the call has input Files. GPT image models always return base64 data.
type ImageModel struct {
	client *transport.Client
	id     string
}

type imageGenerationRequest struct {
	Model             string `json:"model"`
	Prompt            string `json:"prompt"`
	N                 int    `json:"n,omitzero"`
	Size              string `json:"size,omitzero"`
	Quality           string `json:"quality,omitzero"`
	OutputFormat      string `json:"output_format,omitzero"`
	OutputCompression *int   `json:"output_compression,omitzero"`
	Background        string `json:"background,omitzero"`
	Moderation        string `json:"moderation,omitzero"`
	SafetyIdentifier  string `json:"safety_identifier,omitzero"`
}

type imagesResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
		URL     string `json:"url"`
	} `json:"data"`
	OutputFormat string          `json:"output_format"`
	Usage        json.RawMessage `json:"usage"`
}

func (p *Provider) ImageModel(id string) *ImageModel {
	return &ImageModel{client: p.client, id: id}
}

func (m *ImageModel) Provider() string { return providerName }

func (m *ImageModel) ModelID() string { return m.id }

func (m *ImageModel) GenerateImage(ctx context.Context, c *ai.ImageCall) (*ai.ImageResponse, error) {
	opts, err := imageOptionsOf(c.Provider)
	if err != nil {
		return nil, err
	}
	warnings := imageWarnings(c)
	req, err := m.request(c, opts)
	if err != nil {
		return nil, err
	}
	req.Header = c.Headers
	var body imagesResponse
	header, err := m.client.JSON(ctx, req, &body)
	if err != nil {
		return nil, err
	}
	format := body.OutputFormat
	if format == "" {
		format = cmp.Or(opts.OutputFormat, "png")
	}
	images, err := decodeImages(body, format)
	if err != nil {
		return nil, err
	}
	return &ai.ImageResponse{Images: images, Usage: openaiwire.Usage(body.Usage), Warnings: warnings, Header: header}, nil
}

func (m *ImageModel) request(c *ai.ImageCall, opts ImageOptions) (transport.Request, error) {
	if len(c.Files) == 0 {
		if c.Mask != nil {
			return transport.Request{}, errors.New("openai: an image mask needs input Files to edit")
		}
		return transport.Request{Path: "/images/generations", Body: imageGenerationRequest{
			Model:             m.id,
			Prompt:            c.Prompt,
			N:                 c.N,
			Size:              c.Size,
			Quality:           opts.Quality,
			OutputFormat:      opts.OutputFormat,
			OutputCompression: opts.OutputCompression,
			Background:        opts.Background,
			Moderation:        opts.Moderation,
			SafetyIdentifier:  opts.SafetyIdentifier,
		}}, nil
	}
	f := form.New()
	f.Field("model", m.id)
	f.Field("prompt", c.Prompt)
	if c.N > 0 {
		f.Field("n", strconv.Itoa(c.N))
	}
	f.Field("size", c.Size)
	f.Field("quality", opts.Quality)
	f.Field("output_format", opts.OutputFormat)
	if opts.OutputCompression != nil {
		f.Field("output_compression", strconv.Itoa(*opts.OutputCompression))
	}
	f.Field("background", opts.Background)
	f.Field("moderation", opts.Moderation)
	f.Field("input_fidelity", opts.InputFidelity)
	f.Field("safety_identifier", opts.SafetyIdentifier)
	for i, file := range c.Files {
		data, err := imageInput(file)
		if err != nil {
			return transport.Request{}, err
		}
		f.File("image[]", fileName(file, "image", i), file.MediaType, bytes.NewReader(data))
	}
	if c.Mask != nil {
		data, err := imageInput(*c.Mask)
		if err != nil {
			return transport.Request{}, err
		}
		f.File("mask", fileName(*c.Mask, "mask", 0), c.Mask.MediaType, bytes.NewReader(data))
	}
	return f.Request("/images/edits")
}

func imageInput(f ai.File) ([]byte, error) {
	if len(f.Data) == 0 {
		return nil, fmt.Errorf("openai: image edit input %q needs Data; URLs and file ids are not accepted by the edits endpoint", cmp.Or(f.Filename, f.URL))
	}
	if len(f.Data) > maxImageInputBytes {
		return nil, fmt.Errorf("openai: image edit input exceeds %d MB", maxImageInputBytes>>20)
	}
	return f.Data, nil
}

func fileName(f ai.File, stem string, i int) string {
	if f.Filename != "" {
		return f.Filename
	}
	exts, err := mime.ExtensionsByType(f.MediaType)
	ext := ".png"
	if err == nil && len(exts) > 0 {
		ext = exts[0]
	}
	return stem + strconv.Itoa(i) + ext
}

func imageWarnings(c *ai.ImageCall) []ai.Warning {
	var out []ai.Warning
	if c.AspectRatio != "" {
		out = append(out, ai.Warning{Message: "openai: aspectRatio is not supported by the Images API and was ignored; set Size to WIDTHxHEIGHT"})
	}
	if c.Seed != nil {
		out = append(out, ai.Warning{Message: "openai: seed is not supported by the Images API and was ignored"})
	}
	return out
}

func decodeImages(body imagesResponse, format string) ([]ai.File, error) {
	images := make([]ai.File, 0, len(body.Data))
	for i, d := range body.Data {
		switch {
		case d.B64JSON != "":
			data, err := base64.StdEncoding.DecodeString(d.B64JSON)
			if err != nil {
				return nil, fmt.Errorf("openai: decode image %d: %w", i, err)
			}
			images = append(images, ai.File{MediaType: imageMediaType(format), Data: data})
		case d.URL != "":
			images = append(images, ai.File{MediaType: imageMediaType(format), URL: d.URL})
		default:
			return nil, fmt.Errorf("openai: image %d has neither data nor url", i)
		}
	}
	return images, nil
}

func imageMediaType(format string) string {
	switch format {
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	case "gif":
		return "image/gif"
	case "svg":
		return "image/svg+xml"
	default:
		return "image/png"
	}
}

func imageOptionsOf(d ai.ProviderData) (ImageOptions, error) { return decodeOptions[ImageOptions](d) }

func decodeOptions[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("openai: decode provider options: %w", err)
	}
	return o, nil
}

func readLimited(r io.Reader, limit int64, what string) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, fmt.Errorf("openai: read %s: %w", what, err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("openai: %s exceeds the %d MB limit", what, limit>>20)
	}
	return data, nil
}

var _ ai.ImageModel = (*ImageModel)(nil)
