package fal

import (
	"cmp"
	"maps"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
)

// ImageOptions are image settings passed with ai.Providers. Unset fields keep the model default. Extra is merged last into the model input and wins over every other field.
type ImageOptions struct {
	Sync                bool           `json:"sync,omitzero"`
	KeepURL             bool           `json:"keepUrl,omitzero"`
	OutputFormat        string         `json:"outputFormat,omitzero"`
	EnableSafetyChecker *bool          `json:"enableSafetyChecker,omitzero"`
	SafetyTolerance     string         `json:"safetyTolerance,omitzero"`
	ImageField          string         `json:"imageField,omitzero"`
	ImagesField         string         `json:"imagesField,omitzero"`
	MaskField           string         `json:"maskField,omitzero"`
	Priority            string         `json:"priority,omitzero"`
	StartTimeoutSeconds int            `json:"startTimeoutSeconds,omitzero"`
	NoRetry             bool           `json:"noRetry,omitzero"`
	Extra               map[string]any `json:"extra,omitzero"`
}

func (ImageOptions) ProviderName() string { return providerName }

type DurationFormat string

const (
	DurationString  DurationFormat = "string"
	DurationInteger DurationFormat = "integer"
	DurationSuffix  DurationFormat = "suffix"
)

// VideoOptions are video settings passed with ai.Providers. DurationFormat selects how the model spells its duration input: "5" (default), 5, or "5s". Extra is merged last into the model input.
type VideoOptions struct {
	KeepURL             bool           `json:"keepUrl,omitzero"`
	DurationFormat      DurationFormat `json:"durationFormat,omitzero"`
	EnableSafetyChecker *bool          `json:"enableSafetyChecker,omitzero"`
	ImageField          string         `json:"imageField,omitzero"`
	ImagesField         string         `json:"imagesField,omitzero"`
	Priority            string         `json:"priority,omitzero"`
	StartTimeoutSeconds int            `json:"startTimeoutSeconds,omitzero"`
	NoRetry             bool           `json:"noRetry,omitzero"`
	Extra               map[string]any `json:"extra,omitzero"`
}

func (VideoOptions) ProviderName() string { return providerName }

type inputFiles struct {
	files       []ai.File
	mask        *ai.File
	imageField  string
	imagesField string
	maskField   string
}

func (f inputFiles) apply(body map[string]any) error {
	switch len(f.files) {
	case 0:
	case 1:
		u, err := mediaio.Input(providerName, f.files[0])
		if err != nil {
			return err
		}
		body[cmp.Or(f.imageField, "image_url")] = u
	default:
		urls := make([]string, 0, len(f.files))
		for _, file := range f.files {
			u, err := mediaio.Input(providerName, file)
			if err != nil {
				return err
			}
			urls = append(urls, u)
		}
		body[cmp.Or(f.imagesField, "image_urls")] = urls
	}
	if f.mask != nil {
		u, err := mediaio.Input(providerName, *f.mask)
		if err != nil {
			return err
		}
		body[cmp.Or(f.maskField, "mask_url")] = u
	}
	return nil
}

func sizeInput(size string, body map[string]any) {
	w, h, ok := strings.Cut(size, "x")
	width, werr := strconv.Atoi(w)
	height, herr := strconv.Atoi(h)
	if ok && werr == nil && herr == nil {
		body["image_size"] = map[string]int{"width": width, "height": height}
		return
	}
	body["image_size"] = size
}

func durationInput(d time.Duration, format DurationFormat) any {
	secs := int((d + time.Second/2) / time.Second)
	switch format {
	case DurationInteger:
		return secs
	case DurationSuffix:
		return strconv.Itoa(secs) + "s"
	default:
		return strconv.Itoa(secs)
	}
}

func mergeExtra(body, extra map[string]any) {
	maps.Copy(body, extra)
}
