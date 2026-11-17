package replicate

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/pdata"
)

// VideoOptions are video settings passed with ai.Providers. Extra is merged into the model's input and wins over the fields mapped from the call.
// ImageField names the model's input field for the first input file and DurationField the one for the length in whole seconds.
type VideoOptions struct {
	CancelAfter   time.Duration  `json:"cancelAfter,omitzero"`
	ImageField    string         `json:"imageField,omitzero"`
	DurationField string         `json:"durationField,omitzero"`
	KeepURL       bool           `json:"keepUrl,omitzero"`
	Extra         map[string]any `json:"extra,omitzero"`
}

func (VideoOptions) ProviderName() string { return providerName }

// VideoModel runs a Replicate model that returns a video file. StartVideo creates the prediction without waiting and PollVideo reads it.
type VideoModel struct {
	p  *Provider
	id string
}

func (p *Provider) VideoModel(id string) *VideoModel { return &VideoModel{p: p, id: id} }

func NewVideoModel(id string) *VideoModel { return New().VideoModel(id) }

func (m *VideoModel) Provider() string { return providerName }

func (m *VideoModel) ModelID() string { return m.id }

func (m *VideoModel) StartVideo(ctx context.Context, c *ai.VideoCall) (*ai.VideoJob, error) {
	opts, err := mediaio.DecodeOptions[VideoOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	input, warnings, err := videoInput(c, opts)
	if err != nil {
		return nil, err
	}
	zero := time.Duration(0)
	pred, err := m.p.create(ctx, m.id, input, createOptions{wait: &zero, cancelAfter: opts.CancelAfter}, c.Headers)
	if err != nil {
		return nil, err
	}
	data := ai.ProviderData{}.WithRaw("keepUrl", pdata.JSON(providerName, opts.KeepURL))
	if len(warnings) > 0 {
		data = data.WithRaw("warnings", pdata.JSON(providerName, warnings))
	}
	return &ai.VideoJob{ID: pred.ID, Provider: data}, nil
}

func (m *VideoModel) PollVideo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	pred, err := m.p.get(ctx, job.ID)
	if err != nil {
		return nil, err
	}
	if !pred.terminal() {
		return &ai.VideoPoll{}, nil
	}
	if failure := pred.failure(); failure != nil {
		return &ai.VideoPoll{Failure: failure}, nil //nolint:nilerr // a failed job is reported through VideoPoll.Failure
	}
	keepURL, _ := ai.ProviderValue[bool](job.Provider, "keepUrl")
	warnings, _ := ai.ProviderValue[[]ai.Warning](job.Provider, "warnings")
	videos, err := m.p.files(ctx, pred, "video/mp4", mediaio.MaxVideoBytes, keepURL)
	if err != nil {
		return nil, err
	}
	return &ai.VideoPoll{Done: true, Progress: 100, Videos: videos, Warnings: warnings}, nil
}

func videoInput(c *ai.VideoCall, opts VideoOptions) (map[string]any, []ai.Warning, error) {
	input := map[string]any{"prompt": c.Prompt}
	var warnings []ai.Warning
	if c.Seed != nil {
		input["seed"] = *c.Seed
	}
	if c.AspectRatio != "" {
		input["aspect_ratio"] = c.AspectRatio
	}
	if c.Duration > 0 {
		input[cmp.Or(opts.DurationField, "duration")] = int((c.Duration + time.Second/2) / time.Second)
	}
	if len(c.Files) > 0 {
		ref, err := mediaio.Input(providerName, c.Files[0])
		if err != nil {
			return nil, nil, err
		}
		input[cmp.Or(opts.ImageField, "image")] = ref
		if len(c.Files) > 1 {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("replicate: only the first of %d input files was sent as %q; pass further inputs in VideoOptions.Extra", len(c.Files), cmp.Or(opts.ImageField, "image"))})
		}
	}
	maps.Copy(input, opts.Extra)
	return input, warnings, nil
}

var _ ai.VideoModel = (*VideoModel)(nil)
