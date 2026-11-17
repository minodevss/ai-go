package luma

import (
	"context"
	"fmt"
	"maps"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/pdata"
)

// VideoOptions are video settings passed with ai.Providers. Extra is merged into the request body and wins over the fields mapped from the call.
// Resolution is 540p, 720p, 1080p or 4k, and Concepts are camera concept keys such as "dolly_zoom".
type VideoOptions struct {
	Resolution string         `json:"resolution,omitzero"`
	Loop       *bool          `json:"loop,omitzero"`
	Concepts   []string       `json:"concepts,omitzero"`
	KeepURL    bool           `json:"keepUrl,omitzero"`
	Extra      map[string]any `json:"extra,omitzero"`
}

func (VideoOptions) ProviderName() string { return providerName }

// VideoModel generates videos with a Ray model. The first input File is the start frame and the second the end frame; a File whose Provider is GenerationID continues or leads into an earlier generation.
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
	body, warnings, err := m.body(c, opts)
	if err != nil {
		return nil, err
	}
	gen, err := m.p.create(ctx, "/generations/video", body, c.Headers)
	if err != nil {
		return nil, err
	}
	data := ai.ProviderData{}.WithRaw("keepUrl", pdata.JSON(providerName, opts.KeepURL))
	if len(warnings) > 0 {
		data = data.WithRaw("warnings", pdata.JSON(providerName, warnings))
	}
	return &ai.VideoJob{ID: gen.ID, Provider: data}, nil
}

func (m *VideoModel) PollVideo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	gen, err := m.p.get(ctx, job.ID)
	if err != nil {
		return nil, err
	}
	if !gen.terminal() {
		return &ai.VideoPoll{}, nil
	}
	if failure := gen.failure(); failure != nil {
		return &ai.VideoPoll{Failure: failure}, nil //nolint:nilerr // a failed job is reported through VideoPoll.Failure
	}
	keepURL, _ := ai.ProviderValue[bool](job.Provider, "keepUrl")
	warnings, _ := ai.ProviderValue[[]ai.Warning](job.Provider, "warnings")
	file, err := m.p.file(ctx, gen.Assets.Video, "video/mp4", mediaio.MaxVideoBytes, keepURL)
	if err != nil {
		return nil, err
	}
	return &ai.VideoPoll{Done: true, Progress: 100, Videos: []ai.File{file}, Warnings: warnings}, nil
}

type keyframe struct {
	Type string `json:"type"`
	URL  string `json:"url,omitzero"`
	ID   string `json:"id,omitzero"`
}

func (m *VideoModel) body(c *ai.VideoCall, opts VideoOptions) (map[string]any, []ai.Warning, error) {
	body := map[string]any{"model": m.id, "prompt": c.Prompt}
	var warnings []ai.Warning
	if c.AspectRatio != "" {
		body["aspect_ratio"] = c.AspectRatio
	}
	if c.Duration > 0 {
		body["duration"] = fmt.Sprintf("%ds", int((c.Duration+time.Second/2)/time.Second))
	}
	if c.Seed != nil {
		warnings = append(warnings, ai.Warning{Message: "luma: seed is not supported by the Dream Machine API and was ignored"})
	}
	if opts.Resolution != "" {
		body["resolution"] = opts.Resolution
	}
	if opts.Loop != nil {
		body["loop"] = *opts.Loop
	}
	if len(opts.Concepts) > 0 {
		concepts := make([]map[string]string, 0, len(opts.Concepts))
		for _, key := range opts.Concepts {
			concepts = append(concepts, map[string]string{"key": key})
		}
		body["concepts"] = concepts
	}
	frames, err := keyframes(c.Files)
	if err != nil {
		return nil, nil, err
	}
	if len(frames) > 0 {
		body["keyframes"] = frames
	}
	if len(c.Files) > 2 {
		warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("luma: a video takes at most 2 input files (start and end frame); the last %d were ignored", len(c.Files)-2)})
	}
	maps.Copy(body, opts.Extra)
	return body, warnings, nil
}

func keyframes(files []ai.File) (map[string]keyframe, error) {
	if len(files) == 0 {
		return nil, nil
	}
	frames := make(map[string]keyframe, 2)
	for i, f := range files[:min(len(files), 2)] {
		kf, err := frameOf(f)
		if err != nil {
			return nil, err
		}
		frames[fmt.Sprintf("frame%d", i)] = kf
	}
	return frames, nil
}

func frameOf(f ai.File) (keyframe, error) {
	if ref, ok := ai.ProviderValue[generationRef](f.Provider, providerName); ok && ref.ID != "" {
		return keyframe{Type: "generation", ID: ref.ID}, nil
	}
	u, err := inputURL(f)
	if err != nil {
		return keyframe{}, err
	}
	return keyframe{Type: "image", URL: u}, nil
}

var _ ai.VideoModel = (*VideoModel)(nil)
