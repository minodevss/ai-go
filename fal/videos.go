package fal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/pdata"
)

// VideoModel runs any fal video endpoint through the queue. The model id is the endpoint id, for example "fal-ai/kling-video/v2.1/master/image-to-video".
type VideoModel struct {
	p  *Provider
	id string
}

type videoOutput struct {
	Video  *outputFile  `json:"video"`
	Videos []outputFile `json:"videos"`
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
	body := map[string]any{"prompt": c.Prompt}
	if c.Duration > 0 {
		body["duration"] = durationInput(c.Duration, opts.DurationFormat)
	}
	if c.AspectRatio != "" {
		body["aspect_ratio"] = c.AspectRatio
	}
	if c.Seed != nil {
		body["seed"] = *c.Seed
	}
	if opts.EnableSafetyChecker != nil {
		body["enable_safety_checker"] = *opts.EnableSafetyChecker
	}
	files := inputFiles{files: c.Files, imageField: opts.ImageField, imagesField: opts.ImagesField}
	if err := files.apply(body); err != nil {
		return nil, err
	}
	mergeExtra(body, opts.Extra)
	headers := requestHeaders{Priority: opts.Priority, StartTimeout: opts.StartTimeoutSeconds, NoRetry: opts.NoRetry}.header(c.Headers)
	t, err := m.p.submit(ctx, m.id, body, headers)
	if err != nil {
		return nil, err
	}
	t.KeepURL = opts.KeepURL
	return &ai.VideoJob{ID: t.RequestID, Provider: pdata.Encode(providerName, t)}, nil
}

func (m *VideoModel) PollVideo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	t, ok := ai.ProviderValue[ticket](job.Provider, providerName)
	if !ok || t.StatusURL == "" {
		return nil, fmt.Errorf("fal: job %q has no queue urls; it was not started by this provider", job.ID)
	}
	st, err := m.p.status(ctx, t)
	if err != nil {
		return nil, err
	}
	if st.Status != statusCompleted {
		return &ai.VideoPoll{}, nil
	}
	if failure := st.failure(); failure != nil {
		return &ai.VideoPoll{Failure: failure}, nil //nolint:nilerr // a failed job is a poll result, not a poll error
	}
	raw, err := m.p.result(ctx, t)
	if apiErr, ok := errors.AsType[*ai.APIError](err); ok {
		return &ai.VideoPoll{Failure: apiErr}, nil
	}
	if err != nil {
		return nil, err
	}
	var out videoOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("fal: decode video result: %w", err)
	}
	list := out.Videos
	if out.Video != nil {
		list = append([]outputFile{*out.Video}, list...)
	}
	dl := m.p.downloader()
	videos := make([]ai.File, 0, len(list))
	for _, v := range list {
		f, err := dl.Result(ctx, v.URL, videoType(v.ContentType), mediaio.MaxVideoBytes, t.KeepURL)
		if err != nil {
			return nil, err
		}
		videos = append(videos, f)
	}
	return &ai.VideoPoll{Done: true, Progress: 100, Videos: videos}, nil
}

func videoType(contentType string) string {
	if contentType != "" {
		return contentType
	}
	return "video/mp4"
}

var _ ai.VideoModel = (*VideoModel)(nil)
