package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"time"

	"github.com/minodevss/ai-go/internal/retry"
)

const (
	defaultPollInterval = 5 * time.Second
	maxPollInterval     = 30 * time.Second
)

type ImageRequest struct {
	Prompt      string
	N           int
	Size        string
	AspectRatio string
	Seed        *int
	Files       []File
	Mask        *File
	MaxRetries  *int
	Headers     http.Header
	Provider    ProviderData
	Telemetry   Telemetry
}

type ImageResult struct {
	Images   []File
	Usage    Usage
	Warnings []Warning
}

func GenerateImage(ctx context.Context, m ImageModel, req ImageRequest) (*ImageResult, error) {
	switch {
	case m == nil:
		return nil, errors.New("ai: nil model")
	case req.Prompt == "":
		return nil, errors.New("ai: image: empty prompt")
	case req.N < 0:
		return nil, errors.New("ai: image: negative N")
	}
	call := &ImageCall{
		Prompt:      req.Prompt,
		N:           req.N,
		Size:        req.Size,
		AspectRatio: req.AspectRatio,
		Seed:        clonePtr(req.Seed),
		Files:       slices.Clone(req.Files),
		Headers:     req.Headers.Clone(),
		Provider:    req.Provider.clone(),
	}
	if req.Mask != nil {
		mask := *req.Mask
		call.Mask = &mask
	}
	ctx, finish := startMedia(ctx, req.Telemetry, m.Provider(), m.ModelID())
	resp, err := withRetry(ctx, retriesOf(req.MaxRetries), isRetryable, func(ctx context.Context) (*ImageResponse, error) {
		return m.GenerateImage(ctx, call)
	})
	if err == nil && resp == nil {
		err = errors.New("ai: image: model returned no response")
	}
	if err != nil {
		finish(Step{}, err)
		return nil, err
	}
	finish(Step{Usage: resp.Usage, Warnings: resp.Warnings, FinishReason: FinishStop}, nil)
	return &ImageResult{Images: slices.Clone(resp.Images), Usage: resp.Usage, Warnings: resp.Warnings}, nil
}

// VideoRequest.PollInterval is the first wait between polls (0 = 5s); it grows by half each poll up to 30s unless the provider names an interval.
type VideoRequest struct {
	Prompt       string
	Duration     time.Duration
	AspectRatio  string
	Seed         *int
	Files        []File
	PollInterval time.Duration
	MaxRetries   *int
	Headers      http.Header
	Provider     ProviderData
	Telemetry    Telemetry
}

type VideoResult struct {
	ID       string
	Videos   []File
	Warnings []Warning
}

// GenerateVideo submits the job and polls it until it finishes. ctx bounds the whole wait; when it ends first, the returned *JobError carries the job id.
func GenerateVideo(ctx context.Context, m VideoModel, req VideoRequest) (*VideoResult, error) {
	switch {
	case m == nil:
		return nil, errors.New("ai: nil model")
	case req.Prompt == "":
		return nil, errors.New("ai: video: empty prompt")
	}
	call := &VideoCall{
		Prompt:      req.Prompt,
		Duration:    req.Duration,
		AspectRatio: req.AspectRatio,
		Seed:        clonePtr(req.Seed),
		Files:       slices.Clone(req.Files),
		Headers:     req.Headers.Clone(),
		Provider:    req.Provider.clone(),
	}
	ctx, finish := startMedia(ctx, req.Telemetry, m.Provider(), m.ModelID())
	res, err := runVideo(ctx, m, call, req)
	if err != nil {
		finish(Step{}, err)
		return nil, err
	}
	finish(Step{Warnings: res.Warnings, FinishReason: FinishStop}, nil)
	return res, nil
}

func runVideo(ctx context.Context, m VideoModel, call *VideoCall, req VideoRequest) (*VideoResult, error) {
	retries := retriesOf(req.MaxRetries)
	job, err := withRetry(ctx, retries, isRetryable, func(ctx context.Context) (*VideoJob, error) {
		return m.StartVideo(ctx, call)
	})
	if err == nil && (job == nil || job.ID == "") {
		err = errors.New("ai: video: model returned no job")
	}
	if err != nil {
		return nil, err
	}
	interval := req.PollInterval
	if interval <= 0 {
		interval = defaultPollInterval
	}
	var warnings []Warning
	for {
		poll, err := withRetry(ctx, retries, isRetryable, func(ctx context.Context) (*VideoPoll, error) {
			return m.PollVideo(ctx, job)
		})
		switch {
		case err != nil:
			return nil, &JobError{Kind: "video", ID: job.ID, Err: err}
		case poll == nil:
			return nil, &JobError{Kind: "video", ID: job.ID, Err: errors.New("ai: model returned no poll result")}
		case poll.Failure != nil:
			return nil, &JobError{Kind: "video", ID: job.ID, Err: poll.Failure}
		case poll.Done && len(poll.Videos) == 0:
			return nil, &JobError{Kind: "video", ID: job.ID, Err: errors.New("ai: job finished without a video")}
		case poll.Done:
			return &VideoResult{ID: job.ID, Videos: slices.Clone(poll.Videos), Warnings: append(warnings, poll.Warnings...)}, nil
		}
		warnings = append(warnings, poll.Warnings...)
		wait := interval
		if poll.Interval > 0 {
			wait = poll.Interval
		} else {
			interval = min(interval+interval/2, maxPollInterval)
		}
		if err := retry.Sleep(ctx, wait); err != nil {
			return nil, &JobError{Kind: "video", ID: job.ID, Err: err}
		}
	}
}

type SpeechRequest struct {
	Text         string
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	MaxRetries   *int
	Headers      http.Header
	Provider     ProviderData
	Telemetry    Telemetry
}

type SpeechResult struct {
	Audio    File
	Warnings []Warning
}

func GenerateSpeech(ctx context.Context, m SpeechModel, req SpeechRequest) (*SpeechResult, error) {
	switch {
	case m == nil:
		return nil, errors.New("ai: nil model")
	case req.Text == "":
		return nil, errors.New("ai: speech: empty text")
	}
	call := &SpeechCall{
		Text:         req.Text,
		Voice:        req.Voice,
		Format:       req.Format,
		Instructions: req.Instructions,
		Speed:        clonePtr(req.Speed),
		Headers:      req.Headers.Clone(),
		Provider:     req.Provider.clone(),
	}
	ctx, finish := startMedia(ctx, req.Telemetry, m.Provider(), m.ModelID())
	resp, err := withRetry(ctx, retriesOf(req.MaxRetries), isRetryable, func(ctx context.Context) (*SpeechResponse, error) {
		return m.GenerateSpeech(ctx, call)
	})
	if err == nil && resp == nil {
		err = errors.New("ai: speech: model returned no response")
	}
	if err != nil {
		finish(Step{}, err)
		return nil, err
	}
	finish(Step{Warnings: resp.Warnings, FinishReason: FinishStop}, nil)
	return &SpeechResult{Audio: resp.Audio, Warnings: resp.Warnings}, nil
}

// TranscribeRequest.Audio is read once per attempt; a failed attempt is retried only when Audio is an io.Seeker.
type TranscribeRequest struct {
	Audio      io.Reader
	MediaType  string
	Language   string
	MaxRetries *int
	Headers    http.Header
	Provider   ProviderData
	Telemetry  Telemetry
}

type TranscribeResult struct {
	Text     string
	Segments []Segment
	Language string
	Warnings []Warning
}

func Transcribe(ctx context.Context, m TranscriptionModel, req TranscribeRequest) (*TranscribeResult, error) {
	switch {
	case m == nil:
		return nil, errors.New("ai: nil model")
	case req.Audio == nil:
		return nil, errors.New("ai: transcribe: no audio")
	}
	call := &TranscribeCall{
		Audio:     req.Audio,
		MediaType: req.MediaType,
		Language:  req.Language,
		Headers:   req.Headers.Clone(),
		Provider:  req.Provider.clone(),
	}
	ctx, finish := startMedia(ctx, req.Telemetry, m.Provider(), m.ModelID())
	rewind := rewinder(req.Audio)
	attempts := 0
	canRetry := func(err error) bool { return rewind != nil && isRetryable(err) }
	resp, err := withRetry(ctx, retriesOf(req.MaxRetries), canRetry, func(ctx context.Context) (*TranscribeResponse, error) {
		if attempts++; attempts > 1 {
			if err := rewind(); err != nil {
				return nil, fmt.Errorf("ai: transcribe: rewind audio: %w", err)
			}
		}
		return m.Transcribe(ctx, call)
	})
	if err == nil && resp == nil {
		err = errors.New("ai: transcribe: model returned no response")
	}
	if err != nil {
		finish(Step{}, err)
		return nil, err
	}
	finish(Step{Warnings: resp.Warnings, FinishReason: FinishStop}, nil)
	return &TranscribeResult{Text: resp.Text, Segments: slices.Clone(resp.Segments), Language: resp.Language, Warnings: resp.Warnings}, nil
}

func rewinder(r io.Reader) func() error {
	seeker, ok := r.(io.Seeker)
	if !ok {
		return nil
	}
	start, err := seeker.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil
	}
	return func() error {
		_, err := seeker.Seek(start, io.SeekStart)
		return err
	}
}

func startMedia(ctx context.Context, t Telemetry, provider, modelID string) (context.Context, func(Step, error)) {
	if t == nil {
		return ctx, func(Step, error) {}
	}
	info := StepInfo{Provider: provider, ModelID: modelID}
	ctx = t.StartStep(ctx, info)
	return ctx, func(step Step, err error) { t.FinishStep(ctx, info, step, err) }
}
