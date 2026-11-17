package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

func omniAspectRatios() []string { return []string{"16:9", "9:16"} }

const (
	interactionsRevision = "2026-05-20"
	maxVeoReferences     = 3
	maxOmniDuration      = 15 * time.Second
	maxOmniVideoBytes    = 512 << 20
	minOmniDuration      = 3 * time.Second
)

// VideoOptions are video settings passed with ai.Providers under "google" and apply to Veo on Vertex.
// ReferenceImages sends the call's Files as asset references instead of first and last frames.
type VideoOptions struct {
	StorageURI       string `json:"storageUri,omitzero"`
	NegativePrompt   string `json:"negativePrompt,omitzero"`
	PersonGeneration string `json:"personGeneration,omitzero"`
	Resolution       string `json:"resolution,omitzero"`
	GenerateAudio    *bool  `json:"generateAudio,omitzero"`
	SampleCount      int    `json:"sampleCount,omitzero"`
	ReferenceImages  bool   `json:"referenceImages,omitzero"`
}

func (VideoOptions) ProviderName() string { return providerName }

// VideoModel generates videos as long-running jobs: Veo on Vertex and Gemini Omni on AI Studio.
type VideoModel struct {
	backend *backend
	id      string
}

type veoRequest struct {
	Instances  []veoInstance `json:"instances"`
	Parameters veoParameters `json:"parameters"`
}

type veoInstance struct {
	Prompt          string         `json:"prompt"`
	Image           *veoMedia      `json:"image,omitzero"`
	LastFrame       *veoMedia      `json:"lastFrame,omitzero"`
	ReferenceImages []veoReference `json:"referenceImages,omitzero"`
}

type veoMedia struct {
	MimeType    string `json:"mimeType,omitzero"`
	BytesBase64 string `json:"bytesBase64Encoded,omitzero"`
	GCSURI      string `json:"gcsUri,omitzero"`
}

type veoReference struct {
	Image         veoMedia `json:"image"`
	ReferenceType string   `json:"referenceType"`
}

type veoParameters struct {
	AspectRatio      string  `json:"aspectRatio,omitzero"`
	DurationSeconds  float64 `json:"durationSeconds,omitzero"`
	Seed             *int    `json:"seed,omitzero"`
	SampleCount      int     `json:"sampleCount,omitzero"`
	StorageURI       string  `json:"storageUri,omitzero"`
	NegativePrompt   string  `json:"negativePrompt,omitzero"`
	PersonGeneration string  `json:"personGeneration,omitzero"`
	Resolution       string  `json:"resolution,omitzero"`
	GenerateAudio    *bool   `json:"generateAudio,omitzero"`
}

type veoOperation struct {
	Name     string         `json:"name"`
	Done     bool           `json:"done"`
	Error    *errorEnvelope `json:"error"`
	Response struct {
		Videos []struct {
			BytesBase64 string `json:"bytesBase64Encoded"`
			GCSURI      string `json:"gcsUri"`
			MimeType    string `json:"mimeType"`
		} `json:"videos"`
		FilteredCount   int      `json:"raiMediaFilteredCount"`
		FilteredReasons []string `json:"raiMediaFilteredReasons"`
	} `json:"response"`
}

type omniRequest struct {
	Model          string        `json:"model"`
	Input          []omniInput   `json:"input"`
	Background     bool          `json:"background"`
	ResponseFormat omniVideoSpec `json:"response_format"`
}

type omniInput struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitzero"`
	MimeType string `json:"mime_type,omitzero"`
	Data     string `json:"data,omitzero"`
	URI      string `json:"uri,omitzero"`
}

type omniVideoSpec struct {
	Type        string `json:"type"`
	AspectRatio string `json:"aspect_ratio,omitzero"`
	Duration    string `json:"duration,omitzero"`
}

type omniInteraction struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Steps []struct {
		Type    string `json:"type"`
		Content []struct {
			Type     string `json:"type"`
			MimeType string `json:"mime_type"`
			Data     string `json:"data"`
			URI      string `json:"uri"`
		} `json:"content"`
	} `json:"steps"`
}

func (p *Provider) VideoModel(id string) *VideoModel {
	return &VideoModel{backend: p.backend, id: id}
}

func (m *VideoModel) Provider() string { return m.backend.name }

func (m *VideoModel) ModelID() string { return m.id }

func (m *VideoModel) StartVideo(ctx context.Context, c *ai.VideoCall) (*ai.VideoJob, error) {
	opts, err := decodeOptions[VideoOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	if m.backend.vertex {
		return m.startVeo(ctx, c, opts)
	}
	return m.startOmni(ctx, c)
}

func (m *VideoModel) PollVideo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	if job.ID == "" {
		return nil, errors.New("google: video job has no id")
	}
	if m.backend.vertex {
		return m.pollVeo(ctx, job)
	}
	return m.pollOmni(ctx, job)
}

func (m *VideoModel) startVeo(ctx context.Context, c *ai.VideoCall, opts VideoOptions) (*ai.VideoJob, error) {
	inst := veoInstance{Prompt: c.Prompt}
	media := make([]veoMedia, len(c.Files))
	for i, f := range c.Files {
		v, err := veoFile(f)
		if err != nil {
			return nil, fmt.Errorf("google: video input %d: %w", i, err)
		}
		media[i] = v
	}
	switch {
	case opts.ReferenceImages:
		if len(media) > maxVeoReferences {
			return nil, fmt.Errorf("google: a video takes at most %d reference images, got %d", maxVeoReferences, len(media))
		}
		for _, v := range media {
			inst.ReferenceImages = append(inst.ReferenceImages, veoReference{Image: v, ReferenceType: "asset"})
		}
	case len(media) == 1:
		inst.Image = &media[0]
	case len(media) == 2:
		inst.Image, inst.LastFrame = &media[0], &media[1]
	case len(media) > 2:
		return nil, fmt.Errorf("google: a video takes a first and a last frame, got %d images; set VideoOptions.ReferenceImages to send them as references", len(media))
	}
	params := veoParameters{
		AspectRatio:      c.AspectRatio,
		Seed:             c.Seed,
		SampleCount:      opts.SampleCount,
		StorageURI:       opts.StorageURI,
		NegativePrompt:   opts.NegativePrompt,
		PersonGeneration: opts.PersonGeneration,
		Resolution:       opts.Resolution,
		GenerateAudio:    opts.GenerateAudio,
	}
	if c.Duration > 0 {
		params.DurationSeconds = c.Duration.Seconds()
	}
	var op veoOperation
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":predictLongRunning", Body: veoRequest{Instances: []veoInstance{inst}, Parameters: params}, Header: c.Headers}, &op)
	if err != nil {
		return nil, fail(err)
	}
	if op.Name == "" {
		return nil, errors.New("google: video operation has no name")
	}
	return &ai.VideoJob{ID: op.Name, Header: header}, nil
}

func veoFile(f ai.File) (veoMedia, error) {
	switch {
	case len(f.Data) > 0:
		if f.MediaType == "" {
			return veoMedia{}, errors.New("google: image with data needs a media type")
		}
		return veoMedia{MimeType: f.MediaType, BytesBase64: base64.StdEncoding.EncodeToString(f.Data)}, nil
	case strings.HasPrefix(f.URL, "gs://"):
		return veoMedia{MimeType: f.MediaType, GCSURI: f.URL}, nil
	default:
		return veoMedia{}, errors.New("google: image needs Data or a gs:// URL")
	}
}

func (m *VideoModel) pollVeo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	var op veoOperation
	body := map[string]string{"operationName": job.ID}
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":fetchPredictOperation", Body: body}, &op)
	if err != nil {
		return nil, fail(err)
	}
	return m.veoPoll(op, header)
}

func (m *VideoModel) veoPoll(op veoOperation, header http.Header) (*ai.VideoPoll, error) {
	switch {
	case !op.Done:
		return &ai.VideoPoll{Header: header}, nil
	case op.Error != nil:
		return &ai.VideoPoll{Failure: op.Error.apiError(m.backend.name, nil), Header: header}, nil
	}
	videos := make([]ai.File, 0, len(op.Response.Videos))
	for i, v := range op.Response.Videos {
		mediaType := cmp.Or(v.MimeType, "video/mp4")
		switch {
		case v.BytesBase64 != "":
			data, err := base64.StdEncoding.DecodeString(v.BytesBase64)
			if err != nil {
				return nil, fmt.Errorf("google: decode video %d: %w", i, err)
			}
			videos = append(videos, ai.File{MediaType: mediaType, Data: data})
		case v.GCSURI != "":
			videos = append(videos, ai.File{MediaType: mediaType, URL: v.GCSURI})
		}
	}
	if len(videos) == 0 {
		reason := "the operation finished without a video"
		if len(op.Response.FilteredReasons) > 0 {
			reason += ": " + strings.Join(op.Response.FilteredReasons, ", ")
		}
		return &ai.VideoPoll{Failure: transport.ErrorFromEvent(m.backend.name, "no_video", reason), Header: header}, nil
	}
	return &ai.VideoPoll{Done: true, Progress: 100, Videos: videos, Header: header}, nil
}

func (m *VideoModel) startOmni(ctx context.Context, c *ai.VideoCall) (*ai.VideoJob, error) {
	var warnings []ai.Warning
	if c.Seed != nil {
		warnings = append(warnings, ai.Warning{Message: "google: seed is not supported by Gemini Omni and was ignored"})
	}
	if c.AspectRatio != "" && !slices.Contains(omniAspectRatios(), c.AspectRatio) {
		return nil, fmt.Errorf("google: gemini omni supports the aspect ratios %s, got %q", strings.Join(omniAspectRatios(), " and "), c.AspectRatio)
	}
	spec := omniVideoSpec{Type: "video", AspectRatio: c.AspectRatio}
	if c.Duration > 0 {
		if c.Duration < minOmniDuration || c.Duration > maxOmniDuration {
			return nil, fmt.Errorf("google: a video lasts %s to %s, got %s", minOmniDuration, maxOmniDuration, c.Duration)
		}
		spec.Duration = fmt.Sprintf("%ds", int((c.Duration+time.Second/2)/time.Second))
	}
	input := []omniInput{{Type: "text", Text: c.Prompt}}
	for i, f := range c.Files {
		in, err := omniFile(f)
		if err != nil {
			return nil, fmt.Errorf("google: video input %d: %w", i, err)
		}
		input = append(input, in)
	}
	req := transport.Request{Path: "/interactions", Body: omniRequest{Model: strings.TrimPrefix(m.id, "models/"), Input: input, Background: true, ResponseFormat: spec}, Header: omniHeader(c.Headers)}
	var it omniInteraction
	header, err := m.backend.client.JSON(ctx, req, &it)
	if err != nil {
		return nil, fail(err)
	}
	if it.ID == "" {
		return nil, errors.New("google: video interaction has no id")
	}
	data := ai.ProviderData{}
	if len(warnings) > 0 {
		if data, err = data.WithValue("warnings", warnings); err != nil {
			return nil, fmt.Errorf("google: encode video warnings: %w", err)
		}
	}
	return &ai.VideoJob{ID: it.ID, Provider: data, Header: header}, nil
}

func omniFile(f ai.File) (omniInput, error) {
	mediaType := f.MediaType
	switch {
	case len(f.Data) > 0 && mediaType == "":
		return omniInput{}, errors.New("google: image with data needs a media type")
	case len(f.Data) > 0:
		return omniInput{Type: "image", MimeType: mediaType, Data: base64.StdEncoding.EncodeToString(f.Data)}, nil
	case f.URL != "":
		return omniInput{Type: "image", MimeType: mediaType, URI: f.URL}, nil
	default:
		return omniInput{}, errors.New("google: image has no data or url")
	}
}

func omniHeader(h http.Header) http.Header {
	out := h.Clone()
	if out == nil {
		out = http.Header{}
	}
	out.Set("Api-Revision", interactionsRevision)
	return out
}

func (m *VideoModel) pollOmni(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	var it omniInteraction
	req := transport.Request{Method: http.MethodGet, Path: "/interactions/" + url.PathEscape(job.ID), Header: omniHeader(nil)}
	header, err := m.backend.client.JSON(ctx, req, &it)
	if err != nil {
		return nil, fail(err)
	}
	return m.omniPoll(ctx, it, job, header)
}

func (m *VideoModel) omniPoll(ctx context.Context, it omniInteraction, job *ai.VideoJob, header http.Header) (*ai.VideoPoll, error) {
	switch it.Status {
	case "completed":
		return m.omniVideos(ctx, it, job, header)
	case "failed", "cancelled", "requires_action": //nolint:misspell // the Interactions API spells the status this way
		code, message := "video_"+it.Status, "video generation "+it.Status
		if it.Error != nil {
			code, message = cmp.Or(it.Error.Code, code), cmp.Or(it.Error.Message, message)
		}
		return &ai.VideoPoll{Failure: transport.ErrorFromEvent(m.backend.name, code, message), Header: header}, nil
	default:
		return &ai.VideoPoll{Header: header}, nil
	}
}

func (m *VideoModel) omniVideos(ctx context.Context, it omniInteraction, job *ai.VideoJob, header http.Header) (*ai.VideoPoll, error) {
	var videos []ai.File
	for _, step := range it.Steps {
		if step.Type != "model_output" {
			continue
		}
		for _, c := range step.Content {
			if c.Type != "video" {
				continue
			}
			mediaType := cmp.Or(c.MimeType, "video/mp4")
			switch {
			case c.Data != "":
				data, err := base64.StdEncoding.DecodeString(c.Data)
				if err != nil {
					return nil, fmt.Errorf("google: decode video data: %w", err)
				}
				videos = append(videos, ai.File{MediaType: mediaType, Data: data})
			case c.URI != "":
				file, err := m.omniURIFile(ctx, c.URI, mediaType)
				if err != nil {
					return nil, err
				}
				videos = append(videos, file)
			}
		}
	}
	if len(videos) == 0 {
		return &ai.VideoPoll{Failure: transport.ErrorFromEvent(m.backend.name, "no_video", "the interaction finished without a video"), Header: header}, nil
	}
	warnings, _ := ai.ProviderValue[[]ai.Warning](job.Provider, "warnings")
	return &ai.VideoPoll{Done: true, Progress: 100, Videos: videos, Warnings: warnings, Header: header}, nil
}

func (m *VideoModel) omniURIFile(ctx context.Context, uri, mediaType string) (ai.File, error) {
	if !m.backend.sameHost(uri) {
		return ai.File{MediaType: mediaType, URL: uri}, nil
	}
	resp, err := m.backend.send(ctx, http.MethodGet, uri, nil, nil, nil, true)
	if err != nil {
		return ai.File{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxOmniVideoBytes+1))
	if err != nil {
		return ai.File{}, fmt.Errorf("google: download video: %w", err)
	}
	if len(data) > maxOmniVideoBytes {
		return ai.File{}, fmt.Errorf("google: video exceeds %d MB", maxOmniVideoBytes>>20)
	}
	return ai.File{MediaType: mediaType, Data: data}, nil
}

func (b *backend) sameHost(target string) bool {
	u, err := url.Parse(target)
	if err != nil {
		return false
	}
	base, err := url.Parse(b.client.URL(""))
	return err == nil && u.Scheme == base.Scheme && u.Host == base.Host
}

var _ ai.VideoModel = (*VideoModel)(nil)
