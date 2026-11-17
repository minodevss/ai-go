package openai

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxVideoBytes     = 512 << 20
	maxReferenceBytes = 20 << 20
)

// VideoOptions are passed with ai.Providers. Size is WIDTHxHEIGHT, such as "1280x720", and wins over the call AspectRatio.
type VideoOptions struct {
	Size string `json:"size,omitzero"`
}

func (VideoOptions) ProviderName() string { return providerName }

// VideoModel creates jobs with StartVideo and downloads the finished mp4 in PollVideo.
type VideoModel struct {
	client *transport.Client
	id     string
}

type videoJobBody struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Progress int    `json:"progress"`
	Error    *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type videoReference struct {
	ImageURL string `json:"image_url,omitzero"`
	FileID   string `json:"file_id,omitzero"`
}

type videoCreateRequest struct {
	Model          string          `json:"model"`
	Prompt         string          `json:"prompt"`
	Seconds        string          `json:"seconds,omitzero"`
	Size           string          `json:"size,omitzero"`
	InputReference *videoReference `json:"input_reference,omitzero"`
}

func (p *Provider) VideoModel(id string) *VideoModel {
	return &VideoModel{client: p.client, id: id}
}

func (m *VideoModel) Provider() string { return providerName }

func (m *VideoModel) ModelID() string { return m.id }

func (m *VideoModel) StartVideo(ctx context.Context, c *ai.VideoCall) (*ai.VideoJob, error) {
	opts, err := decodeOptions[VideoOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	size, warnings := videoSize(c, opts)
	if len(c.Files) > 1 {
		return nil, fmt.Errorf("openai: a video takes at most one reference image, got %d", len(c.Files))
	}
	seconds := ""
	if c.Duration > 0 {
		var warning *ai.Warning
		seconds, warning = videoSeconds(c.Duration)
		if warning != nil {
			warnings = append(warnings, *warning)
		}
	}
	req := transport.Request{Path: "/videos", Header: c.Headers}
	body := videoCreateRequest{Model: m.id, Prompt: c.Prompt, Seconds: seconds, Size: size}
	if len(c.Files) == 1 {
		req, err = m.referenceRequest(c, body)
		if err != nil {
			return nil, err
		}
	} else {
		req.Body = body
	}
	var job videoJobBody
	header, err := m.client.JSON(ctx, req, &job)
	if err != nil {
		return nil, err
	}
	if job.ID == "" {
		return nil, errors.New("openai: video job has no id")
	}
	data := ai.ProviderData{}
	if len(warnings) > 0 {
		data = data.WithRaw("warnings", pdata.JSON(providerName, warnings))
	}
	return &ai.VideoJob{ID: job.ID, Provider: data, Header: header}, nil
}

func (m *VideoModel) referenceRequest(c *ai.VideoCall, body videoCreateRequest) (transport.Request, error) {
	file := c.Files[0]
	if ref := openaiwire.FileRefOf(providerName, file.Provider); ref.FileID != "" {
		body.InputReference = &videoReference{FileID: ref.FileID}
		return transport.Request{Path: "/videos", Body: body, Header: c.Headers}, nil
	}
	if file.URL != "" {
		body.InputReference = &videoReference{ImageURL: file.URL}
		return transport.Request{Path: "/videos", Body: body, Header: c.Headers}, nil
	}
	if len(file.Data) == 0 {
		return transport.Request{}, errors.New("openai: video reference image has no data, url or file id")
	}
	if len(file.Data) > maxReferenceBytes {
		return transport.Request{}, fmt.Errorf("openai: video reference image exceeds %d MB", maxReferenceBytes>>20)
	}
	f := form.New()
	f.Field("model", body.Model)
	f.Field("prompt", body.Prompt)
	f.Field("seconds", body.Seconds)
	f.Field("size", body.Size)
	f.File("input_reference", fileName(file, "reference", 0), file.MediaType, bytes.NewReader(file.Data))
	req, err := f.Request("/videos")
	req.Header = c.Headers
	return req, err
}

func (m *VideoModel) PollVideo(ctx context.Context, job *ai.VideoJob) (*ai.VideoPoll, error) {
	if err := checkID(job.ID); err != nil {
		return nil, err
	}
	var body videoJobBody
	header, err := m.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/videos/" + escapeID(job.ID)}, &body)
	if err != nil {
		return nil, err
	}
	warnings, _ := ai.ProviderValue[[]ai.Warning](job.Provider, "warnings")
	switch body.Status {
	case "failed":
		code, message := "video_failed", "video generation failed"
		if body.Error != nil {
			code, message = cmp.Or(body.Error.Code, code), cmp.Or(body.Error.Message, message)
		}
		return &ai.VideoPoll{Failure: transport.ErrorFromEvent(providerName, code, message), Header: header}, nil
	case "completed":
		data, err := fetch(ctx, m.client, transport.Request{Method: http.MethodGet, Path: "/videos/" + escapeID(job.ID) + "/content?variant=video", Header: http.Header{"Accept": {"video/mp4"}}}, maxVideoBytes)
		if err != nil {
			return nil, err
		}
		return &ai.VideoPoll{Done: true, Progress: 100, Videos: []ai.File{{MediaType: "video/mp4", Data: data}}, Warnings: warnings, Header: header}, nil
	default:
		return &ai.VideoPoll{Progress: body.Progress, Header: header}, nil
	}
}

func videoSeconds(d time.Duration) (string, *ai.Warning) {
	want := int((d + time.Second/2) / time.Second)
	allowed := []int{4, 8, 12}
	best := allowed[0]
	for _, s := range allowed[1:] {
		if abs(s-want) <= abs(best-want) {
			best = s
		}
	}
	if best == want {
		return strconv.Itoa(best), nil
	}
	return strconv.Itoa(best), &ai.Warning{Message: fmt.Sprintf("openai: video duration %s is not one of 4, 8 or 12 seconds and was changed to %d seconds", d, best)}
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func videoSize(c *ai.VideoCall, opts VideoOptions) (string, []ai.Warning) {
	var warnings []ai.Warning
	if c.Seed != nil {
		warnings = append(warnings, ai.Warning{Message: "openai: seed is not supported by the Videos API and was ignored"})
	}
	switch {
	case opts.Size != "":
		return opts.Size, warnings
	case c.AspectRatio == "":
		return "", warnings
	}
	switch c.AspectRatio {
	case "16:9":
		return "1280x720", warnings
	case "9:16":
		return "720x1280", warnings
	default:
		return "", append(warnings, ai.Warning{Message: fmt.Sprintf("openai: aspectRatio %q has no matching video size and was ignored; set VideoOptions.Size", c.AspectRatio)})
	}
}

func fetch(ctx context.Context, client *transport.Client, req transport.Request, limit int64) ([]byte, error) {
	resp, err := client.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return readLimited(resp.Body, limit, "response body")
}

var _ ai.VideoModel = (*VideoModel)(nil)
