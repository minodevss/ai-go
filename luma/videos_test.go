package luma_test

import (
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/luma"
)

func TestVideoModel_generate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, "generation_queued.json", "generation_dreaming.json", "generation_video_completed.json"))
		m := srv.provider().VideoModel("ray-2")
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{
			Prompt: "A tiger walking in snow", Duration: 5 * time.Second, AspectRatio: "16:9", Seed: new(1), PollInterval: time.Second,
			Files: []ai.File{{URL: "https://cdn/start.jpg"}, {Provider: luma.GenerationID("gen-9")}, {URL: "https://cdn/extra.jpg"}},
			Provider: ai.Providers(luma.VideoOptions{
				Resolution: "720p", Loop: new(false), Concepts: []string{"dolly_zoom"},
			}),
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if res.ID != videoID || len(res.Videos) != 1 || res.Videos[0].MediaType != "video/mp4" || string(res.Videos[0].Data) != "mp4-bytes" {
			t.Errorf("GenerateVideo() = %+v, want one downloaded mp4", res)
		}
		var messages []string
		for _, w := range res.Warnings {
			messages = append(messages, w.Message)
		}
		joined := strings.Join(messages, "\n")
		if !strings.Contains(joined, "seed") || !strings.Contains(joined, "at most 2") {
			t.Errorf("warnings = %q, want seed and extra-file warnings", joined)
		}
		create := srv.all()[0]
		if create.Path != "/v1/generations/video" {
			t.Errorf("create path = %s", create.Path)
		}
		equalJSON(t, `{"model":"ray-2","prompt":"A tiger walking in snow","aspect_ratio":"16:9","duration":"5s","resolution":"720p","loop":false,"concepts":[{"key":"dolly_zoom"}],"keyframes":{"frame0":{"type":"image","url":"https://cdn/start.jpg"},"frame1":{"type":"generation","id":"gen-9"}}}`, create.Body)
	})
}

func TestVideoModel_keepURL(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, "generation_queued.json", "generation_video_completed.json"))
		res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("ray-flash-2"), ai.VideoRequest{
			Prompt: "p", PollInterval: time.Second, Duration: 9 * time.Second,
			Provider: ai.Providers(luma.VideoOptions{KeepURL: true, Extra: map[string]any{"callback_url": "https://example.com/hook"}}),
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if res.Videos[0].URL != "http://cdn.test/video.mp4" || len(res.Videos[0].Data) != 0 {
			t.Errorf("video = %+v, want the url only", res.Videos[0])
		}
		equalJSON(t, `{"model":"ray-flash-2","prompt":"p","duration":"9s","callback_url":"https://example.com/hook"}`, srv.all()[0].Body)
	})
}

func TestVideoModel_failures(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		wantCode string
	}{
		{name: "moderation", fixture: "generation_failed_moderation.json", wantCode: "content_policy_violation"},
		{name: "frames", fixture: "generation_failed_frames.json", wantCode: "job_failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, "generation_queued.json", tt.fixture))
				_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("ray-2"), ai.VideoRequest{Prompt: "p", PollInterval: time.Second})
				jobErr, isJob := errors.AsType[*ai.JobError](err)
				apiErr, isAPI := errors.AsType[*ai.APIError](err)
				if !isJob || !isAPI || jobErr.ID != videoID || apiErr.Code != tt.wantCode || apiErr.Retryable() {
					t.Errorf("GenerateVideo() error = %v, want a non-retryable JobError with %s", err, tt.wantCode)
				}
			})
		})
	}
}

func TestVideoModel_rejectsBytes(t *testing.T) {
	srv := newFakeServer(t, routes(t, "generation_queued.json", "generation_video_completed.json"))
	_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("ray-2"), ai.VideoRequest{Prompt: "p", Files: []ai.File{{Data: []byte("x")}}})
	if err == nil || !strings.Contains(err.Error(), "must be URLs") || len(srv.all()) != 0 {
		t.Errorf("GenerateVideo() error = %v after %d requests, want the URL-only error before any request", err, len(srv.all()))
	}
}
