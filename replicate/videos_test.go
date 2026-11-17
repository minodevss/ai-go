package replicate_test

import (
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/replicate"
)

func videoRoutes(t *testing.T, polls ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var n atomic.Int32
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			jsonReply(http.StatusCreated, fixtureAt(t, "prediction_starting.json", ""))(w, r)
		case strings.HasSuffix(r.URL.Path, ".mp4"):
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("\x00\x00\x00\x18ftypmp42"))
		default:
			i := min(int(n.Add(1))-1, len(polls)-1)
			jsonReply(http.StatusOK, fixtureAt(t, polls[i], "http://api.test"))(w, r)
		}
	}
}

func TestVideoModel_generate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, videoRoutes(t, "prediction_processing.json", "prediction_succeeded_video.json"))
		m := srv.provider().VideoModel("minimax/video-01")
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{
			Prompt: "a cat", Duration: 6 * time.Second, AspectRatio: "16:9", Seed: new(3), PollInterval: time.Second,
			Files:    []ai.File{{MediaType: "image/png", Data: []byte("abc")}, {URL: "https://a/2.png"}},
			Provider: ai.Providers(replicate.VideoOptions{ImageField: "first_frame_image", CancelAfter: time.Minute}),
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if len(res.Videos) != 1 || res.Videos[0].MediaType != "video/mp4" || !strings.HasSuffix(string(res.Videos[0].Data), "ftypmp42") {
			t.Errorf("videos = %+v, want one downloaded mp4", res.Videos)
		}
		if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "first of 2") {
			t.Errorf("warnings = %+v, want the extra-file warning", res.Warnings)
		}
		create := srv.all()[0]
		if create.Path != "/v1/models/minimax/video-01/predictions" || create.Header.Get("Prefer") != "" || create.Header.Get("Cancel-After") != "60s" {
			t.Errorf("create = %s prefer=%q cancel-after=%q, want an async create", create.Path, create.Header.Get("Prefer"), create.Header.Get("Cancel-After"))
		}
		equalJSON(t, `{"input":{"prompt":"a cat","seed":3,"aspect_ratio":"16:9","duration":6,"first_frame_image":"data:image/png;base64,YWJj"}}`, create.Body)
	})
}

func TestVideoModel_keepURLAndExtra(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, videoRoutes(t, "prediction_succeeded_video.json"))
		res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("minimax/video-01"), ai.VideoRequest{
			Prompt: "a cat", PollInterval: time.Second, Duration: 5 * time.Second,
			Provider: ai.Providers(replicate.VideoOptions{KeepURL: true, DurationField: "length", Extra: map[string]any{"prompt_optimizer": true}}),
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if res.Videos[0].URL != "http://api.test/out.mp4" || len(res.Videos[0].Data) != 0 {
			t.Errorf("video = %+v, want the url only", res.Videos[0])
		}
		equalJSON(t, `{"input":{"prompt":"a cat","length":5,"prompt_optimizer":true}}`, srv.all()[0].Body)
	})
}

func TestVideoModel_failures(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		wantCode string
	}{
		{name: "moderation", fixture: "prediction_nsfw.json", wantCode: "content_policy_violation"},
		{name: "model error", fixture: "prediction_failed.json", wantCode: "job_failed"},
		{name: "canceled", fixture: "prediction_canceled.json", wantCode: "job_canceled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, videoRoutes(t, tt.fixture))
				_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("o/m"), ai.VideoRequest{Prompt: "p", PollInterval: time.Second})
				jobErr, isJob := errors.AsType[*ai.JobError](err)
				apiErr, isAPI := errors.AsType[*ai.APIError](err)
				if !isJob || !isAPI || jobErr.ID != "gm3qorzdhgbfurvjtvhg6dckhu" || apiErr.Code != tt.wantCode || apiErr.Retryable() {
					t.Errorf("GenerateVideo() error = %v, want a non-retryable JobError with %s", err, tt.wantCode)
				}
			})
		})
	}
}

func TestVideoModel_removedOutput(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		removed := strings.Replace(fixture(t, "prediction_succeeded_video.json"), `"output":"REPLACE/out.mp4"`, `"output":null`, 1)
		srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				jsonReply(http.StatusCreated, fixture(t, "prediction_starting.json"))(w, r)
				return
			}
			jsonReply(http.StatusOK, removed)(w, r)
		})
		_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("o/m"), ai.VideoRequest{Prompt: "p", PollInterval: time.Second})
		if err == nil || !strings.Contains(err.Error(), "no file urls") {
			t.Errorf("GenerateVideo() error = %v, want the removed-output error", err)
		}
	})
}
