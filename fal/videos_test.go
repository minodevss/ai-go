package fal_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/fal"
)

func TestVideoModel_generate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_in_queue.json", "status_in_progress.json", "status_completed.json"}, result: "result_video.json"}.handler(t))
		start := time.Now()
		res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("fal-ai/kling-video/v2.1/master/image-to-video"), ai.VideoRequest{
			Prompt: "waves", Duration: 5 * time.Second, AspectRatio: "16:9", Seed: new(7),
			Files: []ai.File{{URL: "https://example.com/a.png"}},
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if res.ID != "764cabcf-b745-4b3e-ae38-1200304cf45b" || len(res.Videos) != 1 || res.Videos[0].MediaType != "video/mp4" || string(res.Videos[0].Data) != "bytes:/files/out.mp4" {
			t.Errorf("GenerateVideo() = %+v, want the job id and a downloaded mp4", res)
		}
		if got := time.Since(start); got != 12500*time.Millisecond {
			t.Errorf("elapsed = %v, want 12.5s (5s then 7.5s core backoff)", got)
		}
		submit := srv.all()[0]
		if submit.Path != "/fal-ai/kling-video/v2.1/master/image-to-video" {
			t.Errorf("submit path = %q", submit.Path)
		}
		equalJSON(t, `{"prompt":"waves","duration":"5","aspect_ratio":"16:9","seed":7,"image_url":"https://example.com/a.png"}`, submit.Body)
	})
}

func TestVideoModel_options(t *testing.T) {
	tests := []struct {
		name string
		opts fal.VideoOptions
		want string
	}{
		{name: "integer duration", opts: fal.VideoOptions{DurationFormat: fal.DurationInteger}, want: `{"prompt":"p","duration":8}`},
		{name: "suffix duration", opts: fal.VideoOptions{DurationFormat: fal.DurationSuffix}, want: `{"prompt":"p","duration":"8s"}`},
		{name: "extra and field", opts: fal.VideoOptions{ImageField: "start_image_url", Extra: map[string]any{"negative_prompt": "blur"}}, want: `{"prompt":"p","duration":"8","start_image_url":"https://a/1.png","negative_prompt":"blur"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, queueScript{statuses: []string{"status_completed.json"}, result: "result_video.json"}.handler(t))
				req := ai.VideoRequest{Prompt: "p", Duration: 8 * time.Second, Provider: ai.Providers(tt.opts)}
				if tt.opts.ImageField != "" {
					req.Files = []ai.File{{URL: "https://a/1.png"}}
				}
				if _, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("fal-ai/veo"), req); err != nil {
					t.Fatalf("GenerateVideo() error = %v", err)
				}
				equalJSON(t, tt.want, srv.all()[0].Body)
			})
		})
	}
}

func TestVideoModel_keepURL(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_completed.json"}, result: "result_video.json"}.handler(t))
		res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("m"), ai.VideoRequest{Prompt: "p", Provider: ai.Providers(fal.VideoOptions{KeepURL: true})})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if diff := cmp.Diff([]ai.File{{MediaType: "video/mp4", URL: srv.URL + "/files/out.mp4"}}, res.Videos); diff != "" {
			t.Errorf("videos mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestVideoModel_failures(t *testing.T) {
	tests := []struct {
		name     string
		script   queueScript
		wantCode string
		wantRetr bool
	}{
		{name: "status error", script: queueScript{statuses: []string{"status_error.json"}}, wantCode: "runner_error"},
		{name: "content policy on result", script: queueScript{statuses: []string{"status_completed.json"}, result: "error_content_policy.json", resultOK: http.StatusUnprocessableEntity}, wantCode: "content_policy_violation"},
		{name: "timeout on result", script: queueScript{statuses: []string{"status_completed.json"}, result: "error_timeout.json", resultOK: http.StatusGatewayTimeout}, wantCode: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, tt.script.handler(t))
				_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("m"), ai.VideoRequest{Prompt: "p"})
				jobErr, ok := errors.AsType[*ai.JobError](err)
				apiErr, isAPI := errors.AsType[*ai.APIError](err)
				if !ok || !isAPI || jobErr.ID == "" {
					t.Fatalf("GenerateVideo() error = %v, want a JobError wrapping an APIError", err)
				}
				if tt.wantCode != "" && apiErr.Code != tt.wantCode {
					t.Errorf("code = %q, want %q", apiErr.Code, tt.wantCode)
				}
			})
		})
	}
}

func TestVideoModel_pollWithoutTicket(t *testing.T) {
	srv := newFakeServer(t, func(http.ResponseWriter, *http.Request, *fakeServer) {})
	_, err := srv.provider().VideoModel("m").PollVideo(t.Context(), &ai.VideoJob{ID: "x"})
	if err == nil || !strings.Contains(err.Error(), "no queue urls") {
		t.Errorf("PollVideo() error = %v, want a missing ticket error", err)
	}
}

func TestVideoModel_pollRejectsForeignTicketURLs(t *testing.T) {
	tests := []struct {
		name   string
		status string
	}{
		{name: "userinfo host takeover", status: "@evil.example/s"},
		{name: "other scheme", status: "https://fal.test/s"},
		{name: "other host", status: "http://evil.example/s"},
		{name: "userinfo on queue host", status: "http://user@fal.test/s"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(http.ResponseWriter, *http.Request, *fakeServer) {})
			ticket := map[string]any{"requestId": "1", "statusUrl": tt.status, "responseUrl": tt.status, "cancelUrl": tt.status}
			job := &ai.VideoJob{ID: "1", Provider: ticketData(t, ticket)}
			_, err := srv.provider().VideoModel("m").PollVideo(t.Context(), job)
			if err == nil || !strings.Contains(err.Error(), "not on the configured queue host") {
				t.Errorf("PollVideo() error = %v, want a foreign queue url error", err)
			}
			if n := len(srv.all()); n != 0 {
				t.Errorf("%d requests were sent, want 0", n)
			}
		})
	}
}

func TestVideoModel_pollRejectsForeignResultURL(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request, _ *fakeServer) {
		reply(http.StatusOK, `{"status":"COMPLETED"}`)(w)
	})
	ticket := map[string]any{"requestId": "1", "statusUrl": "http://fal.test/s", "responseUrl": "@evil.example/r"}
	job := &ai.VideoJob{ID: "1", Provider: ticketData(t, ticket)}
	_, err := srv.provider().VideoModel("m").PollVideo(t.Context(), job)
	if err == nil || !strings.Contains(err.Error(), "not on the configured queue host") {
		t.Errorf("PollVideo() error = %v, want a foreign queue url error", err)
	}
	for _, r := range srv.all() {
		if strings.Contains(r.Path, "evil") {
			t.Errorf("request reached %s", r.Path)
		}
	}
}

func ticketData(t *testing.T, ticket map[string]any) ai.ProviderData {
	t.Helper()
	data, err := ai.ProviderData{}.WithValue("fal", ticket)
	if err != nil {
		t.Fatalf("WithValue() error = %v", err)
	}
	return data
}
