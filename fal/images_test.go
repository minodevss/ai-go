package fal_test

import (
	"context"
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

func TestImageModel_queue(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_in_queue.json", "status_in_progress.json", "status_completed.json"}, result: "result_image.json"}.handler(t))
		start := time.Now()
		res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "a paper boat", N: 1, Size: "1024x768", Seed: new(42)})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if len(res.Images) != 1 || res.Images[0].MediaType != "image/png" || string(res.Images[0].Data) != "bytes:/files/out.png" {
			t.Errorf("GenerateImage() images = %+v, want one downloaded png", res.Images)
		}
		if got := time.Since(start); got != 2500*time.Millisecond {
			t.Errorf("elapsed = %v, want 2.5s of backoff (1s then 1.5s)", got)
		}
		reqs := srv.all()
		if reqs[0].Method != http.MethodPost || reqs[0].Path != "/fal-ai/flux/dev" {
			t.Errorf("submit = %s %s, want POST /fal-ai/flux/dev", reqs[0].Method, reqs[0].Path)
		}
		if got := reqs[0].Header.Get("Authorization"); got != "Key test-key" {
			t.Errorf("Authorization = %q, want Key test-key", got)
		}
		equalJSON(t, `{"prompt":"a paper boat","num_images":1,"image_size":{"width":1024,"height":768},"seed":42}`, reqs[0].Body)
		last := reqs[len(reqs)-1]
		if last.Path != "/files/out.png" || last.Header.Get("Authorization") != "" {
			t.Errorf("download = %s with Authorization %q, want an unauthenticated fetch", last.Path, last.Header.Get("Authorization"))
		}
	})
}

func TestImageModel_sync(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request, srv *fakeServer) {
		if strings.HasPrefix(r.URL.Path, "/files/") {
			_, _ = w.Write([]byte("png"))
			return
		}
		reply(http.StatusOK, srv.fixture(t, "result_image.json"))(w)
	})
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux/schnell"), ai.ImageRequest{Prompt: "p", Provider: ai.Providers(fal.ImageOptions{Sync: true})})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if reqs := srv.all(); reqs[0].Path != "/sync/fal-ai/flux/schnell" || len(res.Images) != 1 {
		t.Errorf("first request path = %q with %d images, want the sync host", reqs[0].Path, len(res.Images))
	}
}

func TestImageModel_inputs(t *testing.T) {
	tests := []struct {
		name string
		req  ai.ImageRequest
		want string
	}{
		{name: "enum size and ratio", req: ai.ImageRequest{Prompt: "p", Size: "landscape_4_3", AspectRatio: "16:9"}, want: `{"prompt":"p","image_size":"landscape_4_3","aspect_ratio":"16:9"}`},
		{name: "single file", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{MediaType: "image/png", Data: []byte("x")}}}, want: `{"prompt":"p","image_url":"data:image/png;base64,eA=="}`},
		{name: "several files and mask", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{URL: "https://a/1.png"}, {URL: "https://a/2.png"}}, Mask: &ai.File{URL: "https://a/m.png"}}, want: `{"prompt":"p","image_urls":["https://a/1.png","https://a/2.png"],"mask_url":"https://a/m.png"}`},
		{name: "field names and extra", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{URL: "https://a/1.png"}}, Provider: ai.Providers(fal.ImageOptions{ImageField: "reference", OutputFormat: "jpeg", SafetyTolerance: "4", EnableSafetyChecker: new(false), Extra: map[string]any{"guidance_scale": 3.5, "prompt": "override"}})}, want: `{"prompt":"override","reference":"https://a/1.png","output_format":"jpeg","safety_tolerance":"4","enable_safety_checker":false,"guidance_scale":3.5}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, queueScript{statuses: []string{"status_completed.json"}, result: "result_image.json"}.handler(t))
				if _, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux-pro/kontext"), tt.req); err != nil {
					t.Fatalf("GenerateImage() error = %v", err)
				}
				equalJSON(t, tt.want, srv.all()[0].Body)
			})
		})
	}
}

func TestImageModel_headers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_completed.json"}, result: "result_image.json"}.handler(t))
		opts := fal.ImageOptions{Priority: "low", StartTimeoutSeconds: 30, NoRetry: true}
		_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "p", Provider: ai.Providers(opts)})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		h := srv.all()[0].Header
		got := map[string]string{"priority": h.Get("X-Fal-Queue-Priority"), "timeout": h.Get("X-Fal-Request-Start-Timeout"), "noretry": h.Get("X-Fal-No-Retry")}
		want := map[string]string{"priority": "low", "timeout": "30", "noretry": "1"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("headers mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestImageModel_keepURLAndNSFW(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_completed.json"}, result: "result_image_nsfw.json"}.handler(t))
		res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "p", Provider: ai.Providers(fal.ImageOptions{KeepURL: true})})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		want := ai.File{MediaType: "image/jpeg", URL: srv.URL + "/files/out.png"}
		if diff := cmp.Diff([]ai.File{want}, res.Images); diff != "" {
			t.Errorf("images mismatch (-want +got):\n%s", diff)
		}
		if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "safety checker") {
			t.Errorf("warnings = %+v, want the safety checker warning", res.Warnings)
		}
		for _, r := range srv.all() {
			if strings.HasPrefix(r.Path, "/files/") {
				t.Errorf("fetched %s despite KeepURL", r.Path)
			}
		}
	})
}

func TestImageModel_jobFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_error.json"}}.handler(t))
		_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "p"})
		apiErr, ok := errors.AsType[*ai.APIError](err)
		if !ok || apiErr.Code != "runner_error" || apiErr.Message != "runner crashed" {
			t.Errorf("GenerateImage() error = %v, want an APIError runner_error", err)
		}
	})
}

func TestImageModel_cancelOnContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, queueScript{statuses: []string{"status_in_queue.json"}}.handler(t))
		ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
		defer cancel()
		_, err := ai.GenerateImage(ctx, srv.provider().ImageModel("fal-ai/flux/dev"), ai.ImageRequest{Prompt: "p"})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("GenerateImage() error = %v, want DeadlineExceeded", err)
		}
		reqs := srv.all()
		last := reqs[len(reqs)-1]
		if last.Method != http.MethodPut || !strings.HasSuffix(last.Path, "/cancel") {
			t.Errorf("last request = %s %s, want the cancel PUT", last.Method, last.Path)
		}
	})
}

func TestImageModel_rejectsBadInput(t *testing.T) {
	srv := newFakeServer(t, func(http.ResponseWriter, *http.Request, *fakeServer) {})
	tests := []struct {
		name string
		req  ai.ImageRequest
		want string
	}{
		{name: "empty file", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{Filename: "a.png"}}}, want: "neither data nor url"},
		{name: "bad options", req: ai.ImageRequest{Prompt: "p", Provider: ai.ProviderData{"fal": []byte(`{"sync":"yes"}`)}}, want: "decode provider options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("m"), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
	if n := len(srv.all()); n != 0 {
		t.Errorf("%d requests were sent for invalid input", n)
	}
}

func TestImageModel_foreignQueueURL(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request, _ *fakeServer) {
		reply(http.StatusOK, `{"request_id":"r","response_url":"https://evil.example/r","status_url":"https://evil.example/r/status","cancel_url":"https://evil.example/r/cancel"}`)(w)
	})
	_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("m"), ai.ImageRequest{Prompt: "p", Provider: nil})
	if err == nil || !strings.Contains(err.Error(), "not on the configured queue host") {
		t.Errorf("GenerateImage() error = %v, want a queue host error", err)
	}
}
