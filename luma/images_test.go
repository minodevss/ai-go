package luma_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/luma"
)

const (
	imageID = "223e4567-e89b-12d3-a456-426614174001"
	videoID = "123e4567-e89b-12d3-a456-426614174000"
)

func routes(t *testing.T, created string, polls ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var n atomic.Int32
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			jsonReply(http.StatusCreated, fixtureAt(t, created, ""))(w, r)
		case strings.HasSuffix(r.URL.Path, ".jpg"):
			w.Header().Set("Content-Type", "image/jpeg")
			_, _ = w.Write([]byte("jpeg-bytes"))
		case strings.HasSuffix(r.URL.Path, ".mp4"):
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("mp4-bytes"))
		default:
			i := min(int(n.Add(1))-1, len(polls)-1)
			jsonReply(http.StatusOK, fixtureAt(t, polls[i], "http://cdn.test"))(w, r)
		}
	}
}

func TestImageModel_generate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, "generation_image_queued.json", "generation_image_queued.json", "generation_image_queued.json", "generation_image_completed.json"))
		m := srv.provider(luma.WithPollInterval(time.Second)).ImageModel("photon-1")
		res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
			Prompt: "A teddy bear in sunglasses playing electric guitar and dancing", AspectRatio: "3:4", Seed: new(1), Size: "10x10", N: 2,
			Mask:     &ai.File{Data: []byte("m")},
			Provider: ai.Providers(luma.ImageOptions{Format: "jpg"}),
		})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if len(res.Images) != 1 || res.Images[0].MediaType != "image/jpeg" || string(res.Images[0].Data) != "jpeg-bytes" {
			t.Errorf("images = %+v, want one downloaded jpeg", res.Images)
		}
		if len(res.Warnings) != 4 {
			t.Errorf("warnings = %+v, want n, seed, size and mask warnings", res.Warnings)
		}
		reqs := srv.all()
		create := reqs[0]
		if create.Path != "/v1/generations/image" || create.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("create = %s auth=%q", create.Path, create.Header.Get("Authorization"))
		}
		equalJSON(t, `{"model":"photon-1","prompt":"A teddy bear in sunglasses playing electric guitar and dancing","aspect_ratio":"3:4","format":"jpg"}`, create.Body)
		var gaps []time.Duration
		var polls []captured
		for _, r := range reqs[1:] {
			if r.Path == "/v1/generations/"+imageID {
				polls = append(polls, r)
			}
		}
		for i := 1; i < len(polls); i++ {
			gaps = append(gaps, polls[i].At.Sub(polls[i-1].At))
		}
		if diff := cmp.Diff([]time.Duration{time.Second, 1500 * time.Millisecond}, gaps); len(polls) != 3 || diff != "" {
			t.Errorf("poll gaps = %v over %d polls, want 1s, 1.5s over 3 polls", gaps, len(polls))
		}
		for _, r := range reqs {
			if r.Path == "/image.jpg" && r.Header.Get("Authorization") != "" {
				t.Error("download carried credentials")
			}
		}
	})
}

func TestImageModel_keepURL(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, "generation_image_completed.json", "generation_image_completed.json"))
		res, err := ai.GenerateImage(t.Context(), srv.provider(luma.WithPollInterval(time.Second)).ImageModel("photon-flash-1"), ai.ImageRequest{
			Prompt: "p", Provider: ai.Providers(luma.ImageOptions{KeepURL: true, Format: "png", Extra: map[string]any{"sync": false}}),
		})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if res.Images[0].URL != "http://cdn.test/image.jpg" || res.Images[0].MediaType != "image/png" || len(res.Images[0].Data) != 0 {
			t.Errorf("image = %+v, want the url only", res.Images[0])
		}
		equalJSON(t, `{"model":"photon-flash-1","prompt":"p","format":"png","sync":false}`, srv.all()[0].Body)
	})
}

func TestImageModel_fileRoles(t *testing.T) {
	a, b := ai.File{URL: "https://cdn/a.jpg"}, ai.File{URL: "https://cdn/b.jpg"}
	tests := []struct {
		name     string
		files    []ai.File
		opts     luma.ImageOptions
		wantBody string
		wantErr  string
	}{
		{name: "image ref default", files: []ai.File{a, b}, opts: luma.ImageOptions{FileWeight: new(0.85)}, wantBody: `{"model":"photon-1","prompt":"p","image_ref":[{"url":"https://cdn/a.jpg","weight":0.85},{"url":"https://cdn/b.jpg","weight":0.85}]}`},
		{name: "style ref", files: []ai.File{a}, opts: luma.ImageOptions{FileRole: "style_ref", FileWeight: new(0.8)}, wantBody: `{"model":"photon-1","prompt":"p","style_ref":[{"url":"https://cdn/a.jpg","weight":0.8}]}`},
		{name: "character ref", files: []ai.File{a, b}, opts: luma.ImageOptions{FileRole: "character_ref"}, wantBody: `{"model":"photon-1","prompt":"p","character_ref":{"identity0":{"images":["https://cdn/a.jpg","https://cdn/b.jpg"]}}}`},
		{name: "modify image", files: []ai.File{a}, opts: luma.ImageOptions{FileRole: "modify_image_ref", FileWeight: new(0.1)}, wantBody: `{"model":"photon-1","prompt":"p","modify_image_ref":{"url":"https://cdn/a.jpg","weight":0.1}}`},
		{name: "modify takes one", files: []ai.File{a, b}, opts: luma.ImageOptions{FileRole: "modify_image_ref"}, wantErr: "at most 1"},
		{name: "five refs", files: []ai.File{a, a, a, a, a}, wantErr: "at most 4"},
		{name: "bytes are rejected", files: []ai.File{{Data: []byte("x")}}, wantErr: "must be URLs"},
		{name: "unknown role", files: []ai.File{a}, opts: luma.ImageOptions{FileRole: "mood"}, wantErr: "FileRole must be"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, "generation_image_completed.json", "generation_image_completed.json"))
				_, err := ai.GenerateImage(t.Context(), srv.provider(luma.WithPollInterval(time.Second)).ImageModel("photon-1"), ai.ImageRequest{Prompt: "p", Files: tt.files, Provider: ai.Providers(tt.opts)})
				if tt.wantErr != "" {
					if err == nil || !strings.Contains(err.Error(), tt.wantErr) || len(srv.all()) != 0 {
						t.Errorf("GenerateImage() error = %v after %d requests, want %q before any request", err, len(srv.all()), tt.wantErr)
					}
					return
				}
				if err != nil {
					t.Fatalf("GenerateImage() error = %v", err)
				}
				equalJSON(t, tt.wantBody, srv.all()[0].Body)
			})
		})
	}
}

func TestImageModel_failures(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		wantCode string
		wantHTTP int
	}{
		{name: "moderation", fixture: "generation_failed_moderation.json", wantCode: "content_policy_violation", wantHTTP: 422},
		{name: "unreadable frames", fixture: "generation_failed_frames.json", wantCode: "job_failed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, "generation_image_queued.json", tt.fixture))
				_, err := ai.GenerateImage(t.Context(), srv.provider(luma.WithPollInterval(time.Second)).ImageModel("photon-1"), ai.ImageRequest{Prompt: "p"})
				jobErr, isJob := errors.AsType[*ai.JobError](err)
				apiErr, isAPI := errors.AsType[*ai.APIError](err)
				if !isJob || !isAPI || jobErr.ID != imageID || apiErr.Code != tt.wantCode || apiErr.StatusCode != tt.wantHTTP || apiErr.Retryable() {
					t.Errorf("GenerateImage() error = %v, want a non-retryable JobError with %s", err, tt.wantCode)
				}
			})
		})
	}
}

func TestImageModel_contextEndsWhilePolling(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, "generation_image_queued.json", "generation_image_queued.json"))
		ctx, stop := context.WithTimeout(t.Context(), 10*time.Second)
		defer stop()
		_, err := ai.GenerateImage(ctx, srv.provider(luma.WithPollInterval(time.Second)).ImageModel("photon-1"), ai.ImageRequest{Prompt: "p"})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != imageID || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("GenerateImage() error = %v, want a JobError with the id and the deadline", err)
		}
	})
}
