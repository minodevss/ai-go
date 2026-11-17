package replicate_test

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
	"github.com/minodevss/ai-go/replicate"
)

func imageRoutes(t *testing.T, srv *func() string, creates ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var created, polled atomic.Int32
	polls := []string{"prediction_processing.json", "prediction_processing.json", "prediction_succeeded_images.json"}
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			i := min(int(created.Add(1))-1, len(creates)-1)
			status := http.StatusCreated
			if creates[i] != "prediction_succeeded_images.json" {
				status = http.StatusAccepted
			}
			jsonReply(status, fixtureAt(t, creates[i], (*srv)()))(w, r)
		case strings.HasSuffix(r.URL.Path, ".webp"):
			w.Header().Set("Content-Type", "image/webp")
			_, _ = w.Write([]byte("RIFFwebp" + r.URL.Path))
		default:
			i := min(int(polled.Add(1))-1, len(polls)-1)
			jsonReply(http.StatusOK, fixtureAt(t, polls[i], (*srv)()))(w, r)
		}
	}
}

func TestImageModel_pollsWithBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var base func() string
		srv := newFakeServer(t, imageRoutes(t, &base, "prediction_starting.json"))
		base = func() string { return srv.URL }
		m := srv.provider(replicate.WithPollInterval(time.Second)).ImageModel("black-forest-labs/flux-schnell")
		res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
			Prompt: "a paper boat", N: 2, Seed: new(7), AspectRatio: "1:1",
			Provider: ai.Providers(replicate.ImageOptions{Extra: map[string]any{"output_format": "webp"}}),
		})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if len(res.Images) != 2 || res.Images[0].MediaType != "image/webp" || string(res.Images[0].Data) != "RIFFwebp/a.webp" {
			t.Errorf("GenerateImage() images = %+v, want two downloaded webp files", res.Images)
		}
		reqs := srv.all()
		create := reqs[0]
		if create.Path != "/v1/models/black-forest-labs/flux-schnell/predictions" || create.Header.Get("Prefer") != "wait=60" || create.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("create = %s prefer=%q auth=%q", create.Path, create.Header.Get("Prefer"), create.Header.Get("Authorization"))
		}
		equalJSON(t, `{"input":{"prompt":"a paper boat","seed":7,"aspect_ratio":"1:1","num_outputs":2,"output_format":"webp"}}`, create.Body)
		var gaps []time.Duration
		var polls []captured
		for _, r := range reqs[1:] {
			if r.Path == "/v1/predictions/gm3qorzdhgbfurvjtvhg6dckhu" {
				polls = append(polls, r)
			}
		}
		for i := 1; i < len(polls); i++ {
			gaps = append(gaps, polls[i].At.Sub(polls[i-1].At))
		}
		if diff := cmp.Diff([]time.Duration{time.Second, 1500 * time.Millisecond}, gaps); len(polls) != 3 || diff != "" {
			t.Errorf("poll gaps = %v over %d polls, want 1s then 1.5s over 3 polls", gaps, len(polls))
		}
		for _, r := range reqs {
			if strings.HasSuffix(r.Path, ".webp") && r.Header.Get("Authorization") != "" {
				t.Errorf("download of %s carried credentials", r.Path)
			}
		}
	})
}

func TestImageModel_syncWait(t *testing.T) {
	var base func() string
	srv := newFakeServer(t, imageRoutes(t, &base, "prediction_succeeded_images.json"))
	base = func() string { return srv.URL }
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("black-forest-labs/flux-schnell"), ai.ImageRequest{Prompt: "p"})
	if err != nil || len(res.Images) != 2 {
		t.Fatalf("GenerateImage() = %+v, %v, want two images", res, err)
	}
	for _, r := range srv.all() {
		if r.Method == http.MethodGet && strings.HasPrefix(r.Path, "/v1/predictions") {
			t.Errorf("unexpected poll %s after a finished sync create", r.Path)
		}
	}
}

func TestImageModel_keepURL(t *testing.T) {
	var base func() string
	srv := newFakeServer(t, imageRoutes(t, &base, "prediction_succeeded_images.json"))
	base = func() string { return srv.URL }
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("black-forest-labs/flux-schnell"), ai.ImageRequest{
		Prompt: "p", Provider: ai.Providers(replicate.ImageOptions{KeepURL: true}),
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if res.Images[0].URL != srv.URL+"/a.webp" || len(res.Images[0].Data) != 0 {
		t.Errorf("image = %+v, want the url only", res.Images[0])
	}
	for _, r := range srv.all() {
		if strings.HasSuffix(r.Path, ".webp") {
			t.Errorf("KeepURL downloaded %s", r.Path)
		}
	}
}

func TestImageModel_requestShapes(t *testing.T) {
	version := strings.Repeat("ab", 32)
	tests := []struct {
		name     string
		model    string
		req      ai.ImageRequest
		wantPath string
		wantBody string
		wantWait string
	}{
		{name: "versioned model", model: "stability-ai/sdxl:" + version, req: ai.ImageRequest{Prompt: "p"}, wantPath: "/v1/predictions", wantBody: `{"version":"stability-ai/sdxl:` + version + `","input":{"prompt":"p"}}`, wantWait: "wait=60"},
		{name: "bare version id", model: version, req: ai.ImageRequest{Prompt: "p"}, wantPath: "/v1/predictions", wantBody: `{"version":"` + version + `","input":{"prompt":"p"}}`, wantWait: "wait=60"},
		{name: "size and file", model: "o/m", req: ai.ImageRequest{Prompt: "p", Size: "1024x768", Files: []ai.File{{URL: "https://example.com/a.png"}}}, wantPath: "/v1/models/o/m/predictions", wantBody: `{"input":{"prompt":"p","width":1024,"height":768,"image":"https://example.com/a.png"}}`, wantWait: "wait=60"},
		{name: "data uri and mask", model: "o/m", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{MediaType: "image/png", Data: []byte("abc")}}, Mask: &ai.File{MediaType: "image/png", Data: []byte("m")}, Provider: ai.Providers(replicate.ImageOptions{ImageField: "input_image", MaskField: "mask_image"})}, wantPath: "/v1/models/o/m/predictions", wantBody: `{"input":{"prompt":"p","input_image":"data:image/png;base64,YWJj","mask_image":"data:image/png;base64,bQ=="}}`, wantWait: "wait=60"},
		{name: "no wait", model: "o/m", req: ai.ImageRequest{Prompt: "p", Provider: ai.Providers(replicate.ImageOptions{Wait: new(time.Duration(0)), CancelAfter: 90 * time.Second})}, wantPath: "/v1/models/o/m/predictions", wantBody: `{"input":{"prompt":"p"}}`},
		{name: "short wait", model: "o/m", req: ai.ImageRequest{Prompt: "p", Provider: ai.Providers(replicate.ImageOptions{Wait: new(5 * time.Second)})}, wantPath: "/v1/models/o/m/predictions", wantBody: `{"input":{"prompt":"p"}}`, wantWait: "wait=5"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var base func() string
			srv := newFakeServer(t, imageRoutes(t, &base, "prediction_succeeded_images.json"))
			base = func() string { return srv.URL }
			if _, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel(tt.model), tt.req); err != nil {
				t.Fatalf("GenerateImage() error = %v", err)
			}
			create := srv.all()[0]
			if create.Path != tt.wantPath || create.Header.Get("Prefer") != tt.wantWait {
				t.Errorf("create = %s prefer=%q, want %s prefer=%q", create.Path, create.Header.Get("Prefer"), tt.wantPath, tt.wantWait)
			}
			equalJSON(t, tt.wantBody, create.Body)
		})
	}
}

func TestImageModel_cancelAfterHeader(t *testing.T) {
	var base func() string
	srv := newFakeServer(t, imageRoutes(t, &base, "prediction_succeeded_images.json"))
	base = func() string { return srv.URL }
	_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("o/m"), ai.ImageRequest{Prompt: "p", Provider: ai.Providers(replicate.ImageOptions{CancelAfter: time.Second})})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := srv.all()[0].Header.Get("Cancel-After"); got != "5s" {
		t.Errorf("Cancel-After = %q, want the 5s minimum", got)
	}
}

func TestImageModel_rejectsBadInput(t *testing.T) {
	srv := newFakeServer(t, jsonReply(http.StatusCreated, "{}"))
	m := srv.provider().ImageModel("o/m")
	tests := []struct {
		name string
		req  ai.ImageRequest
		want string
	}{
		{name: "bad size", req: ai.ImageRequest{Prompt: "p", Size: "big"}, want: "WIDTHxHEIGHT"},
		{name: "mask without file", req: ai.ImageRequest{Prompt: "p", Mask: &ai.File{Data: []byte("m")}}, want: "needs input Files"},
		{name: "empty file", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{MediaType: "image/png"}}}, want: "neither data nor url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateImage(t.Context(), m, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
	if _, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("not-a-model"), ai.ImageRequest{Prompt: "p"}); err == nil {
		t.Error("GenerateImage() with a malformed model id succeeded")
	}
	if len(srv.all()) != 0 {
		t.Errorf("%d requests were sent for invalid input", len(srv.all()))
	}
}

func TestImageModel_extraFilesWarn(t *testing.T) {
	var base func() string
	srv := newFakeServer(t, imageRoutes(t, &base, "prediction_succeeded_images.json"))
	base = func() string { return srv.URL }
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("o/m"), ai.ImageRequest{Prompt: "p", Files: []ai.File{{URL: "https://a/1.png"}, {URL: "https://a/2.png"}}})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "first of 2") {
		t.Errorf("warnings = %+v, want one about the second file", res.Warnings)
	}
}

func TestImageModel_failedPredictions(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		wantCode string
		wantMsg  string
	}{
		{name: "moderation", fixture: "prediction_nsfw.json", wantCode: "content_policy_violation", wantMsg: "NSFW"},
		{name: "model error", fixture: "prediction_failed.json", wantCode: "job_failed", wantMsg: "CUDA"},
		{name: "canceled", fixture: "prediction_canceled.json", wantCode: "job_canceled", wantMsg: "canceled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var base func() string
			srv := newFakeServer(t, imageRoutes(t, &base, tt.fixture))
			base = func() string { return srv.URL }
			_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("o/m"), ai.ImageRequest{Prompt: "p"})
			jobErr, isJob := errors.AsType[*ai.JobError](err)
			apiErr, isAPI := errors.AsType[*ai.APIError](err)
			if !isJob || !isAPI || jobErr.ID != "gm3qorzdhgbfurvjtvhg6dckhu" || apiErr.Code != tt.wantCode || !strings.Contains(apiErr.Message, tt.wantMsg) {
				t.Fatalf("GenerateImage() error = %v, want a JobError wrapping %s", err, tt.wantCode)
			}
			if apiErr.Retryable() {
				t.Error("a failed prediction must not be retryable")
			}
			if n := len(srv.all()); n != 1 {
				t.Errorf("%d requests, want exactly one create (no resubmission)", n)
			}
		})
	}
}

func TestImageModel_cancelsOnContextEnd(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var base func() string
		var cancels atomic.Int32
		inner := imageRoutes(t, &base, "prediction_starting.json")
		srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/cancel") {
				cancels.Add(1)
				jsonReply(http.StatusOK, fixtureAt(t, "prediction_canceled.json", base()))(w, r)
				return
			}
			if r.Method == http.MethodGet {
				jsonReply(http.StatusOK, fixtureAt(t, "prediction_processing.json", base()))(w, r)
				return
			}
			inner(w, r)
		})
		base = func() string { return srv.URL }
		ctx, stop := context.WithTimeout(t.Context(), 10*time.Second)
		defer stop()
		_, err := ai.GenerateImage(ctx, srv.provider(replicate.WithPollInterval(time.Second)).ImageModel("o/m"), ai.ImageRequest{Prompt: "p"})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != "gm3qorzdhgbfurvjtvhg6dckhu" || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("GenerateImage() error = %v, want a JobError with the deadline", err)
		}
		if cancels.Load() != 1 {
			t.Errorf("cancel requests = %d, want 1", cancels.Load())
		}
	})
}

func TestImageModel_persistentPollErrorsDoNotResubmit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var creates atomic.Int32
		srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				creates.Add(1)
				jsonReply(http.StatusAccepted, fixtureAt(t, "prediction_starting.json", ""))(w, r)
				return
			}
			jsonReply(http.StatusBadGateway, `{"detail":"bad gateway"}`)(w, r)
		})
		_, err := ai.GenerateImage(t.Context(), srv.provider(replicate.WithPollInterval(time.Second)).ImageModel("o/m"), ai.ImageRequest{Prompt: "p"})
		if _, ok := errors.AsType[*ai.JobError](err); !ok {
			t.Fatalf("GenerateImage() error = %v, want a JobError", err)
		}
		if creates.Load() != 1 {
			t.Errorf("creates = %d, want 1", creates.Load())
		}
	})
}
