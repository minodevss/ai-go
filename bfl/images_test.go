package bfl_test

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
	"github.com/minodevss/ai-go/bfl"
)

const taskID = "7ab4d3e1-2c5f-4a1e-9b1d-0a6c3f2e8d11"

func routes(t *testing.T, submit string, polls ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var n atomic.Int32
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			jsonReply(http.StatusOK, submit)(w, r)
		case strings.HasSuffix(r.URL.Path, ".png"):
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("\x89PNG-bytes"))
		default:
			i := min(int(n.Add(1))-1, len(polls)-1)
			jsonReply(http.StatusOK, fixtureAt(t, polls[i], "http://delivery.eu.bfl.ai"))(w, r)
		}
	}
}

func TestImageModel_generate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, fixture(t, "submit.json"), "pending.json", "pending.json", "ready.json"))
		m := srv.provider(bfl.WithPollInterval(500 * time.Millisecond)).ImageModel("flux-2-pro")
		res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
			Prompt: "A serene landscape with mountains", Size: "1440x810", Seed: new(42), AspectRatio: "16:9",
			Provider: ai.Providers(bfl.ImageOptions{OutputFormat: "png", SafetyTolerance: new(3)}),
		})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if len(res.Images) != 1 || res.Images[0].MediaType != "image/png" || string(res.Images[0].Data) != "\x89PNG-bytes" {
			t.Errorf("images = %+v, want one downloaded png", res.Images)
		}
		if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "aspectRatio") {
			t.Errorf("warnings = %+v, want the aspectRatio warning", res.Warnings)
		}
		reqs := srv.all()
		submit := reqs[0]
		if submit.Path != "/v1/flux-2-pro" || submit.Header.Get("X-Key") != "test-key" {
			t.Errorf("submit = %s x-key=%q", submit.Path, submit.Header.Get("X-Key"))
		}
		equalJSON(t, `{"prompt":"A serene landscape with mountains","width":1440,"height":810,"seed":42,"output_format":"png","safety_tolerance":3}`, submit.Body)
		var gaps []time.Duration
		var polls []captured
		for _, r := range reqs[1:] {
			if r.Path == "/v1/get_result" {
				if r.Header.Get("X-Key") != "test-key" {
					t.Errorf("poll without x-key")
				}
				polls = append(polls, r)
			}
		}
		for i := 1; i < len(polls); i++ {
			gaps = append(gaps, polls[i].At.Sub(polls[i-1].At))
		}
		if diff := cmp.Diff([]time.Duration{500 * time.Millisecond, 750 * time.Millisecond}, gaps); len(polls) != 3 || diff != "" {
			t.Errorf("poll gaps = %v over %d polls, want 0.5s then 0.75s over 3 polls", gaps, len(polls))
		}
		last := reqs[len(reqs)-1]
		if last.Path != "/result.png" || last.Header.Get("X-Key") != "" {
			t.Errorf("download = %s x-key=%q, want the result without credentials", last.Path, last.Header.Get("X-Key"))
		}
	})
}

func TestImageModel_keepURL(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, fixture(t, "submit.json"), "ready.json"))
		res, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithPollInterval(time.Second)).ImageModel("flux-pro-1.1"), ai.ImageRequest{
			Prompt: "p", Provider: ai.Providers(bfl.ImageOptions{KeepURL: true}),
		})
		if err != nil {
			t.Fatalf("GenerateImage() error = %v", err)
		}
		if res.Images[0].URL != "http://delivery.eu.bfl.ai/result.png" || len(res.Images[0].Data) != 0 {
			t.Errorf("image = %+v, want the signed url only", res.Images[0])
		}
		if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "10 minutes") {
			t.Errorf("warnings = %+v, want the expiry warning", res.Warnings)
		}
	})
}

func TestImageModel_requestBodies(t *testing.T) {
	png := ai.File{MediaType: "image/png", Data: []byte("abc")}
	tests := []struct {
		name     string
		model    string
		req      ai.ImageRequest
		wantBody string
		wantWarn string
	}{
		{name: "kontext edit with references", model: "flux-kontext-pro", req: ai.ImageRequest{Prompt: "make it blue", AspectRatio: "1:1", Seed: new(1), Files: []ai.File{png, {URL: "https://a/2.png"}}}, wantBody: `{"prompt":"make it blue","aspect_ratio":"1:1","seed":1,"input_image":"YWJj","input_image_2":"https://a/2.png"}`},
		{name: "kontext too many files", model: "flux-kontext-max", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{png, png, png, png, png}}, wantBody: `{"prompt":"p","input_image":"YWJj","input_image_2":"YWJj","input_image_3":"YWJj","input_image_4":"YWJj"}`, wantWarn: "at most 4"},
		{name: "flux 3 images array", model: "flux-3-image", req: ai.ImageRequest{Prompt: "p", Seed: new(1), Files: []ai.File{png}}, wantBody: `{"prompt":"p","images":["YWJj"]}`, wantWarn: "seed"},
		{name: "ultra image prompt", model: "flux-pro-1.1-ultra", req: ai.ImageRequest{Prompt: "p", AspectRatio: "21:9", Files: []ai.File{png}}, wantBody: `{"prompt":"p","aspect_ratio":"21:9","image_prompt":"YWJj"}`},
		{name: "kontext size warns", model: "flux-kontext-pro", req: ai.ImageRequest{Prompt: "p", Size: "1024x1024"}, wantBody: `{"prompt":"p"}`, wantWarn: "size"},
		{name: "fill with mask", model: "flux-pro-1.0-fill", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{png}, Mask: &ai.File{Data: []byte("m")}}, wantBody: `{"prompt":"p","image":"YWJj","mask":"bQ=="}`},
		{name: "mask ignored elsewhere", model: "flux-2-pro", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{png}, Mask: &ai.File{Data: []byte("m")}}, wantBody: `{"prompt":"p","input_image":"YWJj"}`, wantWarn: "masks"},
		{name: "n warns", model: "flux-dev", req: ai.ImageRequest{Prompt: "p", N: 3}, wantBody: `{"prompt":"p"}`, wantWarn: "n=3"},
		{name: "extra wins", model: "flux-dev", req: ai.ImageRequest{Prompt: "p", Provider: ai.Providers(bfl.ImageOptions{PromptUpsampling: new(true), Extra: map[string]any{"steps": 40, "prompt": "override"}})}, wantBody: `{"prompt":"override","prompt_upsampling":true,"steps":40}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, fixture(t, "submit.json"), "ready.json"))
				res, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithPollInterval(time.Second)).ImageModel(tt.model), tt.req)
				if err != nil {
					t.Fatalf("GenerateImage() error = %v", err)
				}
				equalJSON(t, tt.wantBody, srv.all()[0].Body)
				var messages []string
				for _, w := range res.Warnings {
					messages = append(messages, w.Message)
				}
				joined := strings.Join(messages, "\n")
				if (tt.wantWarn == "") != (joined == "") || !strings.Contains(joined, tt.wantWarn) {
					t.Errorf("warnings = %q, want one containing %q", joined, tt.wantWarn)
				}
			})
		})
	}
}

func TestImageModel_rejectsBadInput(t *testing.T) {
	srv := newFakeServer(t, jsonReply(http.StatusOK, "{}"))
	tests := []struct {
		name  string
		model string
		req   ai.ImageRequest
		want  string
	}{
		{name: "bad size", model: "flux-dev", req: ai.ImageRequest{Prompt: "p", Size: "big"}, want: "WIDTHxHEIGHT"},
		{name: "empty file", model: "flux-kontext-pro", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{}}}, want: "needs Data"},
		{name: "fill without file", model: "flux-pro-1.0-fill", req: ai.ImageRequest{Prompt: "p"}, want: "needs an input File"},
		{name: "fill without mask", model: "flux-pro-1.0-fill", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{Data: []byte("a")}}}, want: "needs a Mask"},
		{name: "fill with url", model: "flux-pro-1.0-fill", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{URL: "https://a/1.png"}}, Mask: &ai.File{Data: []byte("m")}}, want: "needs Data"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel(tt.model), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
	if len(srv.all()) != 0 {
		t.Errorf("%d requests were sent for invalid input", len(srv.all()))
	}
}

func TestImageModel_failedTasks(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		wantCode string
		wantMsg  string
		wantHTTP int
	}{
		{name: "request moderated", fixture: "request_moderated.json", wantCode: "content_policy_violation", wantMsg: "Violence, Sexual Content, Self Harm", wantHTTP: 422},
		{name: "content moderated", fixture: "content_moderated.json", wantCode: "content_policy_violation", wantMsg: "Content Moderated: Violence", wantHTTP: 422},
		{name: "error", fixture: "error.json", wantCode: "job_failed", wantMsg: "inference failed"},
		{name: "not found", fixture: "not_found.json", wantCode: "task_not_found", wantMsg: "expired", wantHTTP: 404},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, fixture(t, "submit.json"), tt.fixture))
				_, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithPollInterval(time.Second)).ImageModel("flux-dev"), ai.ImageRequest{Prompt: "p"})
				jobErr, isJob := errors.AsType[*ai.JobError](err)
				apiErr, isAPI := errors.AsType[*ai.APIError](err)
				if !isJob || !isAPI || jobErr.ID != taskID || apiErr.Code != tt.wantCode || apiErr.StatusCode != tt.wantHTTP || !strings.Contains(apiErr.Message, tt.wantMsg) || apiErr.Retryable() {
					t.Fatalf("GenerateImage() error = %v, want a non-retryable JobError with %s", err, tt.wantCode)
				}
				if n := len(srv.all()); n != 2 {
					t.Errorf("%d requests, want submit and one poll", n)
				}
			})
		})
	}
}

func TestImageModel_pollingURL(t *testing.T) {
	tests := []struct {
		name     string
		submit   string
		wantPoll string
		wantErr  string
	}{
		{name: "regional polling url", submit: `{"id":"t1","polling_url":"https://api.eu.bfl.ai/v1/get_result?id=t1"}`, wantPoll: "/v1/get_result"},
		{name: "missing url falls back to get_result", submit: `{"id":"t1"}`, wantPoll: "/v1/get_result"},
		{name: "foreign host is refused", submit: `{"id":"t1","polling_url":"https://evil.example/steal"}`, wantErr: "refusing to send the API key"},
		{name: "relative url is refused", submit: `{"id":"t1","polling_url":"/v1/get_result"}`, wantErr: "not an https url"},
		{name: "userinfo is refused", submit: `{"id":"t1","polling_url":"https://api.eu.bfl.ai@evil.example/steal"}`, wantErr: "not an https url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				srv := newFakeServer(t, routes(t, tt.submit, "ready.json"))
				_, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithPollInterval(time.Second)).ImageModel("flux-dev"), ai.ImageRequest{Prompt: "p"})
				reqs := srv.all()
				if tt.wantErr != "" {
					if err == nil || !strings.Contains(err.Error(), tt.wantErr) || len(reqs) != 1 {
						t.Errorf("GenerateImage() error = %v after %d requests, want %q before any poll", err, len(reqs), tt.wantErr)
					}
					return
				}
				if err != nil || reqs[1].Path != tt.wantPoll || reqs[1].Header.Get("X-Key") != "test-key" {
					t.Errorf("GenerateImage() error = %v, poll = %+v, want %s with the key", err, reqs[1], tt.wantPoll)
				}
			})
		})
	}
}

func TestImageModel_contextEndsWhilePolling(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newFakeServer(t, routes(t, fixture(t, "submit.json"), "pending.json"))
		ctx, stop := context.WithTimeout(t.Context(), 10*time.Second)
		defer stop()
		_, err := ai.GenerateImage(ctx, srv.provider(bfl.WithPollInterval(time.Second)).ImageModel("flux-dev"), ai.ImageRequest{Prompt: "p"})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != taskID || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("GenerateImage() error = %v, want a JobError with the task id and the deadline", err)
		}
	})
}

func TestImageModel_persistentPollErrorsDoNotResubmit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var submits atomic.Int32
		srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				submits.Add(1)
				jsonReply(http.StatusOK, fixture(t, "submit.json"))(w, r)
				return
			}
			jsonReply(http.StatusServiceUnavailable, `{"detail":"maintenance"}`)(w, r)
		})
		_, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithPollInterval(time.Second)).ImageModel("flux-dev"), ai.ImageRequest{Prompt: "p"})
		if _, ok := errors.AsType[*ai.JobError](err); !ok || submits.Load() != 1 {
			t.Errorf("GenerateImage() error = %v after %d submits, want a JobError and one submit", err, submits.Load())
		}
	})
}

func TestImageModel_pollingURLRequiresHTTPS(t *testing.T) {
	srv := newFakeServer(t, routes(t, `{"id":"t1","polling_url":"http://api.bfl.ai/v1/get_result?id=t1"}`, "ready.json"))
	_, err := ai.GenerateImage(t.Context(), srv.provider(bfl.WithBaseURL("https://api.bfl.ai")).ImageModel("flux-dev"), ai.ImageRequest{Prompt: "p"})
	if err == nil || !strings.Contains(err.Error(), "not an https url") || len(srv.all()) != 1 {
		t.Errorf("GenerateImage() error = %v after %d requests, want an https error before any poll", err, len(srv.all()))
	}
}
