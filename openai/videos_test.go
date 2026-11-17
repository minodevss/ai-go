package openai_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func videoServer(t *testing.T, polls ...string) (*fakeServer, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var queries []string
	n := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/videos":
			jsonReply(fixture(t, "video_queued.json"))(w, r)
		case strings.HasSuffix(r.URL.Path, "/content"):
			queries = append(queries, r.URL.RawQuery)
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("\x00\x00\x00\x18ftypmp42"))
		default:
			body := polls[min(n, len(polls)-1)]
			n++
			jsonReply(body)(w, r)
		}
	})
	return srv, &queries
}

func TestVideoModel_generate(t *testing.T) {
	srv, queries := videoServer(t, fixture(t, "video_queued.json"), fixture(t, "video_in_progress.json"), fixture(t, "video_completed.json"))
	m := srv.provider().VideoModel("sora-2")
	res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "A calico cat playing a piano on stage", Duration: 8 * time.Second, AspectRatio: "16:9", Seed: new(1), PollInterval: time.Millisecond})
	if err != nil {
		t.Fatalf("GenerateVideo() error = %v", err)
	}
	if res.ID != "video_123" || len(res.Videos) != 1 || res.Videos[0].MediaType != "video/mp4" || !strings.HasSuffix(string(res.Videos[0].Data), "ftypmp42") {
		t.Errorf("GenerateVideo() = %+v, want video_123 with an mp4", res)
	}
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "seed") {
		t.Errorf("warnings = %+v, want the seed warning", res.Warnings)
	}
	srv.mu.Lock()
	defer srv.mu.Unlock()
	create := srv.requests[0]
	if create.Path != "/v1/videos" || !strings.HasPrefix(create.Header.Get("Content-Type"), "application/json") {
		t.Fatalf("create = %s %q, want a JSON POST to /v1/videos", create.Path, create.Header.Get("Content-Type"))
	}
	equalJSON(t, `{"model":"sora-2","prompt":"A calico cat playing a piano on stage","seconds":"8","size":"1280x720"}`, create.Body)
	var paths []string
	for _, r := range srv.requests[1:] {
		paths = append(paths, r.Path)
	}
	want := []string{"/v1/videos/video_123", "/v1/videos/video_123", "/v1/videos/video_123/content"}
	if diff := cmp.Diff(want[:2], paths[:2]); diff != "" || paths[len(paths)-1] != want[2] {
		t.Errorf("paths = %v, want polls then the content download", paths)
	}
	if diff := cmp.Diff([]string{"variant=video"}, *queries); diff != "" {
		t.Errorf("content queries mismatch (-want +got):\n%s", diff)
	}
}

func TestVideoModel_failedJob(t *testing.T) {
	srv, _ := videoServer(t, fixture(t, "video_failed.json"))
	_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("sora-2"), ai.VideoRequest{Prompt: "p", PollInterval: time.Millisecond})
	jobErr, ok := errors.AsType[*ai.JobError](err)
	apiErr, isAPI := errors.AsType[*ai.APIError](err)
	if !ok || jobErr.ID != "video_123" || !isAPI || apiErr.Code != "moderation_blocked" || !strings.Contains(apiErr.Message, "moderation") {
		t.Errorf("GenerateVideo() error = %v, want a JobError wrapping moderation_blocked", err)
	}
	if apiErr.Retryable() {
		t.Error("a failed job must not be retryable")
	}
}

func TestVideoModel_options(t *testing.T) {
	tests := []struct {
		name     string
		call     ai.VideoRequest
		wantBody string
		wantWarn string
	}{
		{name: "options size wins", call: ai.VideoRequest{Prompt: "p", AspectRatio: "16:9", Provider: ai.Providers(openai.VideoOptions{Size: "1792x1024"})}, wantBody: `{"model":"sora-2-pro","prompt":"p","size":"1792x1024"}`},
		{name: "portrait ratio", call: ai.VideoRequest{Prompt: "p", AspectRatio: "9:16", Duration: 4 * time.Second}, wantBody: `{"model":"sora-2-pro","prompt":"p","seconds":"4","size":"720x1280"}`},
		{name: "seconds snap to an allowed value", call: ai.VideoRequest{Prompt: "p", Duration: 10 * time.Second}, wantBody: `{"model":"sora-2-pro","prompt":"p","seconds":"12"}`, wantWarn: "12 seconds"},
		{name: "short seconds snap up", call: ai.VideoRequest{Prompt: "p", Duration: time.Second}, wantBody: `{"model":"sora-2-pro","prompt":"p","seconds":"4"}`, wantWarn: "4 seconds"},
		{name: "unknown ratio warns", call: ai.VideoRequest{Prompt: "p", AspectRatio: "21:9"}, wantBody: `{"model":"sora-2-pro","prompt":"p"}`, wantWarn: "21:9"},
		{name: "reference url", call: ai.VideoRequest{Prompt: "p", Files: []ai.File{{URL: "https://example.com/a.png"}}}, wantBody: `{"model":"sora-2-pro","prompt":"p","input_reference":{"image_url":"https://example.com/a.png"}}`},
		{name: "reference file id", call: ai.VideoRequest{Prompt: "p", Files: []ai.File{{Provider: openai.FileID("file-9")}}}, wantBody: `{"model":"sora-2-pro","prompt":"p","input_reference":{"file_id":"file-9"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := videoServer(t, fixture(t, "video_completed.json"))
			tt.call.PollInterval = time.Millisecond
			res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("sora-2-pro"), tt.call)
			if err != nil {
				t.Fatalf("GenerateVideo() error = %v", err)
			}
			srv.mu.Lock()
			body := srv.requests[0].Body
			srv.mu.Unlock()
			equalJSON(t, tt.wantBody, body)
			if tt.wantWarn != "" && (len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, tt.wantWarn)) {
				t.Errorf("warnings = %+v, want one mentioning %q", res.Warnings, tt.wantWarn)
			}
		})
	}
}

func TestVideoModel_referenceUpload(t *testing.T) {
	srv, _ := videoServer(t, fixture(t, "video_completed.json"))
	_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("sora-2"), ai.VideoRequest{
		Prompt: "animate", Duration: 12 * time.Second, PollInterval: time.Millisecond,
		Files: []ai.File{{MediaType: "image/png", Data: []byte("png-bytes")}},
	})
	if err != nil {
		t.Fatalf("GenerateVideo() error = %v", err)
	}
	srv.mu.Lock()
	create := srv.requests[0]
	srv.mu.Unlock()
	parts := parseForm(t, create)
	if diff := cmp.Diff(map[string]string{"model": "sora-2", "prompt": "animate", "seconds": "12"}, formFields(parts)); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
	if files := formFiles(parts, "input_reference"); len(files) != 1 || string(files[0].Data) != "png-bytes" || files[0].Type != "image/png" {
		t.Errorf("input_reference parts = %+v, want the png", files)
	}
}

func TestVideoModel_rejectsBadReferences(t *testing.T) {
	srv, _ := videoServer(t, fixture(t, "video_completed.json"))
	m := srv.provider().VideoModel("sora-2")
	tests := []struct {
		name  string
		files []ai.File
		want  string
	}{
		{name: "two images", files: []ai.File{{URL: "https://a"}, {URL: "https://b"}}, want: "at most one"},
		{name: "empty file", files: []ai.File{{MediaType: "image/png"}}, want: "no data, url or file id"},
		{name: "huge file", files: []ai.File{{Data: make([]byte, 21<<20)}}, want: "exceeds"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "p", Files: tt.files})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateVideo() error = %v, want %q", err, tt.want)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("%d requests were sent for invalid references", srv.count())
	}
}

func TestVideoModel_contentDownloadFailure(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			jsonReply(fixture(t, "video_queued.json"))(w, r)
		case strings.HasSuffix(r.URL.Path, "/content"):
			w.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(w, `{"error":{"message":"expired","code":"not_found"}}`)
		default:
			jsonReply(fixture(t, "video_completed.json"))(w, r)
		}
	})
	_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("sora-2"), ai.VideoRequest{Prompt: "p", PollInterval: time.Millisecond})
	if jobErr, ok := errors.AsType[*ai.JobError](err); !ok || jobErr.ID != "video_123" {
		t.Errorf("GenerateVideo() error = %v, want a JobError so the id survives", err)
	}
}
