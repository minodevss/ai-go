package google_test

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

const veoModelPath = "/v1/projects/my-proj/locations/us-central1/publishers/google/models/veo-3.1-generate-001"

func operationHandler(t *testing.T, polls ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if strings.HasSuffix(r.URL.Path, ":predictLongRunning") {
			jsonReply(fixture(t, "veo_started.json"))(w, r)
			return
		}
		body := polls[min(n, len(polls)-1)]
		n++
		jsonReply(body)(w, r)
	}
}

func operationServer(t *testing.T, polls ...string) *fakeServer {
	t.Helper()
	return newFakeServer(t, operationHandler(t, polls...))
}

func veoProviderAt(base string, client *http.Client) *google.Provider {
	return google.New(
		google.WithVertex("my-proj", "us-central1"),
		google.WithBaseURL(base+"/v1/projects/my-proj/locations/us-central1"),
		google.WithTokenSource(func(context.Context) (string, error) { return "tok", nil }),
		google.WithHTTPClient(client),
	)
}

func veoProvider(t *testing.T, srv *fakeServer) *google.Provider {
	t.Helper()
	return veoProviderAt(srv.URL, srv.Client())
}

func TestVideoModel_veoPollsWithBackoffUntilDone(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newBubbleServer(operationHandler(t, fixture(t, "veo_running.json"), fixture(t, "veo_running.json"), fixture(t, "veo_done_bytes.json")))
		m := veoProviderAt(bubbleURL, srv.client()).VideoModel("veo-3.1-generate-001")
		start := time.Now()
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{
			Prompt: "A close up of two people", Duration: 8 * time.Second, AspectRatio: "16:9", Seed: new(5),
			Provider: ai.Providers(google.VideoOptions{Resolution: "1080p", GenerateAudio: new(true), NegativePrompt: "blur", SampleCount: 1}),
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if elapsed := time.Since(start); elapsed != 12500*time.Millisecond {
			t.Errorf("elapsed = %s, want 12.5s of polling", elapsed)
		}
		if len(res.Videos) != 1 || string(res.Videos[0].Data) != "mp4-bytes" || res.Videos[0].MediaType != "video/mp4" {
			t.Errorf("Videos = %+v, want the decoded mp4", res.Videos)
		}
		srv.mu.Lock()
		defer srv.mu.Unlock()
		if got := srv.requests[0]; got.Path != veoModelPath+":predictLongRunning" {
			t.Errorf("start path = %q", got.Path)
		}
		equalJSON(t, `{"instances":[{"prompt":"A close up of two people"}],"parameters":{"aspectRatio":"16:9","durationSeconds":8,"seed":5,"sampleCount":1,"negativePrompt":"blur","resolution":"1080p","generateAudio":true}}`, srv.requests[0].Body)
		poll := srv.requests[1]
		if poll.Path != veoModelPath+":fetchPredictOperation" {
			t.Errorf("poll path = %q", poll.Path)
		}
		equalJSON(t, `{"operationName":"projects/my-proj/locations/us-central1/publishers/google/models/veo-3.1-generate-001/operations/a1b07c8e-7b5a-4aba-bb34-3e1ccb8afcc8"}`, poll.Body)
	})
}

func TestVideoModel_veoGCSResult(t *testing.T) {
	srv := operationServer(t, fixture(t, "veo_done_gcs.json"))
	res, err := ai.GenerateVideo(t.Context(), veoProvider(t, srv).VideoModel("veo-3.1-generate-001"), ai.VideoRequest{
		Prompt: "p", Provider: ai.Providers(google.VideoOptions{StorageURI: "gs://bucket/folder/"}),
	})
	if err != nil {
		t.Fatalf("GenerateVideo() error = %v", err)
	}
	if len(res.Videos) != 1 || res.Videos[0].URL != "gs://bucket/folder/sample_0.mp4" || len(res.Videos[0].Data) != 0 {
		t.Errorf("Videos = %+v, want the gs:// uri", res.Videos)
	}
	srv.mu.Lock()
	defer srv.mu.Unlock()
	equalJSON(t, `{"instances":[{"prompt":"p"}],"parameters":{"storageUri":"gs://bucket/folder/"}}`, srv.requests[0].Body)
}

func TestVideoModel_veoInputs(t *testing.T) {
	tests := []struct {
		name     string
		files    []ai.File
		opts     google.VideoOptions
		wantBody string
		wantErr  string
	}{
		{name: "first frame", files: []ai.File{{MediaType: "image/png", Data: []byte("a")}}, wantBody: `{"instances":[{"prompt":"p","image":{"mimeType":"image/png","bytesBase64Encoded":"YQ=="}}],"parameters":{}}`},
		{name: "first and last frame", files: []ai.File{{MediaType: "image/png", URL: "gs://b/a.png"}, {MediaType: "image/png", URL: "gs://b/z.png"}}, wantBody: `{"instances":[{"prompt":"p","image":{"mimeType":"image/png","gcsUri":"gs://b/a.png"},"lastFrame":{"mimeType":"image/png","gcsUri":"gs://b/z.png"}}],"parameters":{}}`},
		{name: "references", files: []ai.File{{MediaType: "image/png", Data: []byte("a")}}, opts: google.VideoOptions{ReferenceImages: true}, wantBody: `{"instances":[{"prompt":"p","referenceImages":[{"image":{"mimeType":"image/png","bytesBase64Encoded":"YQ=="},"referenceType":"asset"}]}],"parameters":{}}`},
		{name: "three frames", files: gsFiles(3), wantErr: "first and a last frame"},
		{name: "four references", files: gsFiles(4), opts: google.VideoOptions{ReferenceImages: true}, wantErr: "at most 3"},
		{name: "http url", files: []ai.File{{URL: "https://example.com/a.png"}}, wantErr: "gs:// URL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := operationServer(t, fixture(t, "veo_done_bytes.json"))
			_, err := ai.GenerateVideo(t.Context(), veoProvider(t, srv).VideoModel("veo-3.1-generate-001"), ai.VideoRequest{Prompt: "p", Files: tt.files, Provider: ai.Providers(tt.opts)})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) || srv.count() != 0 {
					t.Errorf("GenerateVideo() error = %v after %d requests, want %q before any request", err, srv.count(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("GenerateVideo() error = %v", err)
			}
			srv.mu.Lock()
			defer srv.mu.Unlock()
			equalJSON(t, tt.wantBody, srv.requests[0].Body)
		})
	}
}

func TestVideoModel_veoFailures(t *testing.T) {
	tests := []struct {
		name     string
		reply    string
		wantCode string
		wantText string
	}{
		{name: "operation error", reply: fixture(t, "veo_error.json"), wantCode: "INVALID_ARGUMENT", wantText: "could not be submitted"},
		{name: "filtered", reply: fixture(t, "veo_done_filtered.json"), wantCode: "no_video", wantText: "58061214"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := operationServer(t, tt.reply)
			_, err := ai.GenerateVideo(t.Context(), veoProvider(t, srv).VideoModel("veo-3.1-generate-001"), ai.VideoRequest{Prompt: "p"})
			jobErr, isJob := errors.AsType[*ai.JobError](err)
			apiErr, isAPI := errors.AsType[*ai.APIError](err)
			if !isJob || !isAPI || apiErr.Code != tt.wantCode || !strings.Contains(apiErr.Message, tt.wantText) || jobErr.ID == "" {
				t.Errorf("GenerateVideo() error = %v, want a JobError wrapping %s", err, tt.wantCode)
			}
			if isAPI && apiErr.Retryable() {
				t.Error("a failed operation must not be retryable")
			}
		})
	}
}

func omniHandler(t *testing.T, polls ...string) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if r.Method == http.MethodPost {
			jsonReply(fixture(t, "omni_started.json"))(w, r)
			return
		}
		body := polls[min(n, len(polls)-1)]
		n++
		jsonReply(body)(w, r)
	}
}

func omniServer(t *testing.T, polls ...string) *fakeServer {
	t.Helper()
	return newFakeServer(t, omniHandler(t, polls...))
}

func TestVideoModel_omniBackgroundInteraction(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srv := newBubbleServer(omniHandler(t, fixture(t, "omni_started.json"), fixture(t, "omni_completed.json")))
		p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(bubbleURL+"/v1beta"), google.WithHTTPClient(srv.client()))
		m := p.VideoModel("gemini-omni-1.2-flash")
		start := time.Now()
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{
			Prompt: "A marble rolling.", Duration: 10 * time.Second, AspectRatio: "9:16", Seed: new(3),
			Files: []ai.File{{MediaType: "image/png", Data: []byte("a")}},
		})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if elapsed := time.Since(start); elapsed != 5*time.Second {
			t.Errorf("elapsed = %s, want one 5s wait", elapsed)
		}
		if res.ID != "v1_abc123" || len(res.Videos) != 1 || string(res.Videos[0].Data) != "mp4-bytes" {
			t.Errorf("GenerateVideo() = %+v, want v1_abc123 with an mp4", res)
		}
		if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "seed") {
			t.Errorf("Warnings = %+v, want the seed warning", res.Warnings)
		}
		srv.mu.Lock()
		defer srv.mu.Unlock()
		create := srv.requests[0]
		if create.Path != "/v1beta/interactions" || create.Header.Get("Api-Revision") != "2026-05-20" || create.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("create = %s revision %q key %q", create.Path, create.Header.Get("Api-Revision"), create.Header.Get("x-goog-api-key"))
		}
		equalJSON(t, `{"model":"gemini-omni-1.2-flash","input":[{"type":"text","text":"A marble rolling."},{"type":"image","mime_type":"image/png","data":"YQ=="}],"background":true,"response_format":{"type":"video","aspect_ratio":"9:16","duration":"10s"}}`, create.Body)
		if got := srv.requests[1]; got.Path != "/v1beta/interactions/v1_abc123" || got.Header.Get("Api-Revision") != "2026-05-20" {
			t.Errorf("poll = %s revision %q", got.Path, got.Header.Get("Api-Revision"))
		}
	})
}

func TestVideoModel_omniFailure(t *testing.T) {
	srv := omniServer(t, fixture(t, "omni_failed.json"))
	_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("gemini-omni-1.2-flash"), ai.VideoRequest{Prompt: "p", PollInterval: time.Millisecond})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.Code != "safety" || !strings.Contains(apiErr.Message, "blocked") {
		t.Errorf("GenerateVideo() error = %v, want an APIError with code safety", err)
	}
}

func TestVideoModel_omniRejectsBadInput(t *testing.T) {
	tests := []struct {
		name string
		req  ai.VideoRequest
		want string
	}{
		{name: "too short", req: ai.VideoRequest{Prompt: "p", Duration: time.Second}, want: "3s to 15s"},
		{name: "too long", req: ai.VideoRequest{Prompt: "p", Duration: 20 * time.Second}, want: "3s to 15s"},
		{name: "unsupported aspect ratio", req: ai.VideoRequest{Prompt: "p", AspectRatio: "1:1"}, want: "16:9 and 9:16"},
		{name: "empty file", req: ai.VideoRequest{Prompt: "p", Files: []ai.File{{MediaType: "image/png"}}}, want: "no data or url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := omniServer(t, fixture(t, "omni_completed.json"))
			_, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("gemini-omni-1.2-flash"), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) || srv.count() != 0 {
				t.Errorf("GenerateVideo() error = %v after %d requests, want %q before any request", err, srv.count(), tt.want)
			}
		})
	}
}

func gsFiles(n int) []ai.File {
	files := make([]ai.File, n)
	for i := range files {
		files[i] = ai.File{MediaType: "image/png", URL: "gs://b/" + strconv.Itoa(i) + ".png"}
	}
	return files
}

func TestVideoModel_omniDownloadsURIsWithTheAPIKey(t *testing.T) {
	var srv *fakeServer
	srv = newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			jsonReply(fixture(t, "omni_started.json"))(w, r)
		case strings.HasPrefix(r.URL.Path, "/v1beta/files/"):
			if r.Header.Get("x-goog-api-key") != "test-key" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			_, _ = w.Write([]byte("big-mp4"))
		default:
			jsonReply(`{"id":"v1_abc123","status":"completed","steps":[{"type":"model_output","content":[{"type":"video","mime_type":"video/mp4","uri":"`+srv.URL+`/v1beta/files/abc:download?alt=media"},{"type":"video","uri":"https://elsewhere.example/v.mp4"}]}]}`)(w, r)
		}
	})
	res, err := ai.GenerateVideo(t.Context(), srv.provider().VideoModel("gemini-omni-1.2-flash"), ai.VideoRequest{Prompt: "p", PollInterval: time.Millisecond})
	if err != nil {
		t.Fatalf("GenerateVideo() error = %v", err)
	}
	if len(res.Videos) != 2 || string(res.Videos[0].Data) != "big-mp4" || res.Videos[0].MediaType != "video/mp4" {
		t.Fatalf("Videos = %+v, want the downloaded file first", res.Videos)
	}
	if res.Videos[1].URL != "https://elsewhere.example/v.mp4" || len(res.Videos[1].Data) != 0 {
		t.Errorf("Videos[1] = %+v, want a foreign host left as a URL without sending the key", res.Videos[1])
	}
}
