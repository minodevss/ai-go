package fal_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/fal"
)

func exampleServer() *httptest.Server {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		base := srv.URL
		switch {
		case r.Method == http.MethodPost:
			job := base + strings.TrimSuffix(r.URL.Path, "/dev") + "/requests/req-1"
			fmt.Fprintf(w, `{"request_id":"req-1","response_url":"%[1]s","status_url":"%[1]s/status","cancel_url":"%[1]s/cancel"}`, job)
		case strings.HasSuffix(r.URL.Path, "/status"):
			_, _ = io.WriteString(w, `{"status":"COMPLETED"}`)
		case strings.HasPrefix(r.URL.Path, "/files/"):
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = io.WriteString(w, "media-bytes")
		case strings.Contains(r.URL.Path, "kling"):
			fmt.Fprintf(w, `{"video":{"url":"%s/files/v.mp4","content_type":"video/mp4"}}`, base)
		default:
			fmt.Fprintf(w, `{"images":[{"url":"%s/files/i.png","content_type":"image/png"}],"has_nsfw_concepts":[false]}`, base)
		}
	}))
	return srv
}

func ExampleProvider_ImageModel() {
	srv := exampleServer()
	defer srv.Close()
	p := fal.New(fal.WithAPIKey("key"), fal.WithBaseURL(srv.URL), fal.WithPollInterval(time.Millisecond))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("fal-ai/flux/dev"), ai.ImageRequest{
		Prompt:   "a paper boat",
		Size:     "1024x1024",
		Provider: ai.Providers(fal.ImageOptions{Extra: map[string]any{"guidance_scale": 3.5}}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Images[0].MediaType, string(res.Images[0].Data))
	// Output: image/png media-bytes
}

func ExampleProvider_VideoModel() {
	srv := exampleServer()
	defer srv.Close()
	p := fal.New(fal.WithAPIKey("key"), fal.WithBaseURL(srv.URL))

	res, err := ai.GenerateVideo(context.Background(), p.VideoModel("fal-ai/kling-video/v2.1/master/text-to-video"), ai.VideoRequest{
		Prompt:       "waves at dusk",
		Duration:     5 * time.Second,
		PollInterval: time.Millisecond,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, res.Videos[0].MediaType, string(res.Videos[0].Data))
	// Output: req-1 video/mp4 media-bytes
}

func ExampleNew() {
	p := fal.New(fal.WithAPIKey("fal-key"))
	fmt.Println(p.ImageModel("fal-ai/flux/dev").ModelID())
	// Output: fal-ai/flux/dev
}

func ExampleNewImageModel() {
	m := fal.NewImageModel("fal-ai/flux/dev")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: fal fal-ai/flux/dev
}

func ExampleNewVideoModel() {
	m := fal.NewVideoModel("fal-ai/kling-video/v2.1/master/text-to-video")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: fal fal-ai/kling-video/v2.1/master/text-to-video
}
