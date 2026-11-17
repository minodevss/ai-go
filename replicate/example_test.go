package replicate_test

import (
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/replicate"
)

func exampleServer() *httptest.Server {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "video"):
			fmt.Fprint(w, `{"id":"v1","status":"starting"}`)
		case r.Method == http.MethodPost:
			fmt.Fprintf(w, `{"id":"p1","status":"succeeded","output":["%s/out.webp"]}`, srv.URL)
		case r.URL.Path == "/v1/predictions/v1":
			fmt.Fprintf(w, `{"id":"v1","status":"succeeded","output":"%s/out.mp4"}`, srv.URL)
		default:
			w.Header().Set("Content-Type", mime.TypeByExtension(path.Ext(r.URL.Path)))
			_, _ = io.WriteString(w, "file-bytes")
		}
	}))
	return srv
}

func ExampleProvider_ImageModel() {
	srv := exampleServer()
	defer srv.Close()
	p := replicate.New(replicate.WithAPIKey("r8_test"), replicate.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("black-forest-labs/flux-schnell"), ai.ImageRequest{
		Prompt:   "a paper boat",
		Provider: ai.Providers(replicate.ImageOptions{Extra: map[string]any{"output_format": "webp"}}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Images[0].MediaType, string(res.Images[0].Data))
	// Output: image/webp file-bytes
}

func ExampleProvider_VideoModel() {
	srv := exampleServer()
	defer srv.Close()
	p := replicate.New(replicate.WithAPIKey("r8_test"), replicate.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateVideo(context.Background(), p.VideoModel("minimax/video-01"), ai.VideoRequest{
		Prompt:       "a cat on a piano",
		Duration:     6 * time.Second,
		PollInterval: time.Millisecond,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, res.Videos[0].MediaType)
	// Output: v1 video/mp4
}

func ExampleProvider_CancelPrediction() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"id":"v1","status":"canceled"}`)
	}))
	defer srv.Close()
	p := replicate.New(replicate.WithAPIKey("r8_test"), replicate.WithBaseURL(srv.URL+"/v1"))

	fmt.Println(p.CancelPrediction(context.Background(), "v1"))
	// Output: <nil>
}

func ExampleNew() {
	p := replicate.New(replicate.WithAPIKey("r8-key"))
	fmt.Println(p.ImageModel("black-forest-labs/flux-schnell").ModelID())
	// Output: black-forest-labs/flux-schnell
}

func ExampleNewImageModel() {
	m := replicate.NewImageModel("black-forest-labs/flux-schnell")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: replicate black-forest-labs/flux-schnell
}

func ExampleNewVideoModel() {
	m := replicate.NewVideoModel("minimax/video-01")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: replicate minimax/video-01
}
