package luma_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/luma"
)

func exampleServer() *httptest.Server {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodDelete:
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/image"):
			fmt.Fprint(w, `{"id":"img-1","state":"queued"}`)
		case r.Method == http.MethodPost:
			fmt.Fprint(w, `{"id":"vid-1","state":"queued"}`)
		case path.Base(r.URL.Path) == "img-1":
			fmt.Fprintf(w, `{"id":"img-1","state":"completed","assets":{"image":"%s/a.jpg"}}`, srv.URL)
		case path.Base(r.URL.Path) == "vid-1":
			fmt.Fprintf(w, `{"id":"vid-1","state":"completed","assets":{"video":"%s/a.mp4"}}`, srv.URL)
		case strings.HasSuffix(r.URL.Path, ".jpg"):
			w.Header().Set("Content-Type", "image/jpeg")
			fmt.Fprint(w, "jpeg-bytes")
		default:
			w.Header().Set("Content-Type", "video/mp4")
			fmt.Fprint(w, "mp4-bytes")
		}
	}))
	return srv
}

func ExampleProvider_ImageModel() {
	srv := exampleServer()
	defer srv.Close()
	p := luma.New(luma.WithAPIKey("luma-test"), luma.WithBaseURL(srv.URL), luma.WithPollInterval(time.Millisecond))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("photon-1"), ai.ImageRequest{
		Prompt:      "A teddy bear in sunglasses",
		AspectRatio: "3:4",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Images[0].MediaType, string(res.Images[0].Data))
	// Output: image/jpeg jpeg-bytes
}

func ExampleProvider_VideoModel() {
	srv := exampleServer()
	defer srv.Close()
	p := luma.New(luma.WithAPIKey("luma-test"), luma.WithBaseURL(srv.URL))

	res, err := ai.GenerateVideo(context.Background(), p.VideoModel("ray-2"), ai.VideoRequest{
		Prompt:       "A tiger walking in snow",
		Files:        []ai.File{{URL: "https://example.com/tiger.jpg"}},
		PollInterval: time.Millisecond,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, res.Videos[0].MediaType)
	// Output: vid-1 video/mp4
}

func ExampleGenerationID() {
	file := ai.File{Provider: luma.GenerationID("123e4567-e89b-12d3-a456-426614174000")}
	fmt.Println(string(file.Provider["luma"]))
	// Output: {"id":"123e4567-e89b-12d3-a456-426614174000"}
}

func ExampleProvider_DeleteGeneration() {
	srv := exampleServer()
	defer srv.Close()
	p := luma.New(luma.WithAPIKey("luma-test"), luma.WithBaseURL(srv.URL))

	fmt.Println(p.DeleteGeneration(context.Background(), "vid-1"))
	// Output: <nil>
}

func ExampleNew() {
	p := luma.New(luma.WithAPIKey("luma-key"))
	fmt.Println(p.ImageModel("photon-1").ModelID())
	// Output: photon-1
}

func ExampleNewImageModel() {
	m := luma.NewImageModel("photon-1")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: luma photon-1
}

func ExampleNewVideoModel() {
	m := luma.NewVideoModel("ray-2")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: luma ray-2
}
