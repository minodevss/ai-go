package bfl_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bfl"
)

func ExampleProvider_ImageModel() {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			fmt.Fprintf(w, `{"id":"task-1","polling_url":"%s/v1/get_result?id=task-1"}`, srv.URL)
		case strings.HasSuffix(r.URL.Path, "/get_result"):
			fmt.Fprintf(w, `{"id":"task-1","status":"Ready","result":{"sample":"%s/sample.png"}}`, srv.URL)
		default:
			w.Header().Set("Content-Type", "image/png")
			fmt.Fprint(w, "png-bytes")
		}
	}))
	defer srv.Close()
	p := bfl.New(bfl.WithAPIKey("bfl-test"), bfl.WithBaseURL(srv.URL), bfl.WithPollInterval(time.Millisecond))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("flux-kontext-pro"), ai.ImageRequest{
		Prompt: "make the sky purple",
		Files:  []ai.File{{URL: "https://example.com/photo.jpg"}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Images[0].MediaType, string(res.Images[0].Data))
	// Output: image/png png-bytes
}

func ExampleNew() {
	p := bfl.New(bfl.WithAPIKey("bfl-key"))
	fmt.Println(p.ImageModel("flux-pro-1.1").ModelID())
	// Output: flux-pro-1.1
}

func ExampleNewImageModel() {
	m := bfl.NewImageModel("flux-pro-1.1")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: bfl flux-pro-1.1
}
