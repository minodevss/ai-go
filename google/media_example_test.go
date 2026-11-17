package google_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func mediaServer(routes map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
}

func ExampleProvider_ImageModel() {
	srv := mediaServer(map[string]string{"POST /v1beta/models/gemini-nano-banana-2.1:generateContent": `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290}}`})
	defer srv.Close()
	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("gemini-nano-banana-2.1"), ai.ImageRequest{
		Prompt:      "a paper boat",
		AspectRatio: "16:9",
		Size:        "2K",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Images[0].Data), res.Images[0].MediaType, res.Usage.Output)
	// Output: hello image/png 1290
}

func ExampleProvider_VideoModel() {
	srv := mediaServer(map[string]string{
		"POST /v1beta/interactions":       `{"id":"v1_abc","status":"in_progress"}`,
		"GET /v1beta/interactions/v1_abc": `{"id":"v1_abc","status":"completed","steps":[{"type":"model_output","content":[{"type":"video","mime_type":"video/mp4","data":"bXA0"}]}]}`,
	})
	defer srv.Close()
	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))

	res, err := ai.GenerateVideo(context.Background(), p.VideoModel("gemini-omni-1.2-flash"), ai.VideoRequest{
		Prompt:       "a fox in snow",
		Duration:     5 * time.Second,
		PollInterval: time.Millisecond,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, string(res.Videos[0].Data))
	// Output: v1_abc mp4
}

func ExampleProvider_SpeechModel() {
	srv := mediaServer(map[string]string{"POST /v1beta/models/gemini-3.8-flash-tts:generateContent": `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/wav","data":"UklGRg=="}}]},"finishReason":"STOP"}]}`})
	defer srv.Close()
	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))

	res, err := ai.GenerateSpeech(context.Background(), p.SpeechModel("gemini-3.8-flash-tts"), ai.SpeechRequest{
		Text:         "Have a wonderful day!",
		Voice:        "Kore",
		Instructions: "cheerful and friendly",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(strings.TrimSpace(res.Audio.MediaType), len(res.Audio.Data))
	// Output: audio/wav 4
}
