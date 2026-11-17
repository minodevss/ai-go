package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
	"github.com/minodevss/ai-go/realtime"
)

func mediaServer(routes map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = io.WriteString(w, body)
	}))
}

func ExampleProvider_ImageModel() {
	srv := mediaServer(map[string]string{"POST /v1/images/generations": `{"created":1,"data":[{"b64_json":"aGVsbG8="}],"usage":{"input_tokens":5,"output_tokens":7}}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateImage(context.Background(), p.ImageModel("gpt-image-2"), ai.ImageRequest{
		Prompt:   "a paper boat",
		Size:     "1024x1024",
		Provider: ai.Providers(openai.ImageOptions{Quality: "low", OutputFormat: "png"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Images[0].Data), res.Images[0].MediaType, res.Usage.Output)
	// Output: hello image/png 7
}

func ExampleProvider_VideoModel() {
	srv := mediaServer(map[string]string{
		"POST /v1/videos":                `{"id":"video_1","status":"queued","progress":0}`,
		"GET /v1/videos/video_1":         `{"id":"video_1","status":"completed","progress":100}`,
		"GET /v1/videos/video_1/content": "mp4-bytes",
	})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateVideo(context.Background(), p.VideoModel("sora-2"), ai.VideoRequest{Prompt: "a fox in snow", Duration: 4 * time.Second, PollInterval: time.Millisecond})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, string(res.Videos[0].Data))
	// Output: video_1 mp4-bytes
}

func ExampleProvider_SpeechModel() {
	srv := mediaServer(map[string]string{"POST /v1/audio/speech": "audio-bytes"})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	text := func(yield func(string) bool) {
		for _, piece := range []string{"The first sentence is long enough. ", "And a second one follows it."} {
			if !yield(piece) {
				return
			}
		}
	}
	s := ai.StreamSpeech(context.Background(), p.SpeechModel("gpt-4o-mini-tts"), ai.SpeechStreamRequest{Text: text, Voice: "marin", Format: "mp3"})
	chunks := 0
	for range s.Audio() {
		chunks++
	}
	fmt.Println(chunks, s.Err())
	// Output: 2 <nil>
}

func ExampleProvider_TranscriptionModel() {
	srv := mediaServer(map[string]string{"POST /v1/audio/transcriptions": `{"text":"Hello world."}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.Transcribe(context.Background(), p.TranscriptionModel("gpt-4o-transcribe"), ai.TranscribeRequest{Audio: strings.NewReader("audio"), MediaType: "audio/mpeg", Language: "en"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output: Hello world.
}

func ExampleFileIDOf() {
	srv := mediaServer(map[string]string{"POST /v1/files": `{"id":"file-abc123","object":"file","filename":"notes.txt","purpose":"user_data"}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	f, err := ai.UploadFile(context.Background(), p, ai.FileUpload{Name: "notes.txt", Data: strings.NewReader("hello")})
	if err != nil {
		fmt.Println(err)
		return
	}
	id, ok := openai.FileIDOf(f)
	fmt.Println(id, ok)
	// Output: file-abc123 true
}

func ExampleLanguageModel_StartBatch() {
	srv := mediaServer(map[string]string{
		"POST /v1/files":                 `{"id":"file-in","object":"file","filename":"batch.jsonl","purpose":"batch"}`,
		"POST /v1/batches":               `{"id":"batch_1","object":"batch","status":"validating"}`,
		"GET /v1/batches/batch_1":        `{"id":"batch_1","object":"batch","status":"completed","output_file_id":"file-out","request_counts":{"total":1,"completed":1,"failed":0}}`,
		"GET /v1/files/file-out/content": `{"custom_id":"q1","response":{"status_code":200,"body":{"id":"resp_1","status":"completed","output":[{"id":"m","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"blue","annotations":[]}]}]}},"error":null}` + "\n",
	})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	ctx := context.Background()
	items := []ai.BatchItem{{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Reply with blue.")}}}}
	job, err := ai.StartBatch(ctx, p.Model("gpt-5-mini"), items)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := job.Wait(ctx, time.Millisecond); err != nil {
		fmt.Println(err)
		return
	}
	for r := range job.Results(ctx) {
		fmt.Println(r.ID, r.Result.Text)
	}
	// Output: q1 blue
}

func ExampleRealtimeModel_MintClientSecret() {
	srv := mediaServer(map[string]string{"POST /v1/realtime/client_secrets": `{"value":"ek_test","expires_at":1756310470,"session":{"type":"realtime"}}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	secret, err := realtime.MintClientSecret(context.Background(), p.RealtimeModel("gpt-realtime-2.1"), realtime.Config{Instructions: "Greet the visitor.", Voice: "marin"}, realtime.MintOptions{TTL: 2 * time.Minute})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(secret.Value, secret.Locked)
	// Output: ek_test false
}

func ExampleNewImageModel() {
	srv := mediaServer(map[string]string{"POST /v1/images/generations": `{"created":1,"data":[{"b64_json":"aGVsbG8="}]}`})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.GenerateImage(context.Background(), openai.NewImageModel("gpt-image-2"), ai.ImageRequest{Prompt: "a paper boat"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Images[0].Data))
	// Output: hello
}

func ExampleNewVideoModel() {
	srv := mediaServer(map[string]string{
		"POST /v1/videos":                `{"id":"video_1","status":"queued","progress":0}`,
		"GET /v1/videos/video_1":         `{"id":"video_1","status":"completed","progress":100}`,
		"GET /v1/videos/video_1/content": "mp4-bytes",
	})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.GenerateVideo(context.Background(), openai.NewVideoModel("sora-2"), ai.VideoRequest{Prompt: "a fox in snow", PollInterval: time.Millisecond})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Videos[0].Data))
	// Output: mp4-bytes
}

func ExampleNewSpeechModel() {
	srv := mediaServer(map[string]string{"POST /v1/audio/speech": "audio-bytes"})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.GenerateSpeech(context.Background(), openai.NewSpeechModel("gpt-4o-mini-tts"), ai.SpeechRequest{Text: "Hello.", Voice: "marin"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Audio.Data))
	// Output: audio-bytes
}

func ExampleNewTranscriptionModel() {
	srv := mediaServer(map[string]string{"POST /v1/audio/transcriptions": `{"text":"Hello world."}`})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.Transcribe(context.Background(), openai.NewTranscriptionModel("gpt-4o-transcribe"), ai.TranscribeRequest{Audio: strings.NewReader("audio"), MediaType: "audio/mpeg"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output: Hello world.
}

func ExampleProvider_RealtimeModel() {
	srv := mediaServer(map[string]string{"POST /v1/realtime/client_secrets": `{"value":"ek_test","expires_at":1756310470,"session":{"type":"realtime"}}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	secret, err := realtime.MintClientSecret(context.Background(), p.RealtimeModel("gpt-realtime-2.1"), realtime.Config{Voice: "marin"}, realtime.MintOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(secret.Value)
	// Output: ek_test
}

func ExampleNewRealtimeModel() {
	srv := mediaServer(map[string]string{"POST /v1/realtime/client_secrets": `{"value":"ek_test","expires_at":1756310470,"session":{"type":"realtime"}}`})
	defer srv.Close()
	defer useServer(srv)()

	secret, err := realtime.MintClientSecret(context.Background(), openai.NewRealtimeModel("gpt-realtime-2.1"), realtime.Config{Voice: "marin"}, realtime.MintOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(secret.Value)
	// Output: ek_test
}
