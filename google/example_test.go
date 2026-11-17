package google_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func exampleServer(responses map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, responses[r.URL.Path])
	}))
}

const exampleReply = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello from Gemini."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"thoughtsTokenCount":4,"candidatesTokenCount":6},"responseId":"resp-1"}`

func ExampleNew() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": exampleReply})
	defer srv.Close()

	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))
	res, err := ai.GenerateText(context.Background(), p.Model("gemini-3-flash"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.Output, res.Usage.Reasoning)
	// Output:
	// Hello from Gemini.
	// 9 10 4
}

func ExampleModel() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": exampleReply})
	defer srv.Close()
	if err := os.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL+"/v1beta"); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("GOOGLE_GEMINI_BASE_URL")

	res, err := ai.GenerateText(context.Background(), google.Model("gemini-3-flash"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Gemini.
}

func ExampleOptions() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": exampleReply})
	defer srv.Close()
	model := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta")).Model("gemini-3-flash")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages:  []ai.Message{ai.User("Think, then answer.")},
		Reasoning: ai.ReasoningLow,
		Provider: ai.Providers(google.Options{
			SafetySettings: []google.SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH"}},
			Labels:         map[string]string{"team": "support"},
		}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, _ := ai.ProviderValue[google.ResponseData](res.Provider, "google")
	fmt.Println(data.ResponseID)
	// Output:
	// resp-1
}

func ExampleWithVertex() {
	srv := exampleServer(map[string]string{"/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3-flash:generateContent": exampleReply})
	defer srv.Close()

	p := google.New(
		google.WithVertex("my-project", "us-central1"),
		google.WithBaseURL(srv.URL+"/v1/projects/my-project/locations/us-central1"),
		google.WithTokenSource(func(context.Context) (string, error) { return "ya29.example", nil }),
	)
	res, err := ai.GenerateText(context.Background(), p.Model("gemini-3-flash"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Gemini.
}

func ExampleSearch() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": `{"candidates":[{"content":{"role":"model","parts":[{"text":"Go 1.26 shipped in February."}]},"finishReason":"STOP",
		"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes"}}],
		"groundingSupports":[{"segment":{"startIndex":0,"endIndex":28,"text":"Go 1.26 shipped in February."},"groundingChunkIndices":[0]}]}}]}`})
	defer srv.Close()
	model := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta")).Model("gemini-3-flash")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("When did Go 1.26 ship?")},
		Tools:    []ai.Tool{google.Search()},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range res.Content {
		if t, ok := p.(ai.Text); ok {
			fmt.Println(t.Text, "-", t.Citations[0].URL)
		}
	}
	// Output:
	// Go 1.26 shipped in February. - https://go.dev/doc/go1.26
}

func ExampleCodeExecution() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": `{"candidates":[{"content":{"role":"model","parts":[
		{"executableCode":{"language":"PYTHON","code":"print(17*23)"}},
		{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"391\n"}},
		{"text":"It is 391."}]},"finishReason":"STOP"}]}`})
	defer srv.Close()
	model := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta")).Model("gemini-3-flash")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("What is 17*23?")},
		Tools:    []ai.Tool{google.CodeExecution()},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range res.Content {
		switch p := p.(type) {
		case ai.ToolCall:
			fmt.Println("ran:", p.Name, p.ProviderExecuted)
		case ai.Text:
			fmt.Println(p.Text)
		}
	}
	// Output:
	// ran: code_execution true
	// It is 391.
}

func ExampleURLContext() {
	tool := google.URLContext()
	fmt.Println(tool.Name)
	// Output:
	// url_context
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-embedding-001:batchEmbedContents": `{"embeddings":[{"values":[0.25,0.5]},{"values":[0.75,1]}]}`})
	defer srv.Close()
	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))

	res, err := ai.Embed(context.Background(), p.EmbeddingModel("gemini-embedding-001"), ai.EmbedRequest{
		Texts:     []string{"first", "second"},
		InputType: ai.InputDocument,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Embeddings[1])
	// Output:
	// 2 [0.75 1]
}

func ExampleServiceAccountTokenSource() {
	_, err := google.ServiceAccountTokenSource([]byte(`{"type":"authorized_user"}`), nil)
	fmt.Println(err)
	// Output:
	// google: credentials of type "authorized_user" are not supported, want service_account
}

func ExampleDefaultTokenSource() {
	ts := google.DefaultTokenSource(nil)
	p := google.New(google.WithVertex("my-project", "global"), google.WithTokenSource(ts))
	fmt.Println(p.Model("gemini-3-flash").Provider())
	// Output:
	// google-vertex
}

func ExampleMetadataTokenSource() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"access_token":"ya29.meta","expires_in":3600,"token_type":"Bearer"}`)
	}))
	defer srv.Close()
	if err := os.Setenv("GCE_METADATA_HOST", srv.Listener.Addr().String()); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("GCE_METADATA_HOST")

	token, err := google.MetadataTokenSource(nil)(context.Background())
	fmt.Println(token, err)
	// Output:
	// ya29.meta <nil>
}

func headerServer(header string, got *string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*got = r.Header.Get(header)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleReply)
	}))
}

func generateHello(p *google.Provider) error {
	_, err := ai.GenerateText(context.Background(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	return err
}

func ExampleWithAPIKey() {
	var key string
	srv := headerServer("x-goog-api-key", &key)
	defer srv.Close()

	if err := generateHello(google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL))); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(key)
	// Output:
	// test-key
}

func ExampleWithBaseURL() {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleReply)
	}))
	defer srv.Close()

	if err := generateHello(google.New(google.WithBaseURL(srv.URL + "/proxy"))); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(path)
	// Output:
	// /proxy/models/gemini-3-flash:generateContent
}

func ExampleWithHTTPClient() {
	var used bool
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		used = true
		rec := httptest.NewRecorder()
		rec.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(rec, exampleReply)
		return rec.Result(), nil
	})}

	if err := generateHello(google.New(google.WithHTTPClient(client))); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(used)
	// Output:
	// true
}

func ExampleWithHeaders() {
	var team string
	srv := headerServer("X-Team", &team)
	defer srv.Close()

	if err := generateHello(google.New(google.WithBaseURL(srv.URL), google.WithHeaders(http.Header{"X-Team": {"support"}}))); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(team)
	// Output:
	// support
}

func ExampleWithLimiter() {
	var key string
	srv := headerServer("x-goog-api-key", &key)
	defer srv.Close()
	limiter := &countingLimiter{}

	if err := generateHello(google.New(google.WithBaseURL(srv.URL), google.WithLimiter(limiter))); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(limiter.n)
	// Output:
	// 1
}

func ExampleWithTokenSource() {
	var auth string
	srv := headerServer("Authorization", &auth)
	defer srv.Close()

	p := google.New(
		google.WithVertex("my-project", "global"),
		google.WithBaseURL(srv.URL),
		google.WithTokenSource(func(context.Context) (string, error) { return "ya29.example", nil }),
	)
	if err := generateHello(p); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(auth)
	// Output:
	// Bearer ya29.example
}

func ExampleProvider_Model() {
	m := google.New(google.WithAPIKey("test-key")).Model("gemini-3-flash")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// google gemini-3-flash
}

func ExampleProvider_LanguageModel() {
	m, err := google.New(google.WithAPIKey("test-key")).LanguageModel("gemini-3-flash")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.ModelID())
	// Output:
	// gemini-3-flash
}

func ExampleNewEmbeddingModel() {
	m := google.NewEmbeddingModel("gemini-embedding-001")
	fmt.Println(m.ModelID(), m.MaxBatch())
	// Output:
	// gemini-embedding-001 100
}

func ExampleNewImageModel() {
	fmt.Println(google.NewImageModel("gemini-nano-banana-2.1").ModelID())
	// Output:
	// gemini-nano-banana-2.1
}

func ExampleNewVideoModel() {
	fmt.Println(google.NewVideoModel("gemini-omni-1.2-flash").ModelID())
	// Output:
	// gemini-omni-1.2-flash
}

func ExampleNewSpeechModel() {
	fmt.Println(google.NewSpeechModel("gemini-3.8-flash-tts").ModelID())
	// Output:
	// gemini-3.8-flash-tts
}

func ExampleNewRealtimeModel() {
	fmt.Println(google.NewRealtimeModel("gemini-3.8-live").ModelID())
	// Output:
	// gemini-3.8-live
}

func ExampleMaps() {
	fmt.Println(google.Maps().Name)
	// Output:
	// google_maps
}

func ExampleFileSearch() {
	fmt.Println(google.FileSearch("fileSearchStores/docs-123").Name)
	// Output:
	// file_search
}

func ExampleFileID() {
	f := google.FileID("abc-123")
	fmt.Println(f.URL)
	// Output:
	// https://generativelanguage.googleapis.com/v1beta/files/abc-123
}

func ExampleFileIDOf() {
	id, ok := google.FileIDOf(google.FileID("files/abc-123"))
	fmt.Println(id, ok)
	_, ok = google.FileIDOf(ai.File{URL: "https://example.com/a.pdf"})
	fmt.Println(ok)
	// Output:
	// abc-123 true
	// false
}

func ExampleResponseData() {
	srv := exampleServer(map[string]string{"/v1beta/models/gemini-3-flash:generateContent": exampleReply})
	defer srv.Close()
	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))

	res, err := ai.GenerateText(context.Background(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, ok := ai.ProviderValue[google.ResponseData](res.Provider, "google")
	fmt.Println(data.ResponseID, ok)
	// Output:
	// resp-1 true
}

func ExampleImageOptions() {
	p := ai.Providers(google.ImageOptions{SafetySettings: []google.SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH"}}})
	_, ok := p["google"]
	fmt.Println(ok)
	// Output:
	// true
}

func ExampleSpeechOptions() {
	p := ai.Providers(google.SpeechOptions{SampleRate: 24000})
	_, ok := p["google"]
	fmt.Println(ok)
	// Output:
	// true
}

func ExampleVideoOptions() {
	p := ai.Providers(google.VideoOptions{ReferenceImages: true})
	_, ok := p["google"]
	fmt.Println(ok)
	// Output:
	// true
}

func ExampleRealtimeOptions() {
	p := ai.Providers(google.RealtimeOptions{ResponseModalities: []string{"TEXT"}})
	_, ok := p["google"]
	fmt.Println(ok)
	// Output:
	// true
}
