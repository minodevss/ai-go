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

func ExampleLanguageModel_StartBatch() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1beta/models/gemini-3-flash:batchGenerateContent":
			_, _ = io.WriteString(w, `{"name":"batches/123456"}`)
		default:
			_, _ = io.WriteString(w, `{"name":"batches/123456","done":true,"metadata":{"state":"JOB_STATE_SUCCEEDED"},"response":{"inlinedResponses":{"inlinedResponses":[{"metadata":{"key":"q1"},"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Paris."}]},"finishReason":"STOP"}]}}]}}}`)
		}
	}))
	defer srv.Close()

	model := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta")).Model("gemini-3-flash")
	job, err := ai.StartBatch(context.Background(), model, []ai.BatchItem{
		{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Capital of France?")}}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := job.Wait(context.Background(), time.Second); err != nil {
		fmt.Println(err)
		return
	}
	for r := range job.Results(context.Background()) {
		fmt.Println(r.ID, r.Result.Text)
	}
	// Output:
	// q1 Paris.
}

func ExampleProvider_UploadFile() {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/upload/v1beta/files":
			w.Header().Set("X-Goog-Upload-Url", srv.URL+"/session")
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"file":{"name":"files/abc-123","displayName":"notes.txt","mimeType":"text/plain","uri":"https://generativelanguage.googleapis.com/v1beta/files/abc-123","state":"ACTIVE"}}`)
		}
	}))
	defer srv.Close()

	p := google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta"))
	f, err := ai.UploadFile(context.Background(), p, ai.FileUpload{Name: "notes.txt", MediaType: "text/plain", Data: strings.NewReader("hello")})
	if err != nil {
		fmt.Println(err)
		return
	}
	id, ok := google.FileIDOf(f)
	fmt.Println(id, ok, f.MediaType)
	// Output:
	// abc-123 true text/plain
}

func filesExampleProvider(routes map[string]string) (*google.Provider, func()) {
	srv := mediaServer(routes)
	return google.New(google.WithAPIKey("test-key"), google.WithBaseURL(srv.URL+"/v1beta")), srv.Close
}

func ExampleProvider_GetFile() {
	p, stop := filesExampleProvider(map[string]string{"GET /v1beta/files/abc-123": `{"name":"files/abc-123","displayName":"notes.txt","mimeType":"text/plain","state":"ACTIVE"}`})
	defer stop()

	f, err := ai.GetFile(context.Background(), p, "abc-123")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(f.Filename, f.MediaType)
	// Output:
	// notes.txt text/plain
}

func ExampleProvider_ListFiles() {
	p, stop := filesExampleProvider(map[string]string{"GET /v1beta/files": `{"files":[{"name":"files/abc-123","displayName":"notes.txt","state":"ACTIVE"}]}`})
	defer stop()

	list, err := ai.ListFiles(context.Background(), p, ai.FileListRequest{})
	if err != nil {
		fmt.Println(err)
		return
	}
	id, _ := google.FileIDOf(list.Files[0])
	fmt.Println(id)
	// Output:
	// abc-123
}

func ExampleProvider_DeleteFile() {
	p, stop := filesExampleProvider(map[string]string{"DELETE /v1beta/files/abc-123": `{}`})
	defer stop()

	fmt.Println(ai.DeleteFile(context.Background(), p, "abc-123"))
	// Output:
	// <nil>
}
