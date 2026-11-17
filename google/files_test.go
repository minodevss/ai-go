package google_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func filesServer(t *testing.T) *fakeServer {
	t.Helper()
	var srv *fakeServer
	srv = newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files":
			w.Header().Set("X-Goog-Upload-Url", srv.URL+"/upload-session")
		case r.URL.Path == "/upload-session":
			jsonReply(batchFixture(t, "upload.json"))(w, r)
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/files":
			jsonReply(batchFixture(t, "list.json"))(w, r)
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/files/vid-1":
			jsonReply(batchFixture(t, "get_active.json"))(w, r)
		case r.Method == http.MethodDelete && r.URL.Path == "/v1beta/files/abc-123":
			jsonReply(`{}`)(w, r)
		default:
			jsonErrorReply(http.StatusNotFound, `{"error":{"code":404,"message":"File not found","status":"NOT_FOUND"}}`)(w, r)
		}
	})
	return srv
}

func TestUploadFile_runsTheResumableProtocol(t *testing.T) {
	srv := filesServer(t)
	m := srv.provider().Model("gemini-3-flash")
	f, err := ai.UploadFile(t.Context(), m, ai.FileUpload{Name: "notes.txt", MediaType: "text/plain", Data: strings.NewReader("hello world")})
	if err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	srv.mu.Lock()
	start, finalize := srv.requests[0], srv.requests[1]
	srv.mu.Unlock()
	wantStart := map[string]string{
		"X-Goog-Upload-Protocol":              "resumable",
		"X-Goog-Upload-Command":               "start",
		"X-Goog-Upload-Header-Content-Length": "11",
		"X-Goog-Upload-Header-Content-Type":   "text/plain",
		"X-Goog-Api-Key":                      "test-key",
	}
	for k, v := range wantStart {
		if got := start.Header.Get(k); got != v {
			t.Errorf("start header %s = %q, want %q", k, got, v)
		}
	}
	equalJSON(t, `{"file":{"displayName":"notes.txt"}}`, start.Body)
	if got := finalize.Header.Get("X-Goog-Upload-Command"); got != "upload, finalize" {
		t.Errorf("finalize command = %q, want upload, finalize", got)
	}
	if got := finalize.Header.Get("X-Goog-Upload-Offset"); got != "0" {
		t.Errorf("finalize offset = %q, want 0", got)
	}
	if string(finalize.Body) != "hello world" {
		t.Errorf("finalize body = %q, want the file bytes", finalize.Body)
	}
	want := ai.File{Filename: "notes.txt", MediaType: "text/plain", URL: "https://generativelanguage.googleapis.com/v1beta/files/abc-123"}
	if diff := cmp.Diff(want, f, cmp.FilterPath(func(p cmp.Path) bool { return p.String() == "Provider" }, cmp.Ignore())); diff != "" {
		t.Errorf("UploadFile() mismatch (-want +got):\n%s", diff)
	}
	if got, ok := google.FileIDOf(f); got != "abc-123" || !ok {
		t.Errorf("FileIDOf() = %q, %v, want abc-123, true", got, ok)
	}
}

func TestUploadFile_uploadedFileIsReferencedByURI(t *testing.T) {
	srv := filesServer(t)
	p := srv.provider()
	f, err := ai.UploadFile(t.Context(), p.Model("gemini-3-flash"), ai.FileUpload{MediaType: "text/plain", Data: strings.NewReader("hi")})
	if err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	gen := newFakeServer(t, jsonReply(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	_, err = ai.GenerateText(t.Context(), gen.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.Text{Text: "read"}, f)}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	equalJSON(t, `[{"role":"user","parts":[{"text":"read"},{"fileData":{"mimeType":"text/plain","fileUri":"https://generativelanguage.googleapis.com/v1beta/files/abc-123"}}]}]`, contentsOf(t, gen.last(t).Body))
}

func TestUploadFile_rejectsALifetime(t *testing.T) {
	srv := filesServer(t)
	_, err := ai.UploadFile(t.Context(), srv.provider().Model("gemini-3-flash"), ai.FileUpload{Data: strings.NewReader("x"), ExpiresAfter: time.Hour})
	if err == nil || srv.count() != 0 {
		t.Errorf("UploadFile() error = %v after %d requests, want a rejection before any request", err, srv.count())
	}
}

func TestUploadFile_waitsWhileProcessing(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var gets int
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			switch r.URL.Path {
			case "/upload/v1beta/files":
				resp := stubResponse(`{}`)
				resp.Header.Set("X-Goog-Upload-Url", "https://example.test/upload-session")
				return resp, nil
			case "/upload-session":
				_, err := io.Copy(io.Discard, r.Body)
				return stubResponse(batchFixture(t, "upload_processing.json")), err
			default:
				gets++
				if gets < 3 {
					return stubResponse(strings.Replace(batchFixture(t, "get_active.json"), "ACTIVE", "PROCESSING", 1)), nil
				}
				return stubResponse(batchFixture(t, "get_active.json")), nil
			}
		})}
		p := google.New(google.WithAPIKey("k"), google.WithBaseURL("https://example.test/v1beta"), google.WithHTTPClient(client))
		start := time.Now()
		f, err := ai.UploadFile(t.Context(), p.Model("gemini-3-flash"), ai.FileUpload{MediaType: "video/mp4", Data: strings.NewReader("mp4")})
		if err != nil {
			t.Fatalf("UploadFile() error = %v", err)
		}
		if f.URL == "" || gets != 3 || time.Since(start) != 6*time.Second {
			t.Errorf("UploadFile() = %+v after %d polls over %s, want an active file after 3 polls over 6s", f, gets, time.Since(start))
		}
	})
}

func TestUploadFile_failedProcessingIsAnError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			switch r.URL.Path {
			case "/upload/v1beta/files":
				resp := stubResponse(`{}`)
				resp.Header.Set("X-Goog-Upload-Url", "https://example.test/upload-session")
				return resp, nil
			case "/upload-session":
				return stubResponse(batchFixture(t, "upload_processing.json")), nil
			default:
				return stubResponse(batchFixture(t, "get_failed.json")), nil
			}
		})}
		p := google.New(google.WithAPIKey("k"), google.WithBaseURL("https://example.test/v1beta"), google.WithHTTPClient(client))
		_, err := ai.UploadFile(t.Context(), p.Model("gemini-3-flash"), ai.FileUpload{MediaType: "video/mp4", Data: strings.NewReader("mp4")})
		if err == nil || !strings.Contains(err.Error(), "cannot decode video") {
			t.Errorf("UploadFile() error = %v, want the processing failure", err)
		}
	})
}

func TestGetFile(t *testing.T) {
	srv := filesServer(t)
	f, err := ai.GetFile(t.Context(), srv.provider().Model("gemini-3-flash"), "files/vid-1")
	if err != nil {
		t.Fatalf("GetFile() error = %v", err)
	}
	if f.Filename != "clip.mp4" || f.MediaType != "video/mp4" || fileIDOf(f) != "vid-1" {
		t.Errorf("GetFile() = %+v, want the clip's metadata", f)
	}
}

func TestListFiles_pages(t *testing.T) {
	srv := filesServer(t)
	list, err := ai.ListFiles(t.Context(), srv.provider(), ai.FileListRequest{Limit: 500, After: "tok1"})
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if len(list.Files) != 2 || list.Next != "tok2" {
		t.Errorf("ListFiles() = %d files, next %q, want 2 files and tok2", len(list.Files), list.Next)
	}
	if got := srv.last(t).Query; got != "pageSize=100&pageToken=tok1" {
		t.Errorf("list query = %q, want pageSize=100&pageToken=tok1", got)
	}
}

func TestDeleteFile(t *testing.T) {
	srv := filesServer(t)
	if err := ai.DeleteFile(t.Context(), srv.provider(), "abc-123"); err != nil {
		t.Fatalf("DeleteFile() error = %v", err)
	}
	if got := srv.last(t); got.Path != "/v1beta/files/abc-123" {
		t.Errorf("delete path = %q, want /v1beta/files/abc-123", got.Path)
	}
}

func TestDeleteFile_missingFileIsAnAPIError(t *testing.T) {
	srv := filesServer(t)
	err := ai.DeleteFile(t.Context(), srv.provider(), "files/nope")
	var apiErr *ai.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("DeleteFile() error = %v, want a 404 APIError", err)
	}
}

func TestGetFile_invalidIDIsRejected(t *testing.T) {
	srv := filesServer(t)
	for _, id := range []string{"", "../batches/1", "files/A_B", "files/x/y"} {
		if _, err := ai.GetFile(t.Context(), srv.provider(), id); err == nil {
			t.Errorf("GetFile(%q) error = nil, want an invalid id error", id)
		}
	}
	if srv.count() != 0 {
		t.Errorf("server received %d requests, want 0", srv.count())
	}
}

func TestFiles_vertexIsUnsupported(t *testing.T) {
	p := google.New(google.WithVertex("p", "us-central1"), google.WithTokenSource(func(context.Context) (string, error) { return "token", nil }))
	_, err := ai.GetFile(t.Context(), p, "files/abc")
	if !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("GetFile() error = %v, want ErrUnsupported", err)
	}
}

func fileIDOf(f ai.File) string {
	id, _ := google.FileIDOf(f)
	return id
}

func TestFileID(t *testing.T) {
	tests := []struct {
		name    string
		id      string
		wantID  string
		wantURL string
	}{
		{name: "bare id", id: "abc-123", wantID: "abc-123", wantURL: "https://generativelanguage.googleapis.com/v1beta/files/abc-123"},
		{name: "resource name", id: "files/abc-123", wantID: "abc-123", wantURL: "https://generativelanguage.googleapis.com/v1beta/files/abc-123"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := google.FileID(tt.id)
			gotID, ok := google.FileIDOf(f)
			if gotID != tt.wantID || !ok || f.URL != tt.wantURL {
				t.Errorf("FileID(%q) = id %q ok %v url %q, want %q %v %q", tt.id, gotID, ok, f.URL, tt.wantID, true, tt.wantURL)
			}
		})
	}
}

func TestFileIDOf_noReference(t *testing.T) {
	if id, ok := google.FileIDOf(ai.File{URL: "https://example.com/a.pdf"}); id != "" || ok {
		t.Errorf("FileIDOf() = %q, %v, want empty and false", id, ok)
	}
}
