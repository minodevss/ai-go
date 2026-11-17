package anthropic_test

import (
	"bytes"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func filesServer(t *testing.T) *fakeServer {
	t.Helper()
	return newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/files":
			jsonReply(fixture(t, "files/list.json"))(w, r)
		case r.Method == http.MethodDelete:
			jsonReply(fixture(t, "files/deleted.json"))(w, r)
		default:
			jsonReply(fixture(t, "files/file.json"))(w, r)
		}
	})
}

func formOf(t *testing.T, c captured) (fields map[string]string, files map[string]string) {
	t.Helper()
	_, params, err := mime.ParseMediaType(c.Header.Get("Content-Type"))
	if err != nil {
		t.Fatalf("parse content type: %v", err)
	}
	r := multipart.NewReader(bytes.NewReader(c.Body), params["boundary"])
	fields, files = map[string]string{}, map[string]string{}
	for {
		part, err := r.NextPart()
		if errors.Is(err, io.EOF) {
			return fields, files
		}
		if err != nil {
			t.Fatalf("read part: %v", err)
		}
		data, err := io.ReadAll(part)
		if err != nil {
			t.Fatalf("read part body: %v", err)
		}
		if part.FileName() != "" {
			files[part.FormName()+":"+part.FileName()+":"+part.Header.Get("Content-Type")] = string(data)
			continue
		}
		fields[part.FormName()] = string(data)
	}
}

func TestFiles_upload(t *testing.T) {
	srv := filesServer(t)
	f, err := ai.UploadFile(t.Context(), srv.provider(), ai.FileUpload{Name: "document.pdf", MediaType: "application/pdf", Data: strings.NewReader("%PDF"), ExpiresAfter: 2 * time.Hour})
	if err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	if fileID(f) != "file_011CNha8iCJcU1wXNR6q4V8w" || f.Filename != "document.pdf" || f.MediaType != "application/pdf" || len(f.Data) != 0 {
		t.Errorf("UploadFile() = %+v, want the file id in Provider and no data", f)
	}
	last := srv.last(t)
	if last.Path != "/v1/files" {
		t.Errorf("path = %s, want /v1/files", last.Path)
	}
	if got := last.Header.Get("Anthropic-Beta"); got != "" {
		t.Errorf("anthropic-beta = %q, want none because the Files API is out of beta", got)
	}
	fields, files := formOf(t, last)
	if diff := cmp.Diff(map[string]string{"expires_in_seconds": "7200"}, fields); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]string{"file:document.pdf:application/pdf": "%PDF"}, files); diff != "" {
		t.Errorf("files mismatch (-want +got):\n%s", diff)
	}
}

func TestFiles_uploadRejectsOutOfRangeExpiry(t *testing.T) {
	for _, d := range []time.Duration{time.Minute, 91 * 24 * time.Hour} {
		srv := filesServer(t)
		_, err := ai.UploadFile(t.Context(), srv.provider(), ai.FileUpload{Data: strings.NewReader("x"), ExpiresAfter: d})
		if err == nil || !strings.Contains(err.Error(), "outside") || srv.count() != 0 {
			t.Errorf("UploadFile(ExpiresAfter %s) error = %v with %d requests, want a local range error", d, err, srv.count())
		}
	}
}

func TestFiles_getListDelete(t *testing.T) {
	srv := filesServer(t)
	p := srv.provider()
	got, err := ai.GetFile(t.Context(), p, "file_011CNha8iCJcU1wXNR6q4V8w")
	if err != nil || fileID(got) != "file_011CNha8iCJcU1wXNR6q4V8w" || got.MediaType != "application/pdf" {
		t.Fatalf("GetFile() = %+v, %v, want the file", got, err)
	}
	list, err := ai.ListFiles(t.Context(), p, ai.FileListRequest{Limit: 10, After: "page_prev"})
	if err != nil || len(list.Files) != 1 || list.Next != "page_abc" {
		t.Fatalf("ListFiles() = %+v, %v, want one file and next page_abc", list, err)
	}
	if q := srv.last(t).Path; q != "/v1/files" {
		t.Errorf("list path = %s, want /v1/files", q)
	}
	if err := ai.DeleteFile(t.Context(), p, "file_011CNha8iCJcU1wXNR6q4V8w"); err != nil {
		t.Errorf("DeleteFile() error = %v", err)
	}
	if last := srv.last(t); last.Path != "/v1/files/file_011CNha8iCJcU1wXNR6q4V8w" {
		t.Errorf("delete path = %s", last.Path)
	}
}

func TestFiles_listQueryAndLastPage(t *testing.T) {
	var query string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.RawQuery
		jsonReply(`{"data":[],"next_page":null}`)(w, r)
	})
	list, err := ai.ListFiles(t.Context(), srv.provider(), ai.FileListRequest{Limit: 10, After: "page_prev"})
	if err != nil || list.Next != "" || len(list.Files) != 0 {
		t.Fatalf("ListFiles() = %+v, %v, want an empty last page", list, err)
	}
	if query != "limit=10&page=page_prev" {
		t.Errorf("query = %q, want limit=10&page=page_prev", query)
	}
	if _, err := ai.ListFiles(t.Context(), srv.provider(), ai.FileListRequest{Limit: 1001}); err == nil {
		t.Error("ListFiles(Limit 1001) error = nil, want a range error")
	}
}

func TestFiles_idsFlowIntoDocumentAndImageBlocks(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	doc := ai.File{MediaType: "application/pdf", Filename: "report.pdf", Provider: anthropic.FileID("file_doc")}
	img := ai.File{MediaType: "image/png", Provider: anthropic.FileID("file_img")}
	generate(t, srv, &ai.Call{Messages: []ai.Message{ai.UserParts(ai.Text{Text: "see"}, doc, img)}})
	equalJSON(t, `[{"role":"user","content":[
		{"type":"text","text":"see"},
		{"type":"document","title":"report.pdf","source":{"type":"file","file_id":"file_doc"}},
		{"type":"image","source":{"type":"file","file_id":"file_img"}}]}]`, []byte(field(t, srv, "messages")))
}

func TestFiles_errorsKeepTheAPIError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"request_too_large","message":"File too large"}}`)
	})
	_, err := ai.UploadFile(t.Context(), srv.provider(), ai.FileUpload{Data: strings.NewReader("x"), MaxRetries: new(0)})
	if err == nil || !strings.Contains(err.Error(), "File too large") {
		t.Errorf("UploadFile() error = %v, want the API message", err)
	}
}

func fileID(f ai.File) string {
	id, _ := anthropic.FileIDOf(f)
	return id
}

func TestFileIDOf(t *testing.T) {
	tests := []struct {
		name   string
		file   ai.File
		wantID string
		wantOK bool
	}{
		{name: "from FileID", file: ai.File{Provider: anthropic.FileID("file_1")}, wantID: "file_1", wantOK: true},
		{name: "plain file", file: ai.File{Data: []byte{1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID, gotOK := anthropic.FileIDOf(tt.file)
			if gotID != tt.wantID || gotOK != tt.wantOK {
				t.Errorf("FileIDOf() = %q, %v, want %q, %v", gotID, gotOK, tt.wantID, tt.wantOK)
			}
		})
	}
}
