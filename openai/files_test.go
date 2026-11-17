package openai_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func fileServer(t *testing.T) *fakeServer {
	t.Helper()
	return newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/files":
			jsonReply(fixture(t, "file_object.json"))(w, r)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/files":
			jsonReply(`{"object":"list","data":[{"id":"file-1","object":"file","bytes":1,"created_at":1,"filename":"a.txt","purpose":"user_data","status":"processed"},{"id":"file-2","object":"file","bytes":1,"created_at":2,"filename":"b.txt","purpose":"user_data","status":"processed"}],"first_id":"file-1","last_id":"file-2","has_more":true}`)(w, r)
		case r.Method == http.MethodDelete:
			jsonReply(`{"id":"file-abc123","object":"file","deleted":true}`)(w, r)
		default:
			jsonReply(fixture(t, "file_object.json"))(w, r)
		}
	})
}

func TestFiles_upload(t *testing.T) {
	srv := fileServer(t)
	f, err := ai.UploadFile(t.Context(), srv.provider(), ai.FileUpload{Name: "batchinput.jsonl", MediaType: "application/jsonl", Data: strings.NewReader("{}\n"), Purpose: "batch", ExpiresAfter: 2 * time.Hour})
	if err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	if fileID(f) != "file-abc123" || f.Filename != "batchinput.jsonl" || f.MediaType != "application/jsonl" || len(f.Data) != 0 {
		t.Errorf("UploadFile() = %+v, want the file id in Provider and no data", f)
	}
	parts := parseForm(t, srv.last(t))
	want := map[string]string{"purpose": "batch", "expires_after[anchor]": "created_at", "expires_after[seconds]": "7200"}
	if diff := cmp.Diff(want, formFields(parts)); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
	if files := formFiles(parts, "file"); len(files) != 1 || files[0].Filename != "batchinput.jsonl" || string(files[0].Data) != "{}\n" {
		t.Errorf("file parts = %+v, want the jsonl", files)
	}
}

func TestFiles_uploadDefaultsToUserData(t *testing.T) {
	srv := fileServer(t)
	if _, err := ai.UploadFile(t.Context(), srv.provider(), ai.FileUpload{Data: strings.NewReader("x")}); err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	fields := formFields(parseForm(t, srv.last(t)))
	if diff := cmp.Diff(map[string]string{"purpose": "user_data"}, fields); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
}

func TestFiles_getListDelete(t *testing.T) {
	srv := fileServer(t)
	p := srv.provider()
	got, err := ai.GetFile(t.Context(), p, "file-abc123")
	if err != nil || fileID(got) != "file-abc123" {
		t.Fatalf("GetFile() = %+v, %v, want file-abc123", got, err)
	}
	if last := srv.last(t); last.Path != "/v1/files/file-abc123" {
		t.Errorf("GetFile path = %s, want /v1/files/file-abc123", last.Path)
	}
	list, err := ai.ListFiles(t.Context(), p, ai.FileListRequest{Limit: 2, After: "file-0", Purpose: "user_data"})
	if err != nil {
		t.Fatalf("ListFiles() error = %v", err)
	}
	if len(list.Files) != 2 || fileID(list.Files[1]) != "file-2" || list.Next != "file-2" {
		t.Errorf("ListFiles() = %+v, want two files and next=file-2", list)
	}
	srv.mu.Lock()
	query := ""
	for _, r := range srv.requests {
		if r.Path == "/v1/files" && r.Header.Get("Accept") != "" {
			query = r.Path
		}
	}
	srv.mu.Unlock()
	if query == "" {
		t.Error("list request was not recorded")
	}
	if err := ai.DeleteFile(t.Context(), p, "file-abc123"); err != nil {
		t.Errorf("DeleteFile() error = %v", err)
	}
}

func TestFiles_listQueryAndLastPage(t *testing.T) {
	var query string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.RawQuery
		jsonReply(`{"object":"list","data":[],"first_id":"","last_id":"file-9","has_more":false}`)(w, r)
	})
	list, err := ai.ListFiles(t.Context(), srv.provider(), ai.FileListRequest{Limit: 5, After: "file-1", Purpose: "batch"})
	if err != nil || list.Next != "" {
		t.Fatalf("ListFiles() = %+v, %v, want an empty last page", list, err)
	}
	if diff := cmp.Diff("after=file-1&limit=5&purpose=batch", query); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
}

func TestFiles_idsCannotEscapeTheirPath(t *testing.T) {
	var escaped string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		escaped = r.URL.EscapedPath()
		jsonReply(fixture(t, "file_object.json"))(w, r)
	})
	p := srv.provider()
	if _, err := ai.GetFile(t.Context(), p, "../batches/x"); err != nil {
		t.Fatalf("GetFile() error = %v", err)
	}
	if want := "/v1/files/..%2Fbatches%2Fx"; escaped != want {
		t.Errorf("escaped path = %q, want %q", escaped, want)
	}
	before := srv.count()
	for _, id := range []string{"", ".", ".."} {
		if _, err := ai.GetFile(t.Context(), p, id); err == nil {
			t.Errorf("GetFile(%q) succeeded", id)
		}
		if err := ai.DeleteFile(t.Context(), p, id); err == nil {
			t.Errorf("DeleteFile(%q) succeeded", id)
		}
		if err := ai.ResumeBatch(p.Model("m"), id).Cancel(t.Context()); err == nil {
			t.Errorf("Cancel(%q) succeeded", id)
		}
	}
	if srv.count() != before {
		t.Errorf("%d requests were sent for invalid ids", srv.count()-before)
	}
}

func TestFiles_notDeleted(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"file-1","object":"file","deleted":false}`))
	if err := ai.DeleteFile(t.Context(), srv.provider(), "file-1"); err == nil || !strings.Contains(err.Error(), "not deleted") {
		t.Errorf("DeleteFile() error = %v, want a not-deleted error", err)
	}
}

func TestFiles_capabilityOnModelAndAPIError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"No such File object: x","type":"invalid_request_error","code":null}}`))
	})
	_, err := ai.GetFile(t.Context(), srv.provider().Model("gpt-5-mini"), "x")
	if apiErr, ok := errors.AsType[*ai.APIError](err); !ok || apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("GetFile() error = %v, want a 404 APIError through the model", err)
	}
	if _, err := ai.GetFile(t.Context(), srv.provider().ChatModel("gpt-5-mini"), "x"); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("GetFile() on a chat model error = %v, want ErrUnsupported", err)
	}
}

func TestFiles_limiterAppliesToEveryKind(t *testing.T) {
	srv := fileServer(t)
	lim := &countingLimiter{}
	p := srv.provider(openai.WithLimiter(lim))
	ctx := t.Context()
	if _, err := ai.UploadFile(ctx, p, ai.FileUpload{Data: strings.NewReader("x")}); err != nil {
		t.Fatal(err)
	}
	if _, err := ai.GetFile(ctx, p, "f"); err != nil {
		t.Fatal(err)
	}
	if err := ai.DeleteFile(ctx, p, "f"); err != nil {
		t.Fatal(err)
	}
	if lim.n != 3 {
		t.Errorf("limiter waited %d times, want 3", lim.n)
	}
}

func fileID(f ai.File) string {
	id, _ := openai.FileIDOf(f)
	return id
}

func TestFileIDOf(t *testing.T) {
	tests := []struct {
		name   string
		file   ai.File
		wantID string
		wantOK bool
	}{
		{name: "from FileID", file: ai.File{Provider: openai.FileID("file-1")}, wantID: "file-1", wantOK: true},
		{name: "no provider data", file: ai.File{Data: []byte("x")}},
		{name: "other provider", file: ai.File{Provider: ai.ProviderData{"anthropic": []byte(`{"fileId":"file_1"}`)}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID, gotOK := openai.FileIDOf(tt.file)
			if gotID != tt.wantID || gotOK != tt.wantOK {
				t.Errorf("FileIDOf() = %q, %v, want %q, %v", gotID, gotOK, tt.wantID, tt.wantOK)
			}
		})
	}
}
