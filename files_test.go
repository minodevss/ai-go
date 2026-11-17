package ai_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/synctest"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type storingModel struct {
	*aitest.Model
	*aitest.Files
}

func TestFiles_lifecycle(t *testing.T) {
	store := aitest.NewFiles()
	up, err := ai.UploadFile(t.Context(), store, ai.FileUpload{Name: "a.txt", MediaType: "text/plain", Data: strings.NewReader("hello"), Purpose: "assistants"})
	if err != nil {
		t.Fatalf("UploadFile() error = %v", err)
	}
	id := aitest.FileID(up)
	if id == "" || up.Filename != "a.txt" || up.MediaType != "text/plain" || len(up.Data) != 0 {
		t.Fatalf("UploadFile() = %+v, want an id in Provider and no data", up)
	}
	if data, ok := store.Data(id); !ok || string(data) != "hello" {
		t.Errorf("stored data = %q, want hello", data)
	}
	got, err := ai.GetFile(t.Context(), store, id)
	if err != nil || aitest.FileID(got) != id {
		t.Errorf("GetFile() = %+v, %v, want the same id", got, err)
	}
	if err := ai.DeleteFile(t.Context(), store, id); err != nil {
		t.Fatalf("DeleteFile() error = %v", err)
	}
	if _, err := ai.GetFile(t.Context(), store, id); err == nil {
		t.Error("GetFile() after DeleteFile succeeded")
	}
}

func TestListFiles_paginates(t *testing.T) {
	store := aitest.NewFiles()
	for _, name := range []string{"a", "b", "c"} {
		if _, err := ai.UploadFile(t.Context(), store, ai.FileUpload{Name: name, Data: strings.NewReader(name)}); err != nil {
			t.Fatalf("UploadFile() error = %v", err)
		}
	}
	var names []string
	after := ""
	for range 5 {
		list, err := ai.ListFiles(t.Context(), store, ai.FileListRequest{Limit: 2, After: after})
		if err != nil {
			t.Fatalf("ListFiles() error = %v", err)
		}
		for _, f := range list.Files {
			names = append(names, f.Filename)
		}
		if after = list.Next; after == "" {
			break
		}
	}
	if diff := cmp.Diff([]string{"a", "b", "c"}, names); diff != "" {
		t.Errorf("listed names mismatch (-want +got):\n%s", diff)
	}
}

func TestFiles_capabilityLookupThroughWrappers(t *testing.T) {
	inner := &storingModel{Model: aitest.NewModel(), Files: aitest.NewFiles()}
	tests := []struct {
		name  string
		model any
	}{
		{name: "direct", model: inner},
		{name: "wrapped", model: ai.WrapModel(inner, ai.Middleware{})},
		{name: "fallback child", model: ai.Fallback(aitest.NewModel(), inner)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := ai.UploadFile(t.Context(), tt.model, ai.FileUpload{Data: strings.NewReader("x")})
			if err != nil || aitest.FileID(f) == "" {
				t.Errorf("UploadFile() = %+v, %v, want a stored file", f, err)
			}
		})
	}
}

func TestFiles_unsupportedAndInvalid(t *testing.T) {
	plain := aitest.NewModel()
	if _, err := ai.UploadFile(t.Context(), plain, ai.FileUpload{Data: strings.NewReader("x")}); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("UploadFile() error = %v, want ErrUnsupported", err)
	}
	if _, err := ai.GetFile(t.Context(), nil, "x"); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("GetFile(nil) error = %v, want ErrUnsupported", err)
	}
	if _, err := ai.ListFiles(t.Context(), plain, ai.FileListRequest{}); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("ListFiles() error = %v, want ErrUnsupported", err)
	}
	if err := ai.DeleteFile(t.Context(), plain, "x"); !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("DeleteFile() error = %v, want ErrUnsupported", err)
	}
	if _, err := ai.UploadFile(t.Context(), aitest.NewFiles(), ai.FileUpload{}); err == nil {
		t.Error("UploadFile() without data succeeded")
	}
}

type flakyFiles struct {
	*aitest.Files
	failures int
	reads    [][]byte
}

func (f *flakyFiles) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	data, err := io.ReadAll(c.Data)
	if err != nil {
		return nil, err
	}
	f.reads = append(f.reads, data)
	if len(f.reads) <= f.failures {
		return nil, apiErr(503)
	}
	return f.Files.UploadFile(ctx, &ai.FileUploadCall{Name: c.Name, Data: bytes.NewReader(data)})
}

func TestUploadFile_retriesOnlyRewindableData(t *testing.T) {
	tests := []struct {
		name      string
		data      io.Reader
		wantReads int
		wantErr   bool
	}{
		{name: "seekable", data: bytes.NewReader([]byte("abc")), wantReads: 2},
		{name: "stream", data: io.MultiReader(strings.NewReader("abc")), wantReads: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				f := &flakyFiles{Files: aitest.NewFiles(), failures: 1}
				_, err := ai.UploadFile(t.Context(), f, ai.FileUpload{Data: tt.data})
				if (err != nil) != tt.wantErr || len(f.reads) != tt.wantReads {
					t.Errorf("UploadFile() error = %v after %d reads, want error=%v after %d", err, len(f.reads), tt.wantErr, tt.wantReads)
				}
				for _, r := range f.reads {
					if string(r) != "abc" {
						t.Errorf("an attempt read %q, want the whole data", r)
					}
				}
			})
		})
	}
}
