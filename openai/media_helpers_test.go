package openai_test

import (
	"bytes"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

type formPart struct {
	Name     string
	Filename string
	Type     string
	Data     []byte
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "media", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}

func parseForm(t *testing.T, c captured) []formPart {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(c.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		t.Fatalf("content type = %q (%v), want multipart/form-data", c.Header.Get("Content-Type"), err)
	}
	r := multipart.NewReader(bytes.NewReader(c.Body), params["boundary"])
	var parts []formPart
	for {
		p, err := r.NextPart()
		if errors.Is(err, io.EOF) {
			return parts
		}
		if err != nil {
			t.Fatalf("read multipart: %v", err)
		}
		data, err := io.ReadAll(p)
		if err != nil {
			t.Fatalf("read part: %v", err)
		}
		parts = append(parts, formPart{Name: p.FormName(), Filename: p.FileName(), Type: p.Header.Get("Content-Type"), Data: data})
	}
}

func formFields(parts []formPart) map[string]string {
	out := map[string]string{}
	for _, p := range parts {
		if p.Filename == "" {
			out[p.Name] = string(p.Data)
		}
	}
	return out
}

func formFiles(parts []formPart, name string) []formPart {
	var out []formPart
	for _, p := range parts {
		if p.Filename != "" && p.Name == name {
			out = append(out, p)
		}
	}
	return out
}

func rawReply(contentType, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, body)
	}
}
