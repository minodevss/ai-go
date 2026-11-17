package form_test

import (
	"io"
	"strings"
	"testing"

	"github.com/minodevss/ai-go/internal/form"
)

func TestForm_File_mediaType(t *testing.T) {
	tests := []struct {
		name      string
		mediaType string
		wantType  string
		wantErr   bool
	}{
		{name: "plain", mediaType: "image/png", wantType: "Content-Type: image/png\r\n"},
		{name: "with parameter", mediaType: "text/plain; charset=utf-8", wantType: "Content-Type: text/plain; charset=utf-8\r\n"},
		{name: "empty omits header"},
		{name: "crlf injection", mediaType: "image/png\r\nX-Injected: 1\r\n\r\nPWNED", wantErr: true},
		{name: "lf injection", mediaType: "image/png\nX-Injected: 1", wantErr: true},
		{name: "garbage", mediaType: "not a type", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := form.New()
			f.File("file", "a.png", tt.mediaType, strings.NewReader("data"))
			req, err := f.Request("/x")
			if tt.wantErr {
				if err == nil {
					raw, _ := io.ReadAll(req.Raw)
					t.Fatalf("Request() error = nil, body %q", raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("Request() error = %v", err)
			}
			raw, _ := io.ReadAll(req.Raw)
			body := string(raw)
			if tt.wantType != "" && !strings.Contains(body, tt.wantType) {
				t.Errorf("body %q lacks %q", body, tt.wantType)
			}
			if strings.Contains(body, "X-Injected") {
				t.Errorf("body %q contains an injected header", body)
			}
		})
	}
}

func TestForm_Flag(t *testing.T) {
	tests := []struct {
		name string
		flag *bool
		want string
	}{
		{name: "nil is omitted"},
		{name: "true", flag: new(true), want: "true"},
		{name: "false", flag: new(false), want: "false"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := form.New()
			f.Flag("verbose", tt.flag)
			req, err := f.Request("/x")
			if err != nil {
				t.Fatalf("Request() error = %v", err)
			}
			raw, _ := io.ReadAll(req.Raw)
			body := string(raw)
			if got := strings.Contains(body, `name="verbose"`); got != (tt.want != "") {
				t.Fatalf("body %q: field present = %v, want %v", body, got, tt.want != "")
			}
			if tt.want != "" && !strings.Contains(body, "\r\n\r\n"+tt.want+"\r\n") {
				t.Errorf("body %q lacks value %q", body, tt.want)
			}
		})
	}
}
