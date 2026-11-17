package ui_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/ui"
)

const userBody = `{"id":"chat-1","trigger":"submit-message","messageId":"m9","extra":{"a":1},"messages":[{"id":"u1","role":"user","parts":[{"type":"text","text":"hi"}]}]}`

func TestReadRequest_decodesUseChatBody(t *testing.T) {
	req := readRequest(t, userBody)
	if req.ID != "chat-1" || req.Trigger != ui.TriggerSubmit || req.MessageID != "m9" || len(req.Messages) != 1 {
		t.Errorf("request = %+v", req)
	}
	if string(req.Extra["extra"]) != `{"a":1}` {
		t.Errorf("Extra = %v, want the transport's body fields", req.Extra)
	}
	if got := textOf(req.Messages[0]); got != "hi" {
		t.Errorf("message text = %q, want hi", got)
	}
}

func TestReadRequest_rejects(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		contentTyp string
		limits     ui.Limits
		wantStatus int
		wantIs     error
	}{
		{name: "oversized body", body: userBody, limits: ui.Limits{MaxBodyBytes: 40}, wantStatus: http.StatusRequestEntityTooLarge, wantIs: ui.ErrRequestTooBig},
		{name: "default cap is 1 MiB", body: `{"messages":[],"pad":"` + strings.Repeat("a", 1<<20) + `"}`, wantStatus: http.StatusRequestEntityTooLarge, wantIs: ui.ErrRequestTooBig},
		{name: "not json", body: `{`, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "not an object", body: `[]`, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "wrong content type", body: userBody, contentTyp: "text/plain", wantStatus: http.StatusUnsupportedMediaType, wantIs: ui.ErrInvalidRequest},
		{name: "too many messages", body: `{"messages":[{"id":"a","role":"user","parts":[]},{"id":"b","role":"user","parts":[]}]}`, limits: ui.Limits{MaxMessages: 1}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "too many parts", body: `{"messages":[{"id":"a","role":"user","parts":[{"type":"text","text":"1"},{"type":"text","text":"2"}]}]}`, limits: ui.Limits{MaxParts: 1}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "too deep", body: `{"messages":[{"id":"a","role":"assistant","parts":[{"type":"tool-x","toolCallId":"c","state":"input-available","input":{"a":{"b":{"c":{}}}}}]}]}`, limits: ui.Limits{MaxDepth: 6}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "unknown role", body: `{"messages":[{"id":"a","role":"tool","parts":[]}]}`, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "unknown trigger", body: `{"trigger":"hack","messages":[]}`, wantStatus: http.StatusBadRequest, wantIs: ui.ErrInvalidRequest},
		{name: "remote file url", body: fileBody("https://169.254.169.254/latest/meta-data"), wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
		{name: "file scheme", body: fileBody("file:///etc/passwd"), limits: ui.Limits{FilePolicy: ui.FilePolicy{AllowURL: func(*url.URL) bool { return true }}}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
		{name: "url rejected by policy", body: fileBody("https://evil.test/a.png"), limits: ui.Limits{FilePolicy: ui.FilePolicy{AllowURL: func(u *url.URL) bool { return u.Host == "cdn.example.com" }}}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
		{name: "provider file reference", body: `{"messages":[{"id":"a","role":"user","parts":[{"type":"file","mediaType":"image/png","url":"data:image/png;base64,AA==","providerReference":{"openai":"file-abc"}}]}]}`, wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
		{name: "media type outside policy", body: fileBody("data:application/x-sh;base64,AA=="), limits: ui.Limits{FilePolicy: ui.FilePolicy{MediaTypes: []string{"image/*"}}}, wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
		{name: "data url over the file cap", body: fileBody("data:image/png;base64,AAAAAAAA"), limits: ui.Limits{FilePolicy: ui.FilePolicy{MaxBytes: 3}}, wantStatus: http.StatusRequestEntityTooLarge, wantIs: ui.ErrFile},
		{name: "malformed data url", body: fileBody("data:image/png;base64,***"), wantStatus: http.StatusBadRequest, wantIs: ui.ErrFile},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := jsonRequest(t, tt.body)
			if tt.contentTyp != "" {
				r.Header.Set("Content-Type", tt.contentTyp)
			}
			_, err := ui.ReadRequest(r, tt.limits)
			re, ok := errors.AsType[*ui.RequestError](err)
			if !ok || re.Status != tt.wantStatus || !errors.Is(err, tt.wantIs) {
				t.Errorf("ReadRequest() error = %v, want status %d wrapping %v", err, tt.wantStatus, tt.wantIs)
			}
		})
	}
}

func fileBody(u string) string {
	return `{"messages":[{"id":"a","role":"user","parts":[{"type":"file","mediaType":"image/png","url":"` + u + `"}]}]}`
}

func TestReadRequest_acceptsFilesByPolicy(t *testing.T) {
	tests := []struct {
		name   string
		url    string
		limits ui.Limits
	}{
		{name: "data url by default", url: "data:image/png;base64,AAEC"},
		{name: "allowed media type", url: "data:image/png;base64,AAEC", limits: ui.Limits{FilePolicy: ui.FilePolicy{MediaTypes: []string{"image/*"}, MaxBytes: 3}}},
		{name: "allowed host", url: "https://cdn.example.com/a.png", limits: ui.Limits{FilePolicy: ui.FilePolicy{AllowURL: func(u *url.URL) bool { return u.Host == "cdn.example.com" }}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ui.ReadRequest(jsonRequest(t, fileBody(tt.url)), tt.limits); err != nil {
				t.Errorf("ReadRequest() error = %v", err)
			}
		})
	}
}

func TestReadRequest_bodyIsCappedWhileReading(t *testing.T) {
	r := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", strings.NewReader(strings.Repeat("x", 100)))
	r.Header.Set("Content-Type", "application/json")
	_, err := ui.ReadRequest(r, ui.Limits{MaxBodyBytes: 10})
	if !errors.Is(err, ui.ErrRequestTooBig) {
		t.Errorf("ReadRequest() error = %v, want %v", err, ui.ErrRequestTooBig)
	}
}

func TestRequestError_unwraps(t *testing.T) {
	_, err := ui.ReadRequest(jsonRequest(t, `{`), ui.Limits{})
	want := "ui: invalid request: body is not valid JSON"
	if err == nil || err.Error() != want {
		t.Errorf("ReadRequest() error = %v, want %q", err, want)
	}
	if diff := cmp.Diff(http.StatusBadRequest, func() int { re, _ := errors.AsType[*ui.RequestError](err); return re.Status }()); diff != "" {
		t.Errorf("status mismatch (-want +got):\n%s", diff)
	}
}
