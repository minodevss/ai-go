package record_test

import (
	"bytes"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/aitest/record"
)

func setUpdate(t *testing.T, on bool) {
	t.Helper()
	value := "false"
	if on {
		value = "true"
	}
	if err := flag.Set("update", value); err != nil {
		t.Fatalf("flag.Set() error = %v", err)
	}
	t.Cleanup(func() {
		if err := flag.Set("update", "false"); err != nil {
			t.Errorf("flag.Set() error = %v", err)
		}
	})
}

func post(t *testing.T, client *http.Client, url, body string) (*http.Response, string) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	return resp, string(raw)
}

func TestTransport_recordAndReplay(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Set-Cookie", "session=abc")
		w.Header().Set("X-Request-Id", "req_1")
		if strings.HasSuffix(r.URL.Path, "/stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n")
			return
		}
		_, _ = io.WriteString(w, `{"ok":true,"text":"<&>"}`)
	}))
	defer upstream.Close()
	path := filepath.Join(t.TempDir(), "nested", "fixture.json")

	setUpdate(t, true)
	t.Run("record", func(t *testing.T) {
		rt := record.New(t, path, record.WithUpstream(http.DefaultTransport), record.WithBodyRewrite(func(b []byte) []byte {
			return bytes.ReplaceAll(b, []byte(`"q":1`), []byte(`"q":2`))
		}))
		client := &http.Client{Transport: rt}
		_, body := post(t, client, upstream.URL+"/json", `{"q":1}`)
		if body != `{"ok":true,"text":"<&>"}` {
			t.Errorf("recorded body = %q", body)
		}
		post(t, client, upstream.URL+"/stream", `{"stream":true}`)
	})
	setUpdate(t, false)

	file, err := record.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(file.Exchanges) != 2 {
		t.Fatalf("recorded %d exchanges, want 2", len(file.Exchanges))
	}
	first := file.Exchanges[0]
	if string(first.Request.Bytes()) != `{"q":2}` {
		t.Errorf("recorded request body = %s, want the rewritten body", first.Request.Bytes())
	}
	if got := first.Request.Header.Get("Authorization"); got != "REDACTED" {
		t.Errorf("recorded Authorization = %q, want REDACTED", got)
	}
	if got := first.Response.Header.Get("Set-Cookie"); got != "REDACTED" {
		t.Errorf("recorded Set-Cookie = %q, want REDACTED", got)
	}
	if got := first.Response.Header.Get("X-Request-Id"); got != "req_1" {
		t.Errorf("recorded X-Request-Id = %q, want req_1", got)
	}
	if string(first.Response.Bytes()) != `{"ok":true,"text":"<&>"}` || first.Request.BodyLines != nil {
		t.Errorf("JSON bodies were not stored as JSON: %+v", first)
	}
	wantLines := []string{"event: a", "data: 1", "", "event: b", "data: 2", "", ""}
	if diff := cmp.Diff(wantLines, file.Exchanges[1].Response.BodyLines); diff != "" {
		t.Errorf("SSE lines mismatch (-want +got):\n%s", diff)
	}

	t.Run("replay", func(t *testing.T) {
		rt := record.New(t, path, record.WithBodyMatch())
		client := &http.Client{Transport: rt}
		resp, body := post(t, client, "http://offline.invalid/json", `{"q":2}`)
		if body != `{"ok":true,"text":"<&>"}` || resp.Header.Get("X-Request-Id") != "req_1" || resp.StatusCode != http.StatusOK {
			t.Errorf("replayed %d %q with header %v", resp.StatusCode, body, resp.Header)
		}
		_, stream := post(t, client, "http://offline.invalid/stream", `{"stream":true}`)
		if stream != "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n" {
			t.Errorf("replayed stream = %q", stream)
		}
		if got := len(rt.Requests()); got != 2 {
			t.Errorf("Requests() has %d entries, want 2", got)
		}
	})
}

func TestTransport_replayErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f.json")
	if err := record.Save(path, record.File{Exchanges: []record.Exchange{{
		Request:  record.Request{Method: "POST", URL: "https://api.test/v1/a", Body: []byte(`{"x":1}`)},
		Response: record.Response{Status: http.StatusOK, Body: []byte(`{}`)},
	}}}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	tests := []struct {
		name string
		opts []record.Option
		url  string
		body string
		want string
	}{
		{name: "different path", url: "https://api.test/v1/b", body: `{"x":1}`, want: "recorded POST /v1/a"},
		{name: "body mismatch", opts: []record.Option{record.WithBodyMatch()}, url: "https://api.test/v1/a", body: `{"x":2}`, want: "body differs"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &http.Client{Transport: record.New(t, path, tt.opts...)}
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, tt.url, strings.NewReader(tt.body))
			if err != nil {
				t.Fatalf("NewRequest() error = %v", err)
			}
			_, err = client.Do(req) //nolint:bodyclose // the request fails before a response exists
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Do() error = %v, want %q", err, tt.want)
			}
		})
	}
	t.Run("extra request", func(t *testing.T) {
		client := &http.Client{Transport: record.New(t, path)}
		post(t, client, "https://api.test/v1/a", `{}`)
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://api.test/v1/a", strings.NewReader(`{}`))
		if err != nil {
			t.Fatalf("NewRequest() error = %v", err)
		}
		_, err = client.Do(req) //nolint:bodyclose // the request fails before a response exists
		if err == nil || !strings.Contains(err.Error(), "no recording") {
			t.Errorf("Do() error = %v, want no recording", err)
		}
	})
}

func TestLoad_missing(t *testing.T) {
	if _, err := record.Load(filepath.Join(t.TempDir(), "none.json")); err == nil {
		t.Error("Load() error = nil, want error")
	}
}

type upstreamFunc func(*http.Request) (*http.Response, error)

func (f upstreamFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestTransport_redactsSecrets(t *testing.T) {
	setUpdate(t, true)
	path := filepath.Join(t.TempDir(), "f.json")
	upstream := upstreamFunc(func(*http.Request) (*http.Response, error) {
		h := http.Header{
			"X-Goog-Upload-Url": {"https://upload.example/up?upload_id=secret-upload-id&upload_protocol=resumable"},
			"X-Custom-Token":    {"RESP-TOKEN"},
			"Content-Type":      {"application/json"},
		}
		return &http.Response{StatusCode: http.StatusOK, Header: h, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	t.Run("record", func(t *testing.T) {
		client := &http.Client{Transport: record.New(t, path, record.WithUpstream(upstream))}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://user:pass@api.example/v1?key=URLKEY&alt=sse", strings.NewReader(`{}`))
		if err != nil {
			t.Fatalf("NewRequest() error = %v", err)
		}
		for k, v := range map[string]string{"X-Goog-Api-Key": "AIzaSECRET", "Xi-Api-Key": "XI-SECRET", "X-Key": "BFL-SECRET", "Authorization": "Bearer S", "X-Team-Secret": "TEAM-SECRET", "Content-Type": "application/json"} {
			req.Header.Set(k, v)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("Do() error = %v", err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	})
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	for _, secret := range []string{"AIzaSECRET", "XI-SECRET", "BFL-SECRET", "Bearer S", "TEAM-SECRET", "RESP-TOKEN", "secret-upload-id", "URLKEY", "user:pass"} {
		if bytes.Contains(raw, []byte(secret)) {
			t.Errorf("fixture leaks %q:\n%s", secret, raw)
		}
	}
	for _, kept := range []string{"alt=sse", "application/json", "upload_protocol=resumable"} {
		if !bytes.Contains(raw, []byte(kept)) {
			t.Errorf("fixture lost %q", kept)
		}
	}
}

func TestWithRedactedHeaders(t *testing.T) {
	setUpdate(t, true)
	path := filepath.Join(t.TempDir(), "f.json")
	var live http.Header
	upstream := upstreamFunc(func(r *http.Request) (*http.Response, error) {
		live = r.Header.Clone()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"X-Tenant-Reply": {"tenant-reply"}, "X-Public": {"public"}},
			Body:       io.NopCloser(strings.NewReader(`{}`)),
		}, nil
	})
	t.Run("record", func(t *testing.T) {
		client := &http.Client{Transport: record.New(t, path, record.WithUpstream(upstream), record.WithRedactedHeaders("X-Tenant", "X-Tenant-Reply"))}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://api.example/v1", strings.NewReader(`{}`))
		if err != nil {
			t.Fatalf("NewRequest() error = %v", err)
		}
		req.Header.Set("X-Tenant", "tenant-1")
		req.Header.Set("X-Public", "public")
		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("Do() error = %v", err)
		}
		_ = resp.Body.Close()
	})
	if got := live.Get("X-Tenant"); got != "tenant-1" {
		t.Errorf("live X-Tenant = %q, want tenant-1", got)
	}
	file, err := record.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	ex := file.Exchanges[0]
	tests := []struct {
		name string
		got  string
		want string
	}{
		{name: "request listed", got: ex.Request.Header.Get("X-Tenant"), want: "REDACTED"},
		{name: "request unlisted", got: ex.Request.Header.Get("X-Public"), want: "public"},
		{name: "response listed", got: ex.Response.Header.Get("X-Tenant-Reply"), want: "REDACTED"},
		{name: "response unlisted", got: ex.Response.Header.Get("X-Public"), want: "public"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("recorded header = %q, want %q", tt.got, tt.want)
			}
		})
	}
}
