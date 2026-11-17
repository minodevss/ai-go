package mediaio_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

type roundTripper func(*http.Request) (*http.Response, error)

func (f roundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func reply(status int, contentType, body string) roundTripper {
	return func(*http.Request) (*http.Response, error) {
		h := http.Header{}
		if contentType != "" {
			h.Set("Content-Type", contentType)
		}
		return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(body))}, nil
	}
}

func downloader(rt roundTripper) mediaio.Downloader {
	return mediaio.Downloader{Provider: "x", Client: transport.New(transport.Config{Provider: "x", BaseURL: "http://api.test", UnrestrictedDownloads: true, Client: &http.Client{Transport: rt}})}
}

func TestDownloader_Result(t *testing.T) {
	tests := []struct {
		name      string
		rt        roundTripper
		url       string
		keepURL   bool
		limit     int64
		want      ai.File
		wantError string
	}{
		{name: "downloads with the served type", rt: reply(200, "image/webp", "data"), url: "https://cdn/a", limit: 10, want: ai.File{MediaType: "image/webp", Data: []byte("data")}},
		{name: "falls back on octet-stream", rt: reply(200, "application/octet-stream", "data"), url: "https://cdn/a", limit: 10, want: ai.File{MediaType: "image/png", Data: []byte("data")}},
		{name: "keeps the url", rt: reply(500, "", ""), url: "https://cdn/a", keepURL: true, want: ai.File{MediaType: "image/png", URL: "https://cdn/a"}},
		{name: "rejects other schemes", rt: reply(200, "", ""), url: "file:///etc/passwd", limit: 10, wantError: "not an https url"},
		{name: "reports status", rt: reply(404, "", "gone"), url: "https://cdn/a", limit: 10, wantError: "status 404"},
		{name: "enforces the limit", rt: reply(200, "image/png", "0123456789ab"), url: "https://cdn/a", limit: 4, wantError: "exceeds"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := downloader(tt.rt).Result(t.Context(), tt.url, "image/png", tt.limit, tt.keepURL)
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Errorf("Result() error = %v, want %q", err, tt.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("Result() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Result() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestInput(t *testing.T) {
	tests := []struct {
		name    string
		file    ai.File
		want    string
		wantErr bool
	}{
		{name: "url wins", file: ai.File{URL: "https://a/b.png", Data: []byte("x")}, want: "https://a/b.png"},
		{name: "data uri", file: ai.File{MediaType: "image/png", Data: []byte("abc")}, want: "data:image/png;base64,YWJj"},
		{name: "empty", file: ai.File{}, wantErr: true},
		{name: "too large", file: ai.File{Data: make([]byte, mediaio.MaxInputBytes+1)}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := mediaio.Input("x", tt.file)
			if (err != nil) != tt.wantErr || got != tt.want {
				t.Errorf("Input() = %q, %v, want %q (error %v)", got, err, tt.want, tt.wantErr)
			}
		})
	}
}

func TestParseError(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantCode    string
		wantMessage string
	}{
		{name: "detail string", body: `{"detail":"Request was throttled."}`, wantMessage: "Request was throttled."},
		{name: "problem json", body: `{"title":"Insufficient credit","detail":"buy credit","status":402}`, wantCode: "Insufficient credit", wantMessage: "buy credit"},
		{name: "validation list", body: `{"detail":[{"loc":["body"],"msg":"too small","type":"x"},{"msg":"too big"}]}`, wantMessage: "too small; too big"},
		{name: "error object", body: `{"error":{"code":"rate_limited","message":"slow"}}`, wantCode: "rate_limited", wantMessage: "slow"},
		{name: "error string", body: `{"error":"boom"}`, wantMessage: "boom"},
		{name: "not json", body: `<html>`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, message := mediaio.ParseError([]byte(tt.body))
			if code != tt.wantCode || message != tt.wantMessage {
				t.Errorf("ParseError() = %q, %q, want %q, %q", code, message, tt.wantCode, tt.wantMessage)
			}
		})
	}
}

func TestBackoff_Poll(t *testing.T) {
	apiErr := &ai.APIError{Provider: "x", StatusCode: http.StatusBadGateway}
	tests := []struct {
		name      string
		results   []error
		done      int
		wantGaps  []time.Duration
		wantErr   bool
		wantAPI   bool
		wantCalls int
	}{
		{name: "grows then caps", results: []error{nil, nil, nil, nil, nil}, done: 5, wantGaps: []time.Duration{time.Second, 1500 * time.Millisecond, 2250 * time.Millisecond, 2500 * time.Millisecond}, wantCalls: 5},
		{name: "transient errors recover", results: []error{apiErr, apiErr, nil}, done: 3, wantGaps: []time.Duration{time.Second, 1500 * time.Millisecond}, wantCalls: 3},
		{name: "persistent errors hide the api error", results: []error{apiErr, apiErr, apiErr, apiErr}, done: 99, wantGaps: []time.Duration{time.Second, 1500 * time.Millisecond}, wantErr: true, wantCalls: 3},
		{name: "permanent error returns at once", results: []error{errors.New("bad")}, done: 99, wantErr: true, wantCalls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var at []time.Time
				calls := 0
				b := mediaio.Backoff{Initial: time.Second, Max: 2500 * time.Millisecond}
				err := b.Poll(t.Context(), "image", "job-1", func(context.Context) (bool, error) {
					at = append(at, time.Now())
					calls++
					return calls >= tt.done, tt.results[calls-1]
				})
				if (err != nil) != tt.wantErr || calls != tt.wantCalls {
					t.Fatalf("Poll() error = %v after %d calls, want error=%v after %d", err, calls, tt.wantErr, tt.wantCalls)
				}
				var gaps []time.Duration
				for i := 1; i < len(at); i++ {
					gaps = append(gaps, at[i].Sub(at[i-1]))
				}
				if diff := cmp.Diff(tt.wantGaps, gaps); diff != "" {
					t.Errorf("gaps mismatch (-want +got):\n%s", diff)
				}
				if jobErr, ok := errors.AsType[*ai.JobError](err); tt.wantErr && (!ok || jobErr.ID != "job-1") {
					t.Errorf("error = %v, want a JobError with the id", err)
				}
				if _, isAPI := errors.AsType[*ai.APIError](err); tt.wantErr && tt.name == "persistent errors hide the api error" && isAPI {
					t.Error("persistent transient error kept its *ai.APIError and would be retried by GenerateImage")
				}
			})
		})
	}
}

func TestBackoff_Poll_contextEnds(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, stop := context.WithTimeout(t.Context(), 3*time.Second)
		defer stop()
		err := mediaio.Backoff{}.Poll(ctx, "image", "job-1", func(context.Context) (bool, error) { return false, nil })
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Poll() error = %v, want the deadline", err)
		}
	})
}

func TestFixError(t *testing.T) {
	err := mediaio.FixError(&ai.APIError{StatusCode: http.StatusPaymentRequired, Code: "x"})
	if !errors.Is(err, ai.ErrQuota) {
		t.Errorf("FixError(402) = %v, want ErrQuota", err)
	}
	if got := mediaio.FixError(nil); got != nil {
		t.Errorf("FixError(nil) = %v, want nil", got)
	}
}

func TestDownloader_Download_blocksInternalTargets(t *testing.T) {
	var hits atomic.Int32
	internal := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = io.WriteString(w, "INTERNAL-METADATA-SECRET")
	}))
	defer internal.Close()
	plain := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer plain.Close()
	redirecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internal.URL, http.StatusFound)
	}))
	defer redirecting.Close()
	d := mediaio.Downloader{Provider: "x", Client: transport.New(transport.Config{Provider: "x", BaseURL: "https://api.example"})}
	tests := []struct {
		name string
		url  string
		want error
	}{
		{name: "loopback https", url: internal.URL + "/latest/meta-data/", want: mediaio.ErrBlockedAddress},
		{name: "loopback http", url: plain.URL},
		{name: "userinfo", url: "https://api.example@" + strings.TrimPrefix(internal.URL, "https://")},
		{name: "link-local metadata", url: "https://169.254.169.254/latest/meta-data/", want: mediaio.ErrBlockedAddress},
		{name: "unspecified", url: "https://0.0.0.0/", want: mediaio.ErrBlockedAddress},
		{name: "private", url: "https://10.0.0.1/", want: mediaio.ErrBlockedAddress},
		{name: "ipv6 loopback", url: "https://[::1]/", want: mediaio.ErrBlockedAddress},
		{name: "mapped loopback", url: "https://[::ffff:127.0.0.1]/", want: mediaio.ErrBlockedAddress},
		{name: "redirect to http loopback", url: redirecting.URL},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := d.Download(t.Context(), tt.url, "image/png", 1<<20)
			if err == nil {
				t.Fatalf("Download() = %q, want error", f.Data)
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("Download() error = %v, want %v", err, tt.want)
			}
		})
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("internal servers saw %d requests, want 0", n)
	}
}

func TestBlockedAddr(t *testing.T) {
	for _, s := range []string{"127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fd00:ec2::254", "::ffff:10.0.0.1", "64:ff9b::7f00:1", "224.0.0.1"} {
		if !mediaio.BlockedAddr(netip.MustParseAddr(s)) {
			t.Errorf("BlockedAddr(%s) = false", s)
		}
	}
	for _, s := range []string{"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"} {
		if mediaio.BlockedAddr(netip.MustParseAddr(s)) {
			t.Errorf("BlockedAddr(%s) = true", s)
		}
	}
}

func TestAudioName(t *testing.T) {
	tests := []struct {
		name      string
		mediaType string
		want      string
	}{
		{name: "empty defaults to mp3", want: "audio.mp3"},
		{name: "mpeg", mediaType: "audio/mpeg", want: "audio.mp3"},
		{name: "wav", mediaType: "audio/wav", want: "audio.wav"},
		{name: "x-wav", mediaType: "audio/x-wav", want: "audio.wav"},
		{name: "webm", mediaType: "audio/webm", want: "audio.webm"},
		{name: "ogg", mediaType: "audio/ogg", want: "audio.ogg"},
		{name: "flac", mediaType: "audio/flac", want: "audio.flac"},
		{name: "m4a", mediaType: "audio/x-m4a", want: "audio.m4a"},
		{name: "parameters are ignored", mediaType: "audio/wav; codecs=1", want: "audio.wav"},
		{name: "unknown type falls back", mediaType: "application/x-nothing", want: "audio.mp3"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mediaio.AudioName(tt.mediaType); got != tt.want {
				t.Errorf("AudioName(%q) = %q, want %q", tt.mediaType, got, tt.want)
			}
		})
	}
}
