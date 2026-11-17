package deepgram_test

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/deepgram"
)

func TestTranscriptionModel_transcribe(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "listen_response.json")))
	m := srv.provider(deepgram.WithHeaders(http.Header{"X-Trace": {"abc"}})).TranscriptionModel("nova-3")
	res, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{
		Audio:     strings.NewReader("audio-bytes"),
		MediaType: "audio/wav",
		Language:  "en",
		Provider:  ai.Providers(deepgram.TranscriptionOptions{Diarize: true, Keyterms: []string{"Deepgram", "Aura"}, Tags: []string{"t1"}}),
	})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	wantSegments := []ai.Segment{
		{Text: "Yeah, as much as it's worth having a talk to the neighbors.", Start: 80 * time.Millisecond, End: 3100 * time.Millisecond, Speaker: "0"},
		{Text: "Sure.", Start: 3200 * time.Millisecond, End: 3600 * time.Millisecond, Speaker: "1"},
	}
	if diff := cmp.Diff(wantSegments, res.Segments); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	if res.Text != "Yeah, as much as it's worth having a talk to the neighbors. Sure." || res.Language != "en" {
		t.Errorf("text = %q language = %q, want the transcript and en", res.Text, res.Language)
	}
	got := srv.last(t)
	wantQuery := url.Values{"model": {"nova-3"}, "language": {"en"}, "smart_format": {"true"}, "utterances": {"true"}, "diarize_model": {"latest"}, "keyterm": {"Deepgram", "Aura"}, "tag": {"t1"}}
	if diff := cmp.Diff(wantQuery, got.Query); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
	if got.Path != "/v1/listen" || got.Header.Get("Authorization") != "Token test-key" || got.Header.Get("Content-Type") != "audio/wav" || got.Header.Get("X-Trace") != "abc" {
		t.Errorf("request = %s %v, want /v1/listen with token auth, audio/wav and trace header", got.Path, got.Header)
	}
	if string(got.Body) != "audio-bytes" {
		t.Errorf("body = %q, want the raw audio", got.Body)
	}
}

func TestTranscriptionModel_optionsReachTheQuery(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"results":{"channels":[]}}`))
	m := srv.provider().TranscriptionModel("nova-3")
	opts := deepgram.TranscriptionOptions{
		SmartFormat:    new(false),
		Punctuate:      new(true),
		Utterances:     new(false),
		UttSplit:       new(1.5),
		Paragraphs:     true,
		DiarizeModel:   "v2",
		DetectLanguage: true,
		Multichannel:   true,
		Redact:         []string{"pci", "pii"},
		Search:         []string{"refund"},
		Replace:        []string{"a:b"},
		Version:        "latest",
		FillerWords:    true,
		Numerals:       true,
		MIPOptOut:      true,
	}
	if _, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{Audio: strings.NewReader("x"), Language: "ko", Provider: ai.Providers(opts)}); err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	want := url.Values{
		"model": {"nova-3"}, "smart_format": {"false"}, "punctuate": {"true"}, "utterances": {"false"}, "utt_split": {"1.5"}, "paragraphs": {"true"},
		"diarize_model": {"v2"}, "detect_language": {"true"}, "multichannel": {"true"}, "redact": {"pci", "pii"}, "search": {"refund"}, "replace": {"a:b"},
		"version": {"latest"}, "filler_words": {"true"}, "numerals": {"true"}, "mip_opt_out": {"true"},
	}
	got := srv.last(t)
	if diff := cmp.Diff(want, got.Query); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
	if got.Header.Get("Content-Type") != "application/octet-stream" {
		t.Errorf("content type = %q, want application/octet-stream when MediaType is empty", got.Header.Get("Content-Type"))
	}
}

func TestTranscriptionModel_segmentSources(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		wantText     string
		wantSegments []ai.Segment
		wantLanguage string
	}{
		{
			name:         "paragraphs",
			body:         fixture(t, "listen_paragraphs.json"),
			wantText:     "Hello there. How are you?",
			wantSegments: []ai.Segment{{Text: "Hello there. How are you?", Start: 100 * time.Millisecond, End: 2500 * time.Millisecond, Speaker: "0"}},
			wantLanguage: "en",
		},
		{
			name:     "diarized words group into speaker runs",
			body:     `{"results":{"channels":[{"alternatives":[{"transcript":"hi there ok","words":[{"word":"hi","punctuated_word":"Hi","start":0,"end":0.5,"speaker":0},{"word":"there","start":0.5,"end":1,"speaker":0},{"word":"ok","start":1.2,"end":1.5,"speaker":1}]}]}]}}`,
			wantText: "hi there ok",
			wantSegments: []ai.Segment{
				{Text: "Hi there", Start: 0, End: time.Second, Speaker: "0"},
				{Text: "ok", Start: 1200 * time.Millisecond, End: 1500 * time.Millisecond, Speaker: "1"},
			},
		},
		{
			name:     "plain transcript has no segments",
			body:     `{"results":{"channels":[{"alternatives":[{"transcript":"plain","words":[{"word":"plain","start":0,"end":1}]}]}]}}`,
			wantText: "plain",
		},
		{
			name:         "multichannel transcripts are joined",
			body:         `{"results":{"channels":[{"alternatives":[{"transcript":"left"}]},{"detected_language":"ko","alternatives":[{"transcript":"right"}]}]}}`,
			wantText:     "left\nright",
			wantLanguage: "ko",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.body))
			res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeRequest{Audio: strings.NewReader("x")})
			if err != nil {
				t.Fatalf("Transcribe() error = %v", err)
			}
			if res.Text != tt.wantText || res.Language != tt.wantLanguage {
				t.Errorf("Transcribe() text = %q language = %q, want %q %q", res.Text, res.Language, tt.wantText, tt.wantLanguage)
			}
			if diff := cmp.Diff(tt.wantSegments, res.Segments); diff != "" {
				t.Errorf("segments mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTranscriptionModel_apiErrors(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		header        map[string]string
		body          string
		wantIs        error
		wantCode      string
		wantMessage   string
		wantRetryable bool
		wantRetry     time.Duration
	}{
		{name: "legacy auth error", status: http.StatusUnauthorized, body: `{"err_code":"INVALID_AUTH","err_msg":"Invalid credentials.","request_id":"r1"}`, wantIs: ai.ErrAuth, wantCode: "INVALID_AUTH", wantMessage: "Invalid credentials."},
		{name: "modern shape", status: http.StatusBadRequest, body: `{"category":"INVALID_REQUEST","message":"bad audio","details":"x","request_id":"r2"}`, wantCode: "INVALID_REQUEST", wantMessage: "bad audio"},
		{name: "rate limit", status: http.StatusTooManyRequests, header: map[string]string{"Retry-After": "2"}, body: `{"err_code":"TOO_MANY_REQUESTS","err_msg":"slow down"}`, wantIs: ai.ErrRateLimit, wantCode: "TOO_MANY_REQUESTS", wantMessage: "slow down", wantRetryable: true, wantRetry: 2 * time.Second},
		{name: "plain text", status: http.StatusBadGateway, body: `upstream down`, wantMessage: "upstream down", wantRetryable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				for k, v := range tt.header {
					w.Header().Set(k, v)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeRequest{Audio: strings.NewReader("x"), MaxRetries: new(0)})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Transcribe() error = %v, want an *ai.APIError", err)
			}
			if apiErr.Provider != "deepgram" || apiErr.StatusCode != tt.status || apiErr.Code != tt.wantCode || apiErr.Message != tt.wantMessage {
				t.Errorf("APIError = %+v, want status %d code %q message %q", apiErr, tt.status, tt.wantCode, tt.wantMessage)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(%v, %v) = false, want true", err, tt.wantIs)
			}
			if apiErr.Retryable() != tt.wantRetryable || apiErr.RetryAfter != tt.wantRetry {
				t.Errorf("retryable = %v after %s, want %v after %s", apiErr.Retryable(), apiErr.RetryAfter, tt.wantRetryable, tt.wantRetry)
			}
		})
	}
}

func TestTranscriptionModel_retriesSeekableAudio(t *testing.T) {
	calls := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls++; calls == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		jsonReply(`{"results":{"channels":[{"alternatives":[{"transcript":"ok"}]}]}}`)(w, nil)
	})
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeRequest{Audio: strings.NewReader("audio")})
	if err != nil || res.Text != "ok" {
		t.Fatalf("Transcribe() = %v, %v, want ok after one retry", res, err)
	}
	if got := srv.last(t).Body; string(got) != "audio" || srv.count() != 2 {
		t.Errorf("second attempt body = %q after %d requests, want the full audio after 2", got, srv.count())
	}
}

func TestProvider_limiterAndEnvironment(t *testing.T) {
	t.Setenv("DEEPGRAM_API_KEY", "env-key")
	srv := newFakeServer(t, jsonReply(`{"results":{"channels":[]}}`))
	waits := 0
	p := deepgram.New(deepgram.WithBaseURL(srv.URL+"/v1"), deepgram.WithHTTPClient(srv.Client()), deepgram.WithLimiter(limiterFunc(func() { waits++ })))
	if _, err := ai.Transcribe(t.Context(), p.TranscriptionModel("nova-3"), ai.TranscribeRequest{Audio: strings.NewReader("x")}); err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	if got := srv.last(t).Header.Get("Authorization"); got != "Token env-key" || waits != 1 {
		t.Errorf("authorization = %q after %d limiter waits, want the env key after 1", got, waits)
	}
}

type limiterFunc func()

func (f limiterFunc) Wait(context.Context) error {
	f()
	return nil
}
