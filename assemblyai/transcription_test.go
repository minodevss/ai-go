package assemblyai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/assemblyai"
)

type memoryTransport struct {
	handler http.Handler
}

func (m memoryTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	rec := httptest.NewRecorder()
	m.handler.ServeHTTP(rec, r)
	return rec.Result(), nil
}

type script struct {
	mu    sync.Mutex
	polls []string
	seen  []string
	body  string
}

func (s *script) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, r.Method+" "+r.URL.Path)
	switch {
	case r.URL.Path == "/v2/upload":
		raw, _ := io.ReadAll(r.Body)
		s.body = string(raw)
		writeJSON(w, http.StatusOK, `{"upload_url":"https://cdn.example/upload/1"}`)
	case r.Method == http.MethodPost:
		writeJSON(w, http.StatusOK, `{"id":"t1","status":"queued"}`)
	default:
		next := s.polls[0]
		if len(s.polls) > 1 {
			s.polls = s.polls[1:]
		}
		if strings.HasPrefix(next, "!") {
			writeJSON(w, http.StatusServiceUnavailable, `{"error":"busy"}`)
			return
		}
		writeJSON(w, http.StatusOK, next)
	}
}

func scripted(s *script) *assemblyai.Provider {
	return assemblyai.New(assemblyai.WithAPIKey("test-key"), assemblyai.WithBaseURL("http://assemblyai.test"), assemblyai.WithHTTPClient(&http.Client{Transport: memoryTransport{s}}))
}

func TestTranscriptionModel_pollsUntilCompleted(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := &script{polls: []string{fixture(t, "transcript_queued.json"), fixture(t, "transcript_processing.json"), fixture(t, "transcript_completed.json")}}
		start := time.Now()
		res, err := ai.Transcribe(t.Context(), scripted(s).TranscriptionModel("universal-3-pro"), ai.TranscribeRequest{
			Audio: strings.NewReader("audio-bytes"), MediaType: "audio/mpeg", Language: "en",
			Provider: ai.Providers(assemblyai.TranscriptionOptions{SpeakerLabels: true, PollInterval: 2 * time.Second}),
		})
		if err != nil {
			t.Fatalf("Transcribe() error = %v", err)
		}
		if elapsed := time.Since(start); elapsed != 4*time.Second {
			t.Errorf("elapsed = %s, want 4s for two poll waits", elapsed)
		}
		if s.body != "audio-bytes" {
			t.Errorf("uploaded body = %q, want the audio", s.body)
		}
		want := []string{"POST /v2/upload", "POST /v2/transcript", "GET /v2/transcript/t1", "GET /v2/transcript/t1", "GET /v2/transcript/t1"}
		if diff := cmp.Diff(want, s.seen); diff != "" {
			t.Errorf("requests mismatch (-want +got):\n%s", diff)
		}
		wantSegments := []ai.Segment{
			{Text: "Hello there.", Start: 240 * time.Millisecond, End: 1100 * time.Millisecond, Speaker: "A"},
			{Text: "Hi, how are you?", Start: 1500 * time.Millisecond, End: 3100 * time.Millisecond, Speaker: "B"},
		}
		if diff := cmp.Diff(wantSegments, res.Segments); diff != "" || res.Text != "Hello there. Hi, how are you?" || res.Language != "en_us" {
			t.Errorf("result mismatch (-want +got):\n%s\ntext = %q language = %q", diff, res.Text, res.Language)
		}
	})
}

func TestTranscriptionModel_requestBodies(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v2/upload":
			writeJSON(w, http.StatusOK, fixture(t, "upload.json"))
		case r.Method == http.MethodPost:
			writeJSON(w, http.StatusOK, fixture(t, "transcript_queued.json"))
		default:
			writeJSON(w, http.StatusOK, fixture(t, "transcript_completed.json"))
		}
	})
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("universal-3-pro"), ai.TranscribeRequest{
		Audio: strings.NewReader("a"), MediaType: "audio/wav", Language: "en",
		Headers: http.Header{"X-Trace": {"1"}},
		Provider: ai.Providers(assemblyai.TranscriptionOptions{
			SpeakerLabels: true, MinSpeakers: 1, MaxSpeakers: 3, Punctuate: new(true), KeytermsPrompt: []string{"Zoe"}, LanguageDetection: new(true), Words: true, PollInterval: time.Millisecond,
		}),
	})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	if len(res.Segments) != 6 || res.Segments[2] != (ai.Segment{Text: "Hi,", Start: 1500 * time.Millisecond, End: 1800 * time.Millisecond, Speaker: "B"}) {
		t.Errorf("segments = %+v, want one per word with speakers", res.Segments)
	}
	all := srv.all()
	if all[0].Header.Get("Content-Type") != "application/octet-stream" || all[0].Header.Get("Authorization") != "test-key" || all[0].Header.Get("X-Trace") != "1" {
		t.Errorf("upload headers = %v, want raw key auth, octet-stream and custom header", all[0].Header)
	}
	equalJSON(t, `{"audio_url":"https://cdn.assemblyai.com/upload/f756988d-1111-2222-3333-444455556666","speech_models":["universal-3-pro"],"language_code":"en","speaker_labels":true,"speaker_options":{"min_speakers_expected":1,"max_speakers_expected":3},"punctuate":true,"keyterms_prompt":["Zoe"]}`, all[1].Body)
}

func TestTranscriptionModel_failedTranscript(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := &script{polls: []string{fixture(t, "transcript_error.json")}}
		_, err := ai.Transcribe(t.Context(), scripted(s).TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("a")})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != "t1" || !strings.Contains(err.Error(), "unable to download") {
			t.Errorf("Transcribe() error = %v, want a JobError for t1 with the API message", err)
		}
	})
}

func TestTranscriptionModel_contextEndKeepsJobID(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := &script{polls: []string{fixture(t, "transcript_processing.json")}}
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		_, err := ai.Transcribe(ctx, scripted(s).TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("a")})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != "t1" || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Transcribe() error = %v, want a JobError for t1 wrapping the deadline", err)
		}
	})
}

func TestTranscriptionModel_transientPollErrors(t *testing.T) {
	tests := []struct {
		name    string
		polls   []string
		wantErr bool
	}{
		{name: "recovers", polls: []string{"!", "!", "COMPLETED"}},
		{name: "gives up without being retried as a whole", polls: []string{"!"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				polls := slices.Clone(tt.polls)
				for i, p := range polls {
					if p == "COMPLETED" {
						polls[i] = fixture(t, "transcript_completed.json")
					}
				}
				s := &script{polls: polls}
				_, err := ai.Transcribe(t.Context(), scripted(s).TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("a")})
				uploads := 0
				for _, r := range s.seen {
					if r == "POST /v2/upload" {
						uploads++
					}
				}
				if (err != nil) != tt.wantErr || uploads != 1 {
					t.Errorf("Transcribe() error = %v after %d uploads, want error %v after exactly one upload", err, uploads, tt.wantErr)
				}
			})
		})
	}
}

func TestTranscriptionModel_apiErrors(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		body    string
		wantErr error
	}{
		{name: "auth", status: http.StatusUnauthorized, body: `{"error":"Authentication error, API token missing/invalid"}`, wantErr: ai.ErrAuth},
		{name: "rate limit", status: http.StatusTooManyRequests, body: `{"error":"Too many requests"}`, wantErr: ai.ErrRateLimit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Retry-After", "7")
				writeJSON(w, tt.status, tt.body)
			})
			_, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("a"), MaxRetries: new(0)})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !errors.Is(err, tt.wantErr) || !ok || apiErr.Provider != "assemblyai" || apiErr.Message == "" {
				t.Errorf("Transcribe() error = %v, want %v as an APIError with the message", err, tt.wantErr)
			}
			if ok && tt.status == http.StatusTooManyRequests && apiErr.RetryAfter != 7*time.Second {
				t.Errorf("RetryAfter = %s, want 7s", apiErr.RetryAfter)
			}
		})
	}
}

func TestTranscriptionModel_plainTextUploadError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = io.WriteString(w, "Upload failed, please try again.")
	})
	_, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("")})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.StatusCode != http.StatusUnprocessableEntity || apiErr.Body == "" {
		t.Errorf("Transcribe() error = %v, want a 422 APIError keeping the plain body", err)
	}
}

func TestNew_regionsAndLimiter(t *testing.T) {
	var waits int
	limiter := limiterFunc(func(context.Context) error { waits++; return nil })
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, http.StatusOK, `{"error":"x"}`) })
	p := srv.provider(assemblyai.WithLimiter(limiter), assemblyai.WithHeaders(http.Header{"X-Org": {"o"}}), assemblyai.WithRegion(assemblyai.RegionEU))
	_, _ = ai.Transcribe(t.Context(), p.TranscriptionModel(""), ai.TranscribeRequest{Audio: strings.NewReader("a"), MaxRetries: new(0)})
	if waits == 0 || srv.last(t).Header.Get("X-Org") != "o" {
		t.Errorf("waits = %d headers = %v, want limiter use and custom header", waits, srv.last(t).Header)
	}
}

type limiterFunc func(context.Context) error

func (f limiterFunc) Wait(ctx context.Context) error { return f(ctx) }
