package elevenlabs_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/elevenlabs"
)

func TestSpeechModel_generate(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusOK, "audio/mpeg", "mp3-bytes"))
	m := srv.provider().SpeechModel("eleven_multilingual_v2")
	res, err := ai.GenerateSpeech(t.Context(), m, ai.SpeechRequest{
		Text:         "Hello there",
		Voice:        "JBFqnCBsd6RMkjVDRZzb",
		Format:       "pcm_24000",
		Speed:        new(1.1),
		Instructions: "warm",
		Provider: ai.Providers(elevenlabs.SpeechOptions{
			LanguageCode:  "en",
			VoiceSettings: &elevenlabs.VoiceSettings{Stability: new(0.4), UseSpeakerBoost: new(false)},
			Seed:          new(int64(7)),
			EnableLogging: new(false),
		}),
	})
	if err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if string(res.Audio.Data) != "mp3-bytes" || res.Audio.MediaType != "audio/pcm" {
		t.Errorf("audio = %+v, want the body as audio/pcm", res.Audio)
	}
	if len(res.Warnings) != 1 {
		t.Errorf("warnings = %v, want one about instructions", res.Warnings)
	}
	got := srv.last(t)
	if got.Path != "/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb" || got.Header.Get("Xi-Api-Key") != "test-key" || got.Header.Get("Accept") != "audio/*" {
		t.Errorf("request = %s key %q accept %q, want the voice path with the API key", got.Path, got.Header.Get("Xi-Api-Key"), got.Header.Get("Accept"))
	}
	if got.Query.Get("output_format") != "pcm_24000" || got.Query.Get("enable_logging") != "false" {
		t.Errorf("query = %v, want output_format and enable_logging", got.Query)
	}
	equalJSON(t, `{"text":"Hello there","model_id":"eleven_multilingual_v2","language_code":"en","voice_settings":{"stability":0.4,"use_speaker_boost":false,"speed":1.1},"seed":7}`, got.Body)
}

func TestSpeechModel_formatsMapToMediaTypes(t *testing.T) {
	for format, want := range map[string]string{"": "audio/mpeg", "mp3_44100_128": "audio/mpeg", "opus_48000_64": "audio/opus", "wav_16000": "audio/wav", "ulaw_8000": "audio/basic", "alaw_8000": "audio/x-alaw-basic"} {
		srv := newFakeServer(t, reply(http.StatusOK, "audio/mpeg", "x"))
		res, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", Voice: "v", Format: format})
		if err != nil || res.Audio.MediaType != want {
			t.Errorf("format %q: media type = %q (%v), want %q", format, res.Audio.MediaType, err, want)
		}
		if _, set := srv.last(t).Query["output_format"]; set != (format != "") {
			t.Errorf("format %q: output_format set = %v", format, set)
		}
	}
}

func TestSpeechModel_rejectsBadInput(t *testing.T) {
	tests := []struct {
		name string
		req  ai.SpeechRequest
	}{
		{name: "no voice", req: ai.SpeechRequest{Text: "x"}},
		{name: "unknown format", req: ai.SpeechRequest{Text: "x", Voice: "v", Format: "flac_44100"}},
		{name: "bad options", req: ai.SpeechRequest{Text: "x", Voice: "v", Provider: ai.ProviderData{"elevenlabs": []byte(`{"seed":"x"}`)}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, reply(http.StatusOK, "audio/mpeg", "x"))
			tt.req.MaxRetries = new(0)
			if _, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), tt.req); err == nil {
				t.Error("GenerateSpeech() error = nil, want an error")
			}
			if len(srv.requests) != 0 {
				t.Errorf("sent %d requests, want none", len(srv.requests))
			}
		})
	}
}

func TestSpeechModel_errors(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		fixture  string
		wantIs   error
		wantCode string
		wantMsg  string
	}{
		{name: "invalid key", status: http.StatusUnauthorized, fixture: "tts_error_invalid_key.json", wantIs: ai.ErrAuth, wantCode: "invalid_api_key", wantMsg: "Invalid API key"},
		{name: "quota", status: http.StatusTooManyRequests, fixture: "tts_error_quota.json", wantIs: ai.ErrQuota, wantCode: "quota_exceeded", wantMsg: "This request exceeds your quota limit."},
		{name: "validation", status: http.StatusUnprocessableEntity, fixture: "tts_error_validation.json", wantCode: "missing", wantMsg: "Field required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, reply(tt.status, "application/json", fixture(t, tt.fixture)))
			_, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", Voice: "v", MaxRetries: new(0)})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("GenerateSpeech() error = %v, want *ai.APIError", err)
			}
			if apiErr.Code != tt.wantCode || apiErr.Message != tt.wantMsg || apiErr.StatusCode != tt.status {
				t.Errorf("error = %+v, want code %q message %q status %d", apiErr, tt.wantCode, tt.wantMsg, tt.status)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(%v, %v) = false, want true", err, tt.wantIs)
			}
		})
	}
}

func TestSpeechModel_streamOverHTTP(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusOK, "audio/mpeg", strings.Repeat("a", 40000)))
	text := func(yield func(string) bool) {
		_ = yield("Hello ") && yield("world.")
	}
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("eleven_flash_v2_5"), ai.SpeechStreamRequest{
		Text:     text,
		Voice:    "v1",
		Provider: ai.Providers(elevenlabs.SpeechOptions{StreamTransport: elevenlabs.StreamHTTP}),
	})
	total := 0
	for chunk := range s.Audio() {
		total += len(chunk)
	}
	if s.Err() != nil || total != 40000 {
		t.Errorf("stream read %d bytes, error %v, want 40000 bytes", total, s.Err())
	}
	got := srv.last(t)
	if got.Path != "/v1/text-to-speech/v1/stream" {
		t.Errorf("path = %q, want the /stream endpoint", got.Path)
	}
	equalJSON(t, `{"text":"Hello world.","model_id":"eleven_flash_v2_5"}`, got.Body)
}

func TestSpeechModel_streamHTTPErrorSurfaces(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusUnauthorized, "application/json", fixture(t, "tts_error_invalid_key.json")))
	text := func(yield func(string) bool) { yield("x") }
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: text, Voice: "v", Provider: ai.Providers(elevenlabs.SpeechOptions{StreamTransport: elevenlabs.StreamHTTP})})
	for range s.Audio() {
	}
	if !errors.Is(s.Err(), ai.ErrAuth) {
		t.Errorf("Err() = %v, want an authentication error", s.Err())
	}
}

func TestProvider_limiterHeadersAndEnv(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusOK, "audio/mpeg", "x"))
	lim := &countingLimiter{}
	p := srv.provider(elevenlabs.WithLimiter(lim), elevenlabs.WithHeaders(http.Header{"X-Trace": {"t1"}}), elevenlabs.WithHTTPClient(srv.Client()))
	if _, err := ai.GenerateSpeech(t.Context(), p.SpeechModel("m"), ai.SpeechRequest{Text: "x", Voice: "v"}); err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if lim.n != 1 || srv.last(t).Header.Get("X-Trace") != "t1" {
		t.Errorf("limiter calls = %d, trace header = %q, want 1 and t1", lim.n, srv.last(t).Header.Get("X-Trace"))
	}

	t.Setenv("ELEVENLABS_API_KEY", "env-key")
	env := elevenlabs.New(elevenlabs.WithBaseURL(srv.URL))
	if _, err := ai.GenerateSpeech(t.Context(), env.SpeechModel("m"), ai.SpeechRequest{Text: "x", Voice: "v"}); err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if got := srv.last(t).Header.Get("Xi-Api-Key"); got != "env-key" {
		t.Errorf("api key header = %q, want env-key", got)
	}
}

type countingLimiter struct{ n int }

func (l *countingLimiter) Wait(_ context.Context) error {
	l.n++
	return nil
}
