package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func echoAudio(t *testing.T) func(http.ResponseWriter, *http.Request) {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "<"+req.Input+">")
	}
}

func TestSpeechModel_generate(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	m := srv.provider().SpeechModel("gpt-4o-mini-tts")
	res, err := ai.GenerateSpeech(t.Context(), m, ai.SpeechRequest{Text: "Hello there", Voice: "marin", Format: "wav", Instructions: "warm", Speed: new(1.5)})
	if err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if string(res.Audio.Data) != "<Hello there>" || res.Audio.MediaType != "audio/wav" {
		t.Errorf("audio = %+v, want the body as audio/wav", res.Audio)
	}
	got := srv.last(t)
	if got.Path != "/v1/audio/speech" || got.Header.Get("Accept") != "audio/*" {
		t.Errorf("request = %s accept %q, want /v1/audio/speech accepting audio", got.Path, got.Header.Get("Accept"))
	}
	equalJSON(t, `{"model":"gpt-4o-mini-tts","input":"Hello there","voice":"marin","instructions":"warm","response_format":"wav","speed":1.5}`, got.Body)
}

func TestSpeechModel_voices(t *testing.T) {
	tests := []struct {
		name  string
		voice string
		want  string
	}{
		{name: "default", voice: "", want: `"alloy"`},
		{name: "built in", voice: "cedar", want: `"cedar"`},
		{name: "custom voice object", voice: "voice_1234", want: `{"id":"voice_1234"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, echoAudio(t))
			if _, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", Voice: tt.voice}); err != nil {
				t.Fatalf("GenerateSpeech() error = %v", err)
			}
			var req struct {
				Voice json.RawMessage `json:"voice"`
			}
			if err := json.Unmarshal(srv.last(t).Body, &req); err != nil || string(req.Voice) != tt.want {
				t.Errorf("voice = %s (%v), want %s", req.Voice, err, tt.want)
			}
		})
	}
}

func TestSpeechModel_formatsMapToMediaTypes(t *testing.T) {
	for format, want := range map[string]string{"": "audio/mpeg", "mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac", "flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm"} {
		srv := newFakeServer(t, echoAudio(t))
		res, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", Format: format})
		if err != nil || res.Audio.MediaType != want {
			t.Errorf("format %q: media type = %q (%v), want %q", format, res.Audio.MediaType, err, want)
		}
	}
}

func TestSpeechModel_rejectsLongText(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	_, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: strings.Repeat("가", 4097)})
	if err == nil || !strings.Contains(err.Error(), "4096") || !strings.Contains(err.Error(), "StreamSpeech") || srv.count() != 0 {
		t.Errorf("GenerateSpeech() error = %v with %d requests, want a local rejection pointing to StreamSpeech", err, srv.count())
	}
	if _, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: strings.Repeat("가", 4096)}); err != nil {
		t.Errorf("GenerateSpeech() at 4096 characters error = %v", err)
	}
}

func collectAudio(t *testing.T, s *ai.SpeechStream) []string {
	t.Helper()
	var out []string
	for chunk := range s.Audio() {
		out = append(out, string(chunk))
	}
	return out
}

func textSeq(pieces ...string) func(func(string) bool) {
	return func(yield func(string) bool) {
		for _, p := range pieces {
			if !yield(p) {
				return
			}
		}
	}
}

func TestSpeechModel_streamSendsOneRequestPerSentence(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	m := srv.provider().SpeechModel("gpt-4o-mini-tts")
	s := ai.StreamSpeech(t.Context(), m, ai.SpeechStreamRequest{
		Text:   textSeq("The weather in Seoul ", "is sunny today. It will ", "stay warm all week! Enjoy", " it."),
		Voice:  "alloy",
		Format: "pcm",
	})
	got := strings.Join(collectAudio(t, s), "")
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if want := "<The weather in Seoul is sunny today.><It will stay warm all week!><Enjoy it.>"; got != want {
		t.Errorf("audio = %q, want %q", got, want)
	}
	if srv.count() != 3 || s.MediaType() != "audio/pcm" {
		t.Errorf("requests = %d, media type = %q, want 3 and audio/pcm", srv.count(), s.MediaType())
	}
}

func TestSpeechModel_streamYieldsAudioWhileTheBodyArrives(t *testing.T) {
	release := make(chan struct{})
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "first")
		_ = http.NewResponseController(w).Flush()
		<-release
		_, _ = io.WriteString(w, "second")
	})
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("One sentence here is enough.")})
	var chunks []string
	for chunk := range s.Audio() {
		chunks = append(chunks, string(chunk))
		if len(chunks) == 1 {
			close(release)
		}
	}
	if diff := cmp.Diff([]string{"first", "second"}, chunks); diff != "" {
		t.Errorf("chunks mismatch (-want +got):\n%s", diff)
	}
}

func TestSpeechModel_streamStopsRequestingWhenTheConsumerStops(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("First sentence is here, long enough. ", "Second sentence is here, also long. ", "Third sentence is here, too.")})
	for range s.Audio() {
		break
	}
	if srv.count() != 1 || s.Err() != nil {
		t.Errorf("requests = %d, Err() = %v, want 1 and nil", srv.count(), s.Err())
	}
}

func TestSpeechModel_streamReportsMidStreamFailure(t *testing.T) {
	calls := 0
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if calls++; calls == 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"slow down","code":"rate_limit_exceeded"}}`)
			return
		}
		echoAudio(t)(w, r)
	})
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("First sentence is here, long enough. ", "Second sentence is here, also long.")})
	got := collectAudio(t, s)
	if len(got) != 1 || s.Err() == nil || !strings.Contains(s.Err().Error(), "429") {
		t.Errorf("audio = %v, Err() = %v, want the first sentence and a 429", got, s.Err())
	}
}

func TestSpeechModel_streamSplitsOversizedSentences(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	long := strings.Repeat("word ", 1500)
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq(long)})
	collectAudio(t, s)
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if srv.count() != 2 {
		t.Errorf("requests = %d, want the 7500 characters split in 2", srv.count())
	}
	srv.mu.Lock()
	defer srv.mu.Unlock()
	for _, r := range srv.requests {
		var req struct {
			Input string `json:"input"`
		}
		if err := json.Unmarshal(r.Body, &req); err != nil || len([]rune(req.Input)) > 4096 {
			t.Errorf("a request carried %d characters (%v), want at most 4096", len([]rune(req.Input)), err)
		}
	}
}

func TestSpeechModel_streamWithoutInputSendsNothing(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("  ", "\n")})
	if got := collectAudio(t, s); len(got) != 0 || srv.count() != 0 {
		t.Errorf("audio = %v, requests = %d, want nothing for blank text", got, srv.count())
	}
}

func TestSpeechModel_providerOptionsTypeName(t *testing.T) {
	if got := (openai.VideoOptions{}).ProviderName(); got != "openai" {
		t.Errorf("ProviderName() = %q, want openai", got)
	}
}

func TestSpeechModel_streamCutsAfterFullWidthTerminators(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("今日は天気がいいです。明日も晴れるでしょう。散歩に行きましょう。", "楽しみです！")})
	got := strings.Join(collectAudio(t, s), "")
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if want := "<今日は天気がいいです。明日も晴れるでしょう。散歩に行きましょう。><楽しみです！>"; got != want || srv.count() != 2 {
		t.Errorf("audio = %q with %d requests, want %q with 2", got, srv.count(), want)
	}
}

func TestSpeechModel_streamRejectsContainerFormats(t *testing.T) {
	srv := newFakeServer(t, echoAudio(t))
	for _, format := range []string{"wav", "flac", "opus", "aac"} {
		t.Run(format, func(t *testing.T) {
			s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: textSeq("Hello there, friend."), Format: format})
			for range s.Audio() {
			}
			if err := s.Err(); err == nil || !strings.Contains(err.Error(), "mp3 or pcm") {
				t.Errorf("Err() = %v, want a format error", err)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("server saw %d requests, want 0", srv.count())
	}
}
