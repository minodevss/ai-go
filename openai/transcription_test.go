package openai_test

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func TestTranscriptionModel_json(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"text":"Hello world.","languages":[{"code":"en"}],"usage":{"type":"tokens","input_tokens":14,"output_tokens":6,"total_tokens":20}}`))
	m := srv.provider().TranscriptionModel("gpt-4o-transcribe")
	res, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{
		Audio: strings.NewReader("RIFF-audio"), MediaType: "audio/wav", Language: "en",
		Provider: ai.Providers(openai.TranscriptionOptions{Prompt: "names: Zoe", Temperature: new(0.2), ChunkingStrategy: "auto", Languages: []string{"en", "ko"}, Keywords: []string{"Zoe"}}),
	})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	if res.Text != "Hello world." || res.Language != "en" || len(res.Segments) != 0 {
		t.Errorf("Transcribe() = %+v, want text and language from languages[0]", res)
	}
	got := srv.last(t)
	if got.Path != "/v1/audio/transcriptions" {
		t.Errorf("path = %s, want /v1/audio/transcriptions", got.Path)
	}
	parts := parseForm(t, got)
	files := formFiles(parts, "file")
	if len(files) != 1 || files[0].Filename != "audio.wav" || string(files[0].Data) != "RIFF-audio" || files[0].Type != "audio/wav" {
		t.Errorf("file parts = %+v, want audio.wav with the bytes", files)
	}
	want := map[string]string{"model": "gpt-4o-transcribe", "language": "en", "prompt": "names: Zoe", "temperature": "0.2", "chunking_strategy": "auto", "response_format": "json", "keywords[]": "Zoe", "languages[]": "ko"}
	if diff := cmp.Diff(want, formFields(parts)); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
}

func TestTranscriptionModel_whisperReturnsSegments(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "transcription_verbose.json")))
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("whisper-1"), ai.TranscribeRequest{Audio: strings.NewReader("a"), MediaType: "audio/mpeg"})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	want := []ai.Segment{{Text: "Hello", End: 1200 * time.Millisecond}, {Text: " world.", Start: 1200 * time.Millisecond, End: 2500 * time.Millisecond}}
	if diff := cmp.Diff(want, res.Segments); diff != "" || res.Language != "english" {
		t.Errorf("segments mismatch (-want +got):\n%s\nlanguage = %q", diff, res.Language)
	}
	fields := formFields(parseForm(t, srv.last(t)))
	if fields["response_format"] != "verbose_json" || fields["timestamp_granularities[]"] != "segment" {
		t.Errorf("fields = %v, want verbose_json with segment timestamps", fields)
	}
}

func TestTranscriptionModel_diarizedSegmentsCarrySpeakers(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"task":"transcribe","duration":3,"text":"Hi. Hello.","segments":[{"type":"transcript.text.segment","id":"s1","start":0,"end":1.5,"text":"Hi.","speaker":"A"},{"type":"transcript.text.segment","id":"s2","start":1.5,"end":3,"text":"Hello.","speaker":"B"}]}`))
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("gpt-4o-transcribe-diarize"), ai.TranscribeRequest{Audio: strings.NewReader("a")})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	if len(res.Segments) != 2 || res.Segments[0].Speaker != "A" || res.Segments[1].Speaker != "B" || res.Segments[1].Start != 1500*time.Millisecond {
		t.Errorf("segments = %+v, want two speakers with times", res.Segments)
	}
	fields := formFields(parseForm(t, srv.last(t)))
	if fields["response_format"] != "diarized_json" {
		t.Errorf("response_format = %q, want diarized_json", fields["response_format"])
	}
	files := formFiles(parseForm(t, srv.last(t)), "file")
	if len(files) != 1 || files[0].Filename != "audio.mp3" {
		t.Errorf("file parts = %+v, want the default audio.mp3 name", files)
	}
}

func TestTranscriptionModel_limitsAudioSize(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"text":"x"}`))
	m := srv.provider().TranscriptionModel("gpt-4o-transcribe")
	_, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{Audio: io.LimitReader(zeroReader{}, 25<<20+1)})
	if err == nil || !strings.Contains(err.Error(), "25 MB") || srv.count() != 0 {
		t.Errorf("Transcribe() error = %v with %d requests, want a local 25 MB rejection", err, srv.count())
	}
	if _, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{Audio: io.LimitReader(zeroReader{}, 25<<20)}); err != nil {
		t.Errorf("Transcribe() at exactly 25 MB error = %v", err)
	}
}

type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) { return len(p), nil }

func TestTranscriptionModel_retriesSeekableAudio(t *testing.T) {
	calls := 0
	var bodies []string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		raw, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(raw))
		if calls == 1 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		jsonReply(`{"text":"ok"}`)(w, r)
	})
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("gpt-4o-transcribe"), ai.TranscribeRequest{Audio: bytes.NewReader([]byte("audio-bytes"))})
	if err != nil || res.Text != "ok" {
		t.Fatalf("Transcribe() = %+v, %v, want ok after a retry", res, err)
	}
	if len(bodies) != 2 || !strings.Contains(bodies[0], "audio-bytes") || !strings.Contains(bodies[1], "audio-bytes") {
		t.Errorf("attempts carried %d bodies, want both to contain the full audio", len(bodies))
	}
}

func TestTranscriptionModel_stream(t *testing.T) {
	srv := newFakeServer(t, sseReply(fixture(t, "transcription_stream.sse")))
	m := srv.provider().TranscriptionModel("gpt-4o-mini-transcribe")
	s := ai.StreamTranscribe(t.Context(), m, ai.TranscribeStreamRequest{Audio: strings.NewReader("audio"), MediaType: "audio/mpeg", Language: "en"})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	want := []ai.Segment{{Text: "Hello", Partial: true}, {Text: ", world", Partial: true}, {Text: "Hello, world"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	last := srv.last(t)
	fields := formFields(parseForm(t, last))
	if fields["stream"] != "true" || fields["response_format"] != "json" || last.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("fields = %v accept %q, want stream=true with an SSE accept", fields, last.Header.Get("Accept"))
	}
}

func TestTranscriptionModel_streamDiarized(t *testing.T) {
	srv := newFakeServer(t, sseReply(fixture(t, "transcription_diarized_stream.sse")))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("gpt-4o-transcribe-diarize"), ai.TranscribeStreamRequest{Audio: strings.NewReader("audio")})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	want := []ai.Segment{
		{Text: "Hi", Partial: true},
		{Text: "Hi there.", End: 1250 * time.Millisecond, Speaker: "A"},
		{Text: "Hello.", Start: 1250 * time.Millisecond, End: 2500 * time.Millisecond, Speaker: "B"},
	}
	if diff := cmp.Diff(want, got); diff != "" || s.Err() != nil {
		t.Errorf("segments mismatch (-want +got):\n%s\nErr() = %v; the done event must not repeat text already sent as segments", diff, s.Err())
	}
}

func TestTranscriptionModel_streamErrors(t *testing.T) {
	tests := []struct {
		name    string
		handler func(http.ResponseWriter, *http.Request)
		model   string
		want    string
	}{
		{name: "error event", model: "gpt-4o-transcribe", handler: sseReply("data: {\"type\":\"error\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}\n\n"), want: "server_error"},
		{name: "bad json", model: "gpt-4o-transcribe", handler: sseReply("data: {nope\n\n"), want: "decode transcription event"},
		{name: "http error", model: "gpt-4o-transcribe", handler: func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusUnauthorized) }, want: "401"},
		{name: "whisper cannot stream", model: "whisper-1", handler: sseReply(""), want: "does not support streaming"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, tt.handler)
			s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel(tt.model), ai.TranscribeStreamRequest{Audio: strings.NewReader("a")})
			for range s.Segments() {
			}
			if err := s.Err(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Err() = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestTranscriptionModel_streamStopsEarly(t *testing.T) {
	srv := newFakeServer(t, sseReply(fixture(t, "transcription_stream.sse")))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("gpt-4o-transcribe"), ai.TranscribeStreamRequest{Audio: strings.NewReader("a")})
	n := 0
	for range s.Segments() {
		n++
		break
	}
	if n != 1 || s.Err() != nil {
		t.Errorf("segments = %d, Err() = %v, want 1 and nil", n, s.Err())
	}
}

func TestTranscriptionModel_apiError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"Invalid file format.","type":"invalid_request_error","code":"invalid_value"}}`)
	})
	_, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("m"), ai.TranscribeRequest{Audio: strings.NewReader("a")})
	if apiErr, ok := errors.AsType[*ai.APIError](err); !ok || apiErr.Code != "invalid_value" {
		t.Errorf("Transcribe() error = %v, want an invalid_value APIError", err)
	}
}
