package deepgram_test

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/deepgram"
)

func audioReply(contentType, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(body))
	}
}

func TestSpeechModel_generate(t *testing.T) {
	srv := newFakeServer(t, audioReply("audio/mpeg", "mp3-bytes"))
	m := srv.provider().SpeechModel("aura-2-thalia-en")
	res, err := ai.GenerateSpeech(t.Context(), m, ai.SpeechRequest{Text: "Hello there", Speed: new(1.2), Instructions: "warm", Provider: ai.Providers(deepgram.SpeechOptions{BitRate: 48000, Tags: []string{"a"}})})
	if err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if string(res.Audio.Data) != "mp3-bytes" || res.Audio.MediaType != "audio/mpeg" || len(res.Warnings) != 1 {
		t.Errorf("result = %+v, want the audio and one warning for instructions", res)
	}
	got := srv.last(t)
	wantQuery := url.Values{"model": {"aura-2-thalia-en"}, "encoding": {"mp3"}, "bit_rate": {"48000"}, "speed": {"1.2"}, "tag": {"a"}}
	if diff := cmp.Diff(wantQuery, got.Query); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
	if got.Path != "/v1/speak" || got.Header.Get("Authorization") != "Token test-key" || got.Header.Get("Accept") != "audio/*" || got.Header.Get("Content-Type") != "application/json" {
		t.Errorf("request = %s %v, want /v1/speak with token auth", got.Path, got.Header)
	}
	equalJSON(t, `{"text":"Hello there"}`, got.Body)
}

func TestSpeechModel_formats(t *testing.T) {
	tests := []struct {
		name          string
		format        string
		voice         string
		wantQuery     url.Values
		wantMediaType string
	}{
		{name: "wav", format: "wav", wantQuery: url.Values{"model": {"m"}, "encoding": {"linear16"}, "container": {"wav"}}, wantMediaType: "audio/wav"},
		{name: "pcm with voice override", format: "pcm", voice: "aura-asteria-en", wantQuery: url.Values{"model": {"aura-asteria-en"}, "encoding": {"linear16"}, "container": {"none"}}, wantMediaType: "audio/pcm"},
		{name: "flac", format: "flac", wantQuery: url.Values{"model": {"m"}, "encoding": {"flac"}}, wantMediaType: "audio/flac"},
		{name: "mulaw", format: "mulaw", wantQuery: url.Values{"model": {"m"}, "encoding": {"mulaw"}, "container": {"none"}}, wantMediaType: "audio/mulaw"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header()["Content-Type"] = nil
				_, _ = w.Write([]byte("x"))
			})
			res, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", Format: tt.format, Voice: tt.voice})
			if err != nil {
				t.Fatalf("GenerateSpeech() error = %v", err)
			}
			if diff := cmp.Diff(tt.wantQuery, srv.last(t).Query); diff != "" {
				t.Errorf("query mismatch (-want +got):\n%s", diff)
			}
			if res.Audio.MediaType != tt.wantMediaType {
				t.Errorf("media type = %q, want %q", res.Audio.MediaType, tt.wantMediaType)
			}
		})
	}
}

func TestSpeechModel_rejects(t *testing.T) {
	srv := newFakeServer(t, audioReply("audio/mpeg", "x"))
	m := srv.provider().SpeechModel("m")
	tests := []struct {
		name string
		req  ai.SpeechRequest
		want string
	}{
		{name: "too long", req: ai.SpeechRequest{Text: strings.Repeat("a", 2001)}, want: "limit is 2000"},
		{name: "format", req: ai.SpeechRequest{Text: "x", Format: "ogg"}, want: "unsupported speech format"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateSpeech(t.Context(), m, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateSpeech() error = %v, want %q", err, tt.want)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("server saw %d requests, want 0", srv.count())
	}
}

func TestSpeechModel_apiError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write([]byte(`{"err_code":"INSUFFICIENT_PERMISSIONS","err_msg":"no credits"}`))
	})
	_, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechRequest{Text: "x", MaxRetries: new(0)})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.StatusCode != http.StatusPaymentRequired || apiErr.Code != "INSUFFICIENT_PERMISSIONS" {
		t.Errorf("GenerateSpeech() error = %v, want a 402 APIError", err)
	}
}

func TestSpeechModel_streamSpeech(t *testing.T) {
	var sent []string
	done := make(chan struct{})
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		defer close(done)
		writeText(ctx, t, c, `{"type":"Metadata","request_id":"r","model_name":"m","model_version":"v","model_uuid":"u"}`)
		flushes := 0
		for f := range readFrames(ctx, c) {
			sent = append(sent, string(f.Data))
			switch string(f.Data) {
			case `{"type":"Flush"}`:
				if err := c.Write(ctx, websocket.MessageBinary, []byte{byte(flushes)}); err != nil {
					t.Errorf("server write: %v", err)
				}
				writeText(ctx, t, c, `{"type":"Flushed","sequence_id":`+string(rune('0'+flushes))+`}`)
				flushes++
			case `{"type":"Close"}`:
				_ = c.Close(websocket.StatusNormalClosure, "")
				return
			}
		}
	})
	text := func(yield func(string) bool) {
		for _, piece := range []string{"First sentence", " is here. Second", " one follows. Tail"} {
			if !yield(piece) {
				return
			}
		}
	}
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("aura-2-thalia-en"), ai.SpeechStreamRequest{Text: text, Format: "mulaw", Speed: new(0.9), Provider: ai.Providers(deepgram.SpeechOptions{SampleRate: 8000})})
	var got [][]byte
	for chunk := range s.Audio() {
		got = append(got, chunk)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	<-done
	if diff := cmp.Diff([][]byte{{0}, {1}, {2}}, got); diff != "" {
		t.Errorf("audio mismatch (-want +got):\n%s", diff)
	}
	if s.MediaType() != "audio/mulaw" {
		t.Errorf("MediaType() = %q, want audio/mulaw", s.MediaType())
	}
	wantSent := []string{
		`{"type":"Speak","text":"First sentence is here."}`, `{"type":"Flush"}`,
		`{"type":"Speak","text":"Second one follows."}`, `{"type":"Flush"}`,
		`{"type":"Speak","text":"Tail"}`, `{"type":"Flush"}`, `{"type":"Close"}`,
	}
	if diff := cmp.Diff(wantSent, sent); diff != "" {
		t.Errorf("client messages mismatch (-want +got):\n%s", diff)
	}
	u, h := srv.seen()
	wantQuery := url.Values{"model": {"aura-2-thalia-en"}, "encoding": {"mulaw"}, "sample_rate": {"8000"}, "speed": {"0.9"}}
	if diff := cmp.Diff(wantQuery, u.Query()); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
	if u.Path != "/v1/speak" || h.Get("Authorization") != "Token test-key" {
		t.Errorf("handshake = %s %v, want /v1/speak with token auth", u.Path, h)
	}
}

func TestSpeechModel_streamRejectsFileFormats(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, _ *websocket.Conn) { <-ctx.Done() })
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: func(func(string) bool) {}, Format: "mp3"})
	for range s.Audio() {
	}
	if err := s.Err(); err == nil || !strings.Contains(err.Error(), "not available over the streaming WebSocket") {
		t.Errorf("Err() = %v, want a streaming format error", err)
	}
}

func TestSpeechModel_streamErrorMessage(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		writeText(ctx, t, c, `{"type":"Error","variant":"SchemaError","description":"bad"}`)
		<-ctx.Done()
	})
	text := func(yield func(string) bool) { yield("Hello there.") }
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: text})
	for range s.Audio() {
	}
	if apiErr, ok := errors.AsType[*ai.APIError](s.Err()); !ok || apiErr.Code != "SchemaError" {
		t.Errorf("Err() = %v, want an APIError with code SchemaError", s.Err())
	}
}

func TestSpeechModel_streamFlushLimit(t *testing.T) {
	tests := []struct {
		name     string
		ackLimit int
	}{
		{name: "acks within limit", ackLimit: 20},
		{name: "warning on every flush", ackLimit: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var flushes atomic.Int64
			srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
				for f := range readFrames(ctx, c) {
					switch string(f.Data) {
					case `{"type":"Flush"}`:
						if int(flushes.Add(1)) <= tt.ackLimit {
							_ = c.Write(ctx, websocket.MessageBinary, []byte{1})
							writeText(ctx, t, c, `{"type":"Flushed","sequence_id":0}`)
						} else {
							writeText(ctx, t, c, `{"type":"Warning","warn_code":"FLUSH_LIMIT"}`)
						}
					case `{"type":"Close"}`:
						_ = c.Close(websocket.StatusNormalClosure, "")
						return
					}
				}
			})
			text := func(yield func(string) bool) {
				for range 25 {
					if !yield("This is a sentence. ") {
						return
					}
				}
			}
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			s := ai.StreamSpeech(ctx, srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: text})
			audio := 0
			for range s.Audio() {
				audio++
			}
			if err := s.Err(); err != nil {
				t.Fatalf("Err() = %v", err)
			}
			if flushes.Load() > 20 {
				t.Errorf("flushes = %d, want at most 20", flushes.Load())
			}
			if want := min(int(flushes.Load()), tt.ackLimit); audio != want {
				t.Errorf("audio chunks = %d, want %d", audio, want)
			}
		})
	}
}
