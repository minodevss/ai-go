package elevenlabs_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/elevenlabs"
)

func tokens(pieces ...string) func(func(string) bool) {
	return func(yield func(string) bool) {
		for _, p := range pieces {
			if !yield(p) {
				return
			}
		}
	}
}

func TestSpeechModel_streamInput(t *testing.T) {
	lines := fixtureLines(t, "tts_stream_messages.jsonl")
	var texts []string
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		bos := readJSON(ctx, t, c)
		equalAny(t, map[string]any{"text": " ", "voice_settings": map[string]any{"stability": 0.3, "speed": 1.2}, "generation_config": map[string]any{"chunk_length_schedule": []any{50.0, 120.0}}}, bos)
		for {
			msg := readJSON(ctx, t, c)
			text, _ := msg["text"].(string)
			if text == "" {
				break
			}
			texts = append(texts, text)
		}
		for _, l := range lines {
			if err := c.Write(ctx, websocket.MessageText, []byte(l)); err != nil {
				t.Errorf("server write: %v", err)
				return
			}
		}
		_ = c.Close(websocket.StatusNormalClosure, "")
	})
	m := srv.provider().SpeechModel("eleven_flash_v2_5")
	s := ai.StreamSpeech(t.Context(), m, ai.SpeechStreamRequest{
		Text:   tokens("Hel", "lo wor", "ld, how ", "are", " you"),
		Voice:  "voice 1",
		Format: "pcm_16000",
		Speed:  new(1.2),
		Provider: ai.Providers(elevenlabs.SpeechOptions{
			VoiceSettings:       &elevenlabs.VoiceSettings{Stability: new(0.3)},
			ChunkLengthSchedule: []int{50, 120},
			InactivityTimeout:   60,
			AutoMode:            new(true),
			Seed:                new(int64(5)),
		}),
	})
	var audio []string
	for chunk := range s.Audio() {
		audio = append(audio, string(chunk))
	}
	if s.Err() != nil {
		t.Fatalf("Err() = %v", s.Err())
	}
	if diff := cmp.Diff([]string{"audio1", "audio2"}, audio); diff != "" {
		t.Errorf("audio mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"Hello ", "world, how ", "are ", "you "}, texts); diff != "" {
		t.Errorf("text messages mismatch (-want +got):\n%s", diff)
	}
	if s.MediaType() != "audio/pcm" {
		t.Errorf("MediaType() = %q, want audio/pcm", s.MediaType())
	}
	u, h := srv.seen()
	if u.EscapedPath() != "/v1/text-to-speech/voice%201/stream-input" || h.Get("Xi-Api-Key") != "test-key" {
		t.Errorf("handshake = %s key %q, want the stream-input path with the API key", u.EscapedPath(), h.Get("Xi-Api-Key"))
	}
	want := map[string]string{"model_id": "eleven_flash_v2_5", "output_format": "pcm_16000", "inactivity_timeout": "60", "auto_mode": "true", "seed": "5"}
	for k, v := range want {
		if got := u.Query().Get(k); got != v {
			t.Errorf("query %s = %q, want %q", k, got, v)
		}
	}
}

func equalAny(t *testing.T, want, got map[string]any) {
	t.Helper()
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("message mismatch (-want +got):\n%s", diff)
	}
}

func TestSpeechModel_streamInputServerError(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readJSON(ctx, t, c)
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"audio":"YXVkaW8x"}`))
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"error":"input_text_error","message":"bad text"}`))
		<-ctx.Done()
	})
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: tokens("one two "), Voice: "v"})
	chunks := 0
	for range s.Audio() {
		chunks++
	}
	apiErr, ok := errors.AsType[*ai.APIError](s.Err())
	if !ok || apiErr.Code != "input_text_error" || apiErr.Message != "bad text" || chunks != 1 {
		t.Errorf("Err() = %v after %d chunks, want an input_text_error after 1 chunk", s.Err(), chunks)
	}
}

func TestSpeechModel_streamInputHandshakeError(t *testing.T) {
	srv := newFakeServer(t, reply(401, "application/json", fixture(t, "tts_error_invalid_key.json")))
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: tokens("x"), Voice: "v"})
	for range s.Audio() {
	}
	if !errors.Is(s.Err(), ai.ErrAuth) {
		t.Errorf("Err() = %v, want an authentication error", s.Err())
	}
}

func TestSpeechModel_streamInputStopsEarly(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readJSON(ctx, t, c)
		for ctx.Err() == nil {
			if err := c.Write(ctx, websocket.MessageText, []byte(`{"audio":"YXVkaW8x"}`)); err != nil {
				return
			}
			time.Sleep(time.Millisecond)
		}
	})
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: tokens("a b "), Voice: "v"})
	n := 0
	for range s.Audio() {
		if n++; n == 3 {
			break
		}
	}
	if n != 3 || s.Err() != nil {
		t.Errorf("read %d chunks, Err() = %v, want 3 and no error", n, s.Err())
	}
}

func TestSpeechModel_streamInputDoesNotWaitForTheTextIterator(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readJSON(ctx, t, c)
		if err := c.Write(ctx, websocket.MessageText, []byte(`{"audio":"YQ==","isFinal":true}`)); err != nil {
			t.Errorf("server write: %v", err)
		}
		<-ctx.Done()
	})
	release := make(chan struct{})
	defer close(release)
	text := func(yield func(string) bool) {
		<-release
		yield("late")
	}
	s := ai.StreamSpeech(t.Context(), srv.provider().SpeechModel("m"), ai.SpeechStreamRequest{Text: text, Voice: "v", Format: "pcm_16000"})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range s.Audio() {
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("StreamSpeech blocked on a text iterator that never yields")
	}
}
