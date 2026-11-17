package elevenlabs_test

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/elevenlabs"
)

func serveLive(t *testing.T, lines []string) (*wsServer, *[]map[string]any) {
	t.Helper()
	var received []map[string]any
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		for {
			msg := readJSON(ctx, t, c)
			received = append(received, msg)
			if commit, _ := msg["commit"].(bool); commit {
				break
			}
		}
		for _, l := range lines {
			if err := c.Write(ctx, websocket.MessageText, []byte(l)); err != nil {
				return
			}
		}
		<-ctx.Done()
	})
	return srv, &received
}

func TestTranscriptionModel_streamLive(t *testing.T) {
	lines := fixtureLines(t, "stt_live_messages.jsonl")[:5]
	srv, received := serveLive(t, lines)
	audio := strings.NewReader(strings.Repeat("ab", 6400))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeStreamRequest{
		Audio:     audio,
		MediaType: "audio/pcm;rate=16000",
		Language:  "en",
		Provider:  ai.Providers(elevenlabs.TranscriptionOptions{IncludeTimestamps: new(true), Diarize: new(true), CommitStrategy: "manual"}),
	})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if s.Err() != nil {
		t.Fatalf("Err() = %v", s.Err())
	}
	want := []ai.Segment{
		{Text: "hel", Partial: true},
		{Text: "Hello there.", Start: 100 * time.Millisecond, End: time.Second, Speaker: "speaker_0"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	u, h := srv.seen()
	if u.Path != "/v1/speech-to-text/realtime" || h.Get("Xi-Api-Key") != "test-key" {
		t.Errorf("handshake = %s key %q, want the realtime path with the API key", u.Path, h.Get("Xi-Api-Key"))
	}
	wantQuery := map[string]string{"model_id": "scribe_v2_realtime", "audio_format": "pcm_16000", "language_code": "en", "commit_strategy": "manual", "include_timestamps": "true", "diarize": "true"}
	for k, v := range wantQuery {
		if u.Query().Get(k) != v {
			t.Errorf("query %s = %q, want %q", k, u.Query().Get(k), v)
		}
	}
	chunks := *received
	if len(chunks) != 4 {
		t.Fatalf("received %d chunks, want 4 of 3200 bytes", len(chunks))
	}
	for i, c := range chunks {
		encoded, _ := c["audio_base_64"].(string)
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(raw) != 3200 || c["message_type"] != "input_audio_chunk" || c["sample_rate"] != 16000.0 {
			t.Errorf("chunk %d = %d bytes %v, want a 3200 byte input_audio_chunk at 16000", i, len(raw), err)
		}
		if commit, _ := c["commit"].(bool); commit != (i == len(chunks)-1) {
			t.Errorf("chunk %d commit = %v, want only the last chunk to commit", i, commit)
		}
	}
}

func TestTranscriptionModel_streamLiveWithoutTimestamps(t *testing.T) {
	srv, _ := serveLive(t, []string{`{"message_type":"partial_transcript","text":"a"}`, `{"message_type":"committed_transcript","text":"All done."}`})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v3"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/pcm"})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if diff := cmp.Diff([]ai.Segment{{Text: "a", Partial: true}, {Text: "All done."}}, got); diff != "" || s.Err() != nil {
		t.Errorf("segments mismatch (-want +got):\n%s\nErr() = %v", diff, s.Err())
	}
	if u, _ := srv.seen(); u.Query().Get("model_id") != "scribe_v3_live" || u.Query().Get("commit_strategy") != "vad" {
		t.Errorf("query = %v, want scribe_v3_live with vad commits", u.Query())
	}
}

func TestTranscriptionModel_streamLiveServerError(t *testing.T) {
	lines := fixtureLines(t, "stt_live_messages.jsonl")
	srv, _ := serveLive(t, []string{lines[5]})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/pcm"})
	for range s.Segments() {
	}
	if !errors.Is(s.Err(), ai.ErrAuth) {
		t.Errorf("Err() = %v, want an authentication error", s.Err())
	}
}

func TestTranscriptionModel_streamLiveRejectsCompressedAudio(t *testing.T) {
	srv := newFakeServer(t, reply(200, "text/plain", ""))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeStreamRequest{Audio: strings.NewReader("x"), MediaType: "audio/mpeg"})
	for range s.Segments() {
	}
	if s.Err() == nil || !strings.Contains(s.Err().Error(), "raw PCM") {
		t.Errorf("Err() = %v, want an error asking for raw PCM", s.Err())
	}
	if len(srv.requests) != 0 {
		t.Errorf("sent %d requests, want none", len(srv.requests))
	}
}

func TestTranscriptionModel_streamLiveStopsEarly(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		for ctx.Err() == nil {
			if err := c.Write(ctx, websocket.MessageText, []byte(`{"message_type":"partial_transcript","text":"x"}`)); err != nil {
				return
			}
			time.Sleep(time.Millisecond)
		}
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/pcm"})
	n := 0
	for range s.Segments() {
		if n++; n == 2 {
			break
		}
	}
	if n != 2 || s.Err() != nil {
		t.Errorf("read %d segments, Err() = %v, want 2 and no error", n, s.Err())
	}
}

func TestTranscriptionModel_streamLiveWaitsForTheCommitAfterAutoCommits(t *testing.T) {
	srv, _ := serveLive(t, []string{
		`{"message_type":"committed_transcript","text":"Earlier speech."}`,
		`{"message_type":"committed_transcript","text":"Final words."}`,
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("scribe_v3"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/pcm"})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if diff := cmp.Diff([]ai.Segment{{Text: "Earlier speech."}, {Text: "Final words."}}, got); diff != "" || s.Err() != nil {
		t.Errorf("segments mismatch (-want +got):\n%s\nErr() = %v", diff, s.Err())
	}
}
