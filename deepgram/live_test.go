package deepgram_test

import (
	"context"
	"errors"
	"io"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/deepgram"
)

func TestTranscriptionModel_streamTranscribe(t *testing.T) {
	messages := fixtureLines(t, "live_messages.jsonl")
	received := make(chan []frame, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		frames := readFrames(ctx, c)
		var got []frame
		for f := range frames {
			got = append(got, f)
			if f.Kind == websocket.MessageText {
				break
			}
		}
		received <- got
		writeText(ctx, t, c, messages[:5]...)
		_ = c.Close(websocket.StatusNormalClosure, "")
	})
	m := srv.provider().TranscriptionModel("nova-3")
	opts := deepgram.TranscriptionOptions{Diarize: true, Endpointing: new(300), UtteranceEndMS: 1000, VADEvents: true}
	s := ai.StreamTranscribe(t.Context(), m, ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm-audio"), MediaType: "audio/pcm;rate=48000", Language: "en", Provider: ai.Providers(opts)})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	want := []ai.Segment{
		{Text: "hello", Partial: true, End: time.Second},
		{Text: "Hello", Speaker: "0", Start: 100 * time.Millisecond, End: 400 * time.Millisecond},
		{Text: "world.", Speaker: "1", Start: 500 * time.Millisecond, End: 900 * time.Millisecond},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	u, h := srv.seen()
	wantQuery := url.Values{"model": {"nova-3"}, "language": {"en"}, "smart_format": {"true"}, "diarize_model": {"latest"}, "encoding": {"linear16"}, "sample_rate": {"48000"}, "interim_results": {"true"}, "endpointing": {"300"}, "utterance_end_ms": {"1000"}, "vad_events": {"true"}}
	if diff := cmp.Diff(wantQuery, u.Query()); diff != "" {
		t.Errorf("query mismatch (-want +got):\n%s", diff)
	}
	if u.Path != "/v1/listen" || h.Get("Authorization") != "Token test-key" {
		t.Errorf("handshake = %s %v, want /v1/listen with token auth", u.Path, h)
	}
	frames := <-received
	if len(frames) != 2 || frames[0].Kind != websocket.MessageBinary || string(frames[0].Data) != "pcm-audio" || string(frames[1].Data) != `{"type":"CloseStream"}` {
		t.Errorf("client frames = %+v, want the audio then CloseStream", frames)
	}
}

func TestTranscriptionModel_streamErrorMessage(t *testing.T) {
	messages := fixtureLines(t, "live_messages.jsonl")
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		writeText(ctx, t, c, messages[len(messages)-1])
		<-ctx.Done()
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeStreamRequest{Audio: blockingReader{t.Context()}})
	for range s.Segments() {
		t.Error("unexpected segment")
	}
	apiErr, ok := errors.AsType[*ai.APIError](s.Err())
	if !ok || apiErr.Code != "SchemaError" || apiErr.Message != "bad message" {
		t.Errorf("Err() = %v, want an APIError with code SchemaError", s.Err())
	}
}

func TestTranscriptionModel_streamSendsKeepAliveWhileIdle(t *testing.T) {
	seen := make(chan string, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		for f := range readFrames(ctx, c) {
			if f.Kind == websocket.MessageText {
				seen <- string(f.Data)
				break
			}
		}
		_ = c.Close(websocket.StatusNormalClosure, "")
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeStreamRequest{Audio: blockingReader{t.Context()}, Provider: ai.Providers(deepgram.TranscriptionOptions{KeepAlive: 5 * time.Millisecond})})
	for range s.Segments() {
	}
	select {
	case got := <-seen:
		if got != `{"type":"KeepAlive"}` {
			t.Errorf("first text frame = %s, want KeepAlive", got)
		}
	default:
		t.Error("server saw no KeepAlive")
	}
}

func TestTranscriptionModel_streamHandshakeFailure(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"err_code":"INVALID_AUTH","err_msg":"no"}`))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeStreamRequest{Audio: strings.NewReader("x")})
	for range s.Segments() {
	}
	if err := s.Err(); err == nil || !strings.HasPrefix(err.Error(), "ai: deepgram:") {
		t.Errorf("Err() = %v, want a deepgram-prefixed error", err)
	}
}

func TestTranscriptionModel_streamStopsWhenTheConsumerBreaks(t *testing.T) {
	messages := fixtureLines(t, "live_messages.jsonl")
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		writeText(ctx, t, c, messages[2], messages[3])
		<-ctx.Done()
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("nova-3"), ai.TranscribeStreamRequest{Audio: blockingReader{t.Context()}})
	n := 0
	for range s.Segments() {
		n++
		break
	}
	if n != 1 || s.Err() != nil {
		t.Errorf("segments = %d, Err() = %v, want 1 and nil", n, s.Err())
	}
}

type blockingReader struct{ ctx context.Context }

func (r blockingReader) Read([]byte) (int, error) {
	<-r.ctx.Done()
	return 0, io.EOF
}
