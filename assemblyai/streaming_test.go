package assemblyai_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/assemblyai"
)

func replay(t *testing.T, lines []string, audio *bytes.Buffer) func(context.Context, *websocket.Conn) {
	t.Helper()
	return func(ctx context.Context, c *websocket.Conn) {
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if typ == websocket.MessageBinary {
				audio.Write(data)
				continue
			}
			if string(data) != `{"type":"Terminate"}` {
				t.Errorf("client sent %s, want Terminate", data)
			}
			break
		}
		for _, line := range lines {
			if err := c.Write(ctx, websocket.MessageText, []byte(line)); err != nil {
				return
			}
		}
		_ = c.Close(websocket.StatusNormalClosure, "")
	}
}

func fixtureLines(tb testing.TB, name string) []string {
	tb.Helper()
	return strings.Split(strings.TrimSpace(fixture(tb, name)), "\n")
}

func TestTranscriptionModel_stream(t *testing.T) {
	var audio bytes.Buffer
	srv := newWSServer(t, replay(t, fixtureLines(t, "streaming_turns.jsonl"), &audio))
	pcm := bytes.Repeat([]byte{1, 2}, 8000)
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("universal-3-6-pro"), ai.TranscribeStreamRequest{
		Audio: bytes.NewReader(pcm), MediaType: "audio/pcm;rate=16000", Language: "en", Headers: http.Header{"X-Trace": {"1"}},
		Provider: ai.Providers(assemblyai.TranscriptionOptions{Mode: "balanced", MaxTurnSilence: 1280 * time.Millisecond, KeytermsPrompt: []string{"Zoe"}, SpeakerLabels: true}),
	})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	want := []ai.Segment{
		{Text: "hello", Start: 480 * time.Millisecond, End: 900 * time.Millisecond, Partial: true},
		{Text: "Hello there.", Start: 480 * time.Millisecond, End: 1300 * time.Millisecond, Speaker: "A"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	if !bytes.Equal(audio.Bytes(), pcm) {
		t.Errorf("server received %d audio bytes, want the %d sent", audio.Len(), len(pcm))
	}
	path, header := srv.seen()
	u, err := url.Parse(path)
	if err != nil {
		t.Fatalf("parse %q: %v", path, err)
	}
	wantQuery := url.Values{
		"encoding": {"pcm_s16le"}, "sample_rate": {"16000"}, "speech_model": {"universal-3-6-pro"}, "mode": {"balanced"}, "max_turn_silence": {"1280"},
		"keyterms_prompt": {`["Zoe"]`}, "language_codes": {`["en"]`}, "speaker_labels": {"true"},
	}
	if diff := cmp.Diff(wantQuery, u.Query()); diff != "" || u.Path != "/v3/ws" {
		t.Errorf("query mismatch (-want +got):\n%s\npath = %s", diff, u.Path)
	}
	if header.Get("Authorization") != "test-key" || header.Get("X-Trace") != "1" {
		t.Errorf("headers = %v, want the raw key and custom header", header)
	}
}

func TestTranscriptionModel_streamPadsShortTail(t *testing.T) {
	var audio bytes.Buffer
	srv := newWSServer(t, replay(t, []string{`{"type":"Termination"}`}, &audio))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeStreamRequest{Audio: strings.NewReader("ab"), MediaType: "audio/basic;rate=8000"})
	for range s.Segments() {
	}
	if err := s.Err(); err != nil || audio.Len() != 160 {
		t.Errorf("Err() = %v with %d bytes sent, want nil and one 20ms mu-law frame", err, audio.Len())
	}
	path, _ := srv.seen()
	if !strings.Contains(path, "encoding=pcm_mulaw") || !strings.Contains(path, "sample_rate=8000") || strings.Contains(path, "speech_model") {
		t.Errorf("path = %s, want mu-law at 8 kHz and no speech model", path)
	}
}

func TestTranscriptionModel_streamErrors(t *testing.T) {
	tests := []struct {
		name   string
		lines  []string
		want   string
		isAPI  bool
		reader io.Reader
	}{
		{name: "error frame", lines: []string{`{"type":"Error","error_code":3007,"error":"audio buffered too long"}`}, want: "3007", isAPI: true},
		{name: "invalid json", lines: []string{`{`}, want: "decode streaming message"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newWSServer(t, replay(t, tt.lines, &bytes.Buffer{}))
			s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeStreamRequest{Audio: strings.NewReader("")})
			for range s.Segments() {
			}
			apiErr, ok := errors.AsType[*ai.APIError](s.Err())
			if s.Err() == nil || !strings.Contains(s.Err().Error(), tt.want) || ok != tt.isAPI {
				t.Errorf("Err() = %v (api %v), want %q", s.Err(), apiErr, tt.want)
			}
		})
	}
}

func TestTranscriptionModel_streamHandshakeAuthError(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusUnauthorized, `{"error":"Unauthorized"}`)
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeStreamRequest{Audio: strings.NewReader("")})
	for range s.Segments() {
	}
	if !errors.Is(s.Err(), ai.ErrAuth) {
		t.Errorf("Err() = %v, want an authentication error", s.Err())
	}
}

func TestTranscriptionModel_streamRejectsEncodedAudio(t *testing.T) {
	s := ai.StreamTranscribe(t.Context(), assemblyai.New().TranscriptionModel(""), ai.TranscribeStreamRequest{Audio: strings.NewReader("x"), MediaType: "audio/mpeg"})
	for range s.Segments() {
	}
	if s.Err() == nil {
		t.Error("StreamTranscribe() accepted audio/mpeg, want an error")
	}
}

type endlessReader struct{ closed chan struct{} }

func (r endlessReader) Read(p []byte) (int, error) {
	select {
	case <-r.closed:
		return 0, io.EOF
	case <-time.After(time.Millisecond):
		return copy(p, make([]byte, len(p))), nil
	}
}

func (r endlessReader) Close() error {
	close(r.closed)
	return nil
}

func TestTranscriptionModel_streamStopsEarlyAndClosesAudio(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"type":"Turn","end_of_turn":true,"turn_is_formatted":true,"transcript":"hi"}`))
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	})
	audio := endlessReader{closed: make(chan struct{})}
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel(""), ai.TranscribeStreamRequest{Audio: audio})
	for seg := range s.Segments() {
		if seg.Text != "hi" {
			t.Errorf("segment = %+v, want hi", seg)
		}
		break
	}
	select {
	case <-audio.closed:
	case <-time.After(5 * time.Second):
		t.Error("audio was not closed after the loop stopped")
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil after an early stop", err)
	}
}
