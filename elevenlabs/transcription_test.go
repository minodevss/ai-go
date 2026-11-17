package elevenlabs_test

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/elevenlabs"
)

func TestTranscriptionModel_diarizedWords(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusOK, "application/json", fixture(t, "stt_response.json")))
	m := srv.provider().TranscriptionModel("scribe_v2")
	res, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{
		Audio:     strings.NewReader("audio"),
		MediaType: "audio/wav",
		Language:  "en",
		Provider: ai.Providers(elevenlabs.TranscriptionOptions{
			Diarize:               new(true),
			NumSpeakers:           2,
			TimestampsGranularity: "word",
			TagAudioEvents:        new(false),
			Keyterms:              []string{"Scribe", "ElevenLabs"},
			Temperature:           new(0.0),
			Seed:                  new(int64(3)),
			EnableLogging:         new(false),
		}),
	})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	wantSegments := []ai.Segment{
		{Text: "Hello world.", Start: 0, End: time.Second, Speaker: "speaker_0"},
		{Text: "How are you?", Start: 1200 * time.Millisecond, End: 2200 * time.Millisecond, Speaker: "speaker_1"},
	}
	if diff := cmp.Diff(wantSegments, res.Segments); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	if res.Text != "Hello world. How are you?" || res.Language != "en" {
		t.Errorf("text = %q language = %q, want the full text in en", res.Text, res.Language)
	}
	got := srv.last(t)
	if got.Path != "/v1/speech-to-text" || got.Header.Get("Xi-Api-Key") != "test-key" || got.Query.Get("enable_logging") != "false" {
		t.Errorf("request = %s key %q query %v, want /v1/speech-to-text", got.Path, got.Header.Get("Xi-Api-Key"), got.Query)
	}
	fields, file := formFields(t, got)
	want := map[string][]string{
		"model_id": {"scribe_v2"}, "language_code": {"en"}, "diarize": {"true"}, "num_speakers": {"2"},
		"timestamps_granularity": {"word"}, "tag_audio_events": {"false"}, "keyterms": {"Scribe", "ElevenLabs"},
		"temperature": {"0"}, "seed": {"3"},
	}
	if diff := cmp.Diff(want, fields); diff != "" {
		t.Errorf("form fields mismatch (-want +got):\n%s", diff)
	}
	if string(file) != "audio" {
		t.Errorf("file = %q, want audio", file)
	}
}

func TestTranscriptionModel_multichannel(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusOK, "application/json", fixture(t, "stt_multichannel.json")))
	res, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeRequest{
		Audio:    strings.NewReader("audio"),
		Provider: ai.Providers(elevenlabs.TranscriptionOptions{UseMultiChannel: new(true)}),
	})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	want := []ai.Segment{
		{Text: "Hi there.", Start: 0, End: 800 * time.Millisecond, Speaker: "channel_0"},
		{Text: "Hello.", Start: 900 * time.Millisecond, End: 1300 * time.Millisecond, Speaker: "channel_1"},
	}
	if diff := cmp.Diff(want, res.Segments); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	if res.Text != "Hi there.\nHello." {
		t.Errorf("text = %q, want the channel texts joined", res.Text)
	}
	fields, _ := formFields(t, srv.last(t))
	if fields["use_multi_channel"][0] != "true" {
		t.Errorf("use_multi_channel = %v, want true", fields["use_multi_channel"])
	}
}

func TestTranscriptionModel_error(t *testing.T) {
	srv := newFakeServer(t, reply(http.StatusUnauthorized, "application/json", fixture(t, "tts_error_invalid_key.json")))
	_, err := ai.Transcribe(t.Context(), srv.provider().TranscriptionModel("scribe_v2"), ai.TranscribeRequest{Audio: strings.NewReader("a"), MaxRetries: new(0)})
	if err == nil || !strings.Contains(err.Error(), "invalid_api_key") {
		t.Errorf("Transcribe() error = %v, want invalid_api_key", err)
	}
}
