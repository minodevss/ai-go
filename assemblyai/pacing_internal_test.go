package assemblyai

import (
	"bytes"
	"context"
	"testing"
	"testing/synctest"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
)

func TestPumpAudio_pacesToRealTime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		format, err := formatOf("audio/pcm;rate=16000")
		if err != nil {
			t.Fatalf("formatOf() error = %v", err)
		}
		const seconds = 60
		audio := bytes.NewReader(make([]byte, format.byteRate*seconds))
		start := time.Now()
		var sent time.Duration
		var worstLead time.Duration
		write := func(_ context.Context, typ websocket.MessageType, data []byte) error {
			if typ != websocket.MessageBinary {
				return nil
			}
			sent += format.duration(len(data))
			worstLead = max(worstLead, sent-time.Since(start))
			return nil
		}
		if err := pumpAudio(t.Context(), audio, format, write); err != nil {
			t.Fatalf("pumpAudio() error = %v", err)
		}
		if worstLead > maxAudioLead+audioChunk {
			t.Errorf("audio ran %s ahead of real time, want at most %s", worstLead, maxAudioLead+audioChunk)
		}
		if elapsed := time.Since(start); elapsed < seconds*time.Second-maxAudioLead-audioChunk {
			t.Errorf("a minute of audio was sent in %s", elapsed)
		}
	})
}

func TestSegmentsOf_withoutUtterances(t *testing.T) {
	tests := []struct {
		name    string
		body    transcriptBody
		perWord bool
		want    []ai.Segment
	}{
		{name: "words give timing", body: transcriptBody{Text: "Hello there.", Words: []wordBody{{Text: "Hello", Start: 240, End: 600}, {Text: "there.", Start: 700, End: 1100}}}, want: []ai.Segment{{Text: "Hello there.", Start: 240 * time.Millisecond, End: 1100 * time.Millisecond}}},
		{name: "text only", body: transcriptBody{Text: "Hello there."}, want: []ai.Segment{{Text: "Hello there."}}},
		{name: "per word without words", body: transcriptBody{Text: "Hello"}, perWord: true, want: []ai.Segment{{Text: "Hello"}}},
		{name: "empty", body: transcriptBody{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, segmentsOf(&tt.body, tt.perWord)); diff != "" {
				t.Errorf("segmentsOf() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
