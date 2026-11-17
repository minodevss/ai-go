package google

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
)

func mediaSeeds(tb testing.TB, prefix string) [][]byte {
	tb.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "media", prefix+"*.json"))
	if err != nil {
		tb.Fatalf("glob fixtures: %v", err)
	}
	seeds := [][]byte{nil, []byte(`{}`), []byte(`null`), []byte(`[`)}
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			tb.Fatalf("read fixture: %v", err)
		}
		seeds = append(seeds, raw)
	}
	return seeds
}

func testBackend() *backend { return &backend{name: providerName} }

func FuzzImageResponse(f *testing.F) {
	for _, seed := range mediaSeeds(f, "image_") {
		f.Add(seed)
	}
	m := &ImageModel{backend: testBackend(), id: "m"}
	f.Fuzz(func(t *testing.T, data []byte) {
		var body responseBody
		if json.Unmarshal(data, &body) != nil {
			return
		}
		images, err := m.images(body)
		if err != nil && len(images) != 0 {
			t.Fatalf("images returned %d images together with error %v", len(images), err)
		}
		for _, img := range images {
			if !strings.HasPrefix(img.MediaType, "image/") {
				t.Fatalf("image media type = %q", img.MediaType)
			}
		}
	})
}

func FuzzVeoOperation(f *testing.F) {
	for _, seed := range mediaSeeds(f, "veo_") {
		f.Add(seed)
	}
	m := &VideoModel{backend: testBackend(), id: "m"}
	f.Fuzz(func(t *testing.T, data []byte) {
		var op veoOperation
		if json.Unmarshal(data, &op) != nil {
			return
		}
		poll, err := m.veoPoll(op, http.Header{})
		if err != nil {
			return
		}
		if poll.Done && (len(poll.Videos) == 0 || poll.Failure != nil) {
			t.Fatalf("done poll = %+v, want videos and no failure", poll)
		}
	})
}

func FuzzOmniInteraction(f *testing.F) {
	for _, seed := range mediaSeeds(f, "omni_") {
		f.Add(seed)
	}
	m := &VideoModel{backend: testBackend(), id: "m"}
	f.Fuzz(func(t *testing.T, data []byte) {
		var it omniInteraction
		if json.Unmarshal(data, &it) != nil {
			return
		}
		poll, err := m.omniPoll(t.Context(), it, &ai.VideoJob{ID: "j"}, http.Header{})
		if err != nil {
			return
		}
		if poll.Done && (len(poll.Videos) == 0 || poll.Failure != nil) {
			t.Fatalf("done poll = %+v, want videos and no failure", poll)
		}
	})
}

func FuzzSpeechAudio(f *testing.F) {
	for _, seed := range mediaSeeds(f, "tts_") {
		f.Add(seed, true)
		f.Add(seed, false)
	}
	f.Fuzz(func(t *testing.T, data []byte, legacy bool) {
		var body responseBody
		if json.Unmarshal(data, &body) != nil {
			return
		}
		id := "gemini-3.8-flash-tts"
		if legacy {
			id = "gemini-2.5-flash-preview-tts"
		}
		m := &SpeechModel{backend: testBackend(), id: id}
		audio, err := m.audio(body, "")
		if err == nil && !strings.HasPrefix(audio.MediaType, "audio/") {
			t.Fatalf("audio media type = %q", audio.MediaType)
		}
	})
}

func FuzzWrapWAV(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, "audio/L16;codec=pcm;rate=24000")
	f.Add([]byte(nil), "")
	f.Add([]byte{0}, "audio/l16;rate=-5")
	f.Fuzz(func(t *testing.T, pcm []byte, mediaType string) {
		out := wrapWAV(pcm, rateOf(mediaType))
		if len(out) != wavHeaderBytes+len(pcm) || string(out[:4]) != "RIFF" {
			t.Fatalf("wrapWAV() = %d bytes, want %d with a RIFF header", len(out), wavHeaderBytes+len(pcm))
		}
	})
}
