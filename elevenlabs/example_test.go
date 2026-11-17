package elevenlabs_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/elevenlabs"
)

func exampleServer(routes map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, ok := routes[r.Method+" "+r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = io.WriteString(w, body)
	}))
}

func ExampleProvider_SpeechModel() {
	srv := exampleServer(map[string]string{"POST /v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb": "audio-bytes"})
	defer srv.Close()
	p := elevenlabs.New(elevenlabs.WithAPIKey("xi-test"), elevenlabs.WithBaseURL(srv.URL))

	res, err := ai.GenerateSpeech(context.Background(), p.SpeechModel("eleven_multilingual_v2"), ai.SpeechRequest{
		Text:     "Hello from ai-go.",
		Voice:    "JBFqnCBsd6RMkjVDRZzb",
		Format:   "mp3_44100_128",
		Provider: ai.Providers(elevenlabs.SpeechOptions{VoiceSettings: &elevenlabs.VoiceSettings{Stability: new(0.5)}}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Audio.Data), res.Audio.MediaType)
	// Output: audio-bytes audio/mpeg
}

func ExampleSpeechModel_StreamSpeech() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			_, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			var msg struct {
				Text string `json:"text"`
			}
			if json.Unmarshal(data, &msg) == nil && msg.Text == "" {
				break
			}
		}
		_ = c.Write(r.Context(), websocket.MessageText, []byte(`{"audio":"YXVkaW8="}`))
		_ = c.Write(r.Context(), websocket.MessageText, []byte(`{"isFinal":true}`))
	}))
	defer srv.Close()
	p := elevenlabs.New(elevenlabs.WithAPIKey("xi-test"), elevenlabs.WithBaseURL(srv.URL))

	text := func(yield func(string) bool) {
		_ = yield("Streaming ") && yield("text ") && yield("in.")
	}
	s := ai.StreamSpeech(context.Background(), p.SpeechModel("eleven_flash_v2_5"), ai.SpeechStreamRequest{Text: text, Voice: "JBFqnCBsd6RMkjVDRZzb", Format: "mp3_44100_64"})
	for chunk := range s.Audio() {
		fmt.Println(string(chunk))
	}
	fmt.Println(s.Err())
	// Output:
	// audio
	// <nil>
}

func ExampleProvider_TranscriptionModel() {
	srv := exampleServer(map[string]string{"POST /v1/speech-to-text": `{"language_code":"en","text":"Hi there.","words":[{"text":"Hi","type":"word","start":0,"end":0.3,"speaker_id":"speaker_0"},{"text":" ","type":"spacing"},{"text":"there.","type":"word","start":0.4,"end":0.8,"speaker_id":"speaker_0"}]}`})
	defer srv.Close()
	p := elevenlabs.New(elevenlabs.WithAPIKey("xi-test"), elevenlabs.WithBaseURL(srv.URL))

	res, err := ai.Transcribe(context.Background(), p.TranscriptionModel("scribe_v2"), ai.TranscribeRequest{
		Audio:     strings.NewReader("audio"),
		MediaType: "audio/mpeg",
		Provider:  ai.Providers(elevenlabs.TranscriptionOptions{Diarize: new(true)}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Segments[0].Speaker, res.Segments[0].Text)
	// Output: speaker_0 Hi there.
}

func ExampleTranscriptionModel_StreamTranscribe() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			_, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			var msg struct {
				Commit bool `json:"commit"`
			}
			if json.Unmarshal(data, &msg) == nil && msg.Commit {
				break
			}
		}
		_ = c.Write(r.Context(), websocket.MessageText, []byte(`{"message_type":"partial_transcript","text":"hel"}`))
		_ = c.Write(r.Context(), websocket.MessageText, []byte(`{"message_type":"committed_transcript","text":"Hello."}`))
		_, _, _ = c.Read(r.Context())
	}))
	defer srv.Close()
	p := elevenlabs.New(elevenlabs.WithAPIKey("xi-test"), elevenlabs.WithBaseURL(srv.URL))

	s := ai.StreamTranscribe(context.Background(), p.TranscriptionModel("scribe_v2"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm-audio"), MediaType: "audio/pcm;rate=16000"})
	for seg := range s.Segments() {
		fmt.Println(seg.Text, seg.Partial)
	}
	fmt.Println(s.Err())
	// Output:
	// hel true
	// Hello. false
	// <nil>
}

func ExampleNew() {
	p := elevenlabs.New(elevenlabs.WithAPIKey("xi-key"))
	fmt.Println(p.SpeechModel("eleven_multilingual_v2").ModelID())
	// Output: eleven_multilingual_v2
}

func ExampleNewSpeechModel() {
	m := elevenlabs.NewSpeechModel("eleven_multilingual_v2")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: elevenlabs eleven_multilingual_v2
}

func ExampleNewTranscriptionModel() {
	m := elevenlabs.NewTranscriptionModel("scribe_v2")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: elevenlabs scribe_v2
}
