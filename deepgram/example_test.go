package deepgram_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/deepgram"
)

func ExampleTranscriptionModel_Transcribe() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"results":{"channels":[{"alternatives":[{"transcript":"Hello world."}]}],"utterances":[{"start":0.1,"end":1.2,"speaker":0,"transcript":"Hello world."}]}}`)
	}))
	defer srv.Close()
	p := deepgram.New(deepgram.WithAPIKey("dg-test"), deepgram.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.Transcribe(context.Background(), p.TranscriptionModel("nova-3"), ai.TranscribeRequest{
		Audio:     strings.NewReader("audio"),
		MediaType: "audio/wav",
		Language:  "en",
		Provider:  ai.Providers(deepgram.TranscriptionOptions{Diarize: true}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text, res.Segments[0].Speaker, res.Segments[0].End)
	// Output: Hello world. 0 1.2s
}

func ExampleTranscriptionModel_StreamTranscribe() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		for _, msg := range []string{
			`{"type":"Results","start":0,"duration":0.5,"is_final":false,"channel":{"alternatives":[{"transcript":"hello"}]}}`,
			`{"type":"Results","start":0,"duration":1,"is_final":true,"channel":{"alternatives":[{"transcript":"hello world"}]}}`,
		} {
			if err := c.Write(ctx, websocket.MessageText, []byte(msg)); err != nil {
				return
			}
		}
		_ = c.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()
	p := deepgram.New(deepgram.WithAPIKey("dg-test"), deepgram.WithBaseURL(srv.URL+"/v1"))

	s := ai.StreamTranscribe(context.Background(), p.TranscriptionModel("nova-3"), ai.TranscribeStreamRequest{
		Audio:     strings.NewReader("pcm"),
		MediaType: "audio/pcm;rate=16000",
	})
	for seg := range s.Segments() {
		fmt.Println(seg.Text, seg.Partial)
	}
	fmt.Println(s.Err())
	// Output:
	// hello true
	// hello world false
	// <nil>
}

func ExampleSpeechModel_GenerateSpeech() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "mp3-bytes")
	}))
	defer srv.Close()
	p := deepgram.New(deepgram.WithAPIKey("dg-test"), deepgram.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateSpeech(context.Background(), p.SpeechModel("aura-2-thalia-en"), ai.SpeechRequest{Text: "Welcome to Deepgram."})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Audio.Data), res.Audio.MediaType)
	// Output: mp3-bytes audio/mpeg
}

func ExampleSpeechModel_StreamSpeech() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			switch {
			case strings.Contains(string(data), `"Flush"`):
				if c.Write(ctx, websocket.MessageBinary, []byte("pcm")) != nil || c.Write(ctx, websocket.MessageText, []byte(`{"type":"Flushed","sequence_id":0}`)) != nil {
					return
				}
			case strings.Contains(string(data), `"Close"`):
				_ = c.Close(websocket.StatusNormalClosure, "")
				return
			}
		}
	}))
	defer srv.Close()
	p := deepgram.New(deepgram.WithAPIKey("dg-test"), deepgram.WithBaseURL(srv.URL+"/v1"))

	text := func(yield func(string) bool) {
		for _, piece := range []string{"Hello", " there."} {
			if !yield(piece) {
				return
			}
		}
	}
	s := ai.StreamSpeech(context.Background(), p.SpeechModel("aura-2-thalia-en"), ai.SpeechStreamRequest{Text: text})
	for chunk := range s.Audio() {
		fmt.Println(string(chunk))
	}
	fmt.Println(s.MediaType(), s.Err())
	// Output:
	// pcm
	// audio/pcm <nil>
}

func ExampleNew() {
	p := deepgram.New(deepgram.WithAPIKey("dg-key"))
	fmt.Println(p.SpeechModel("aura-2-thalia-en").ModelID())
	// Output: aura-2-thalia-en
}

func ExampleNewSpeechModel() {
	m := deepgram.NewSpeechModel("aura-2-thalia-en")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: deepgram aura-2-thalia-en
}

func ExampleNewTranscriptionModel() {
	m := deepgram.NewTranscriptionModel("nova-3")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: deepgram nova-3
}

func ExampleProvider_TranscriptionModel() {
	p := deepgram.New(deepgram.WithAPIKey("dg-key"), deepgram.WithBaseURL("https://api.eu.deepgram.com/v1"))
	m := p.TranscriptionModel("nova-3")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: deepgram nova-3
}

func ExampleProvider_SpeechModel() {
	p := deepgram.New(deepgram.WithAPIKey("dg-key"))
	m := p.SpeechModel("aura-2-thalia-en")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: deepgram aura-2-thalia-en
}
