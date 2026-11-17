package assemblyai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/assemblyai"
)

func ExampleTranscriptionModel_Transcribe() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v2/upload":
			_, _ = io.WriteString(w, `{"upload_url":"https://cdn.assemblyai.com/upload/1"}`)
		case r.Method == http.MethodPost:
			_, _ = io.WriteString(w, `{"id":"t1","status":"queued"}`)
		default:
			_, _ = io.WriteString(w, `{"id":"t1","status":"completed","text":"Hello there.","language_code":"en_us","utterances":[{"text":"Hello there.","start":240,"end":1100,"speaker":"A"}]}`)
		}
	}))
	defer srv.Close()
	p := assemblyai.New(assemblyai.WithAPIKey("test-key"), assemblyai.WithBaseURL(srv.URL))

	res, err := ai.Transcribe(context.Background(), p.TranscriptionModel("universal-3-pro"), ai.TranscribeRequest{
		Audio:    strings.NewReader("audio"),
		Provider: ai.Providers(assemblyai.TranscriptionOptions{SpeakerLabels: true, PollInterval: time.Millisecond}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Segments[0].Speaker, res.Segments[0].Text, res.Segments[0].End)
	// Output: A Hello there. 1.1s
}

func ExampleTranscriptionModel_StreamTranscribe() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		for {
			typ, _, err := c.Read(ctx)
			if err != nil {
				return
			}
			if typ == websocket.MessageText {
				break
			}
		}
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"type":"Turn","turn_order":0,"end_of_turn":false,"turn_is_formatted":false,"transcript":"hello","words":[]}`))
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"type":"Turn","turn_order":0,"end_of_turn":true,"turn_is_formatted":true,"transcript":"Hello.","words":[]}`))
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"type":"Termination","audio_duration_seconds":1,"session_duration_seconds":1}`))
		_ = c.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()
	p := assemblyai.New(assemblyai.WithAPIKey("test-key"), assemblyai.WithStreamingURL(srv.URL))

	s := ai.StreamTranscribe(context.Background(), p.TranscriptionModel("universal-3-6-pro"), ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/pcm;rate=16000"})
	for seg := range s.Segments() {
		fmt.Println(seg.Text, seg.Partial)
	}
	fmt.Println(s.Err())
	// Output:
	// hello true
	// Hello. false
	// <nil>
}

func ExampleProvider_MintStreamingToken() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"token":"temp-token-123","expires_in_seconds":60}`)
	}))
	defer srv.Close()
	p := assemblyai.New(assemblyai.WithAPIKey("test-key"), assemblyai.WithTokenURL(srv.URL))

	tok, err := p.MintStreamingToken(context.Background(), assemblyai.TokenRequest{ExpiresIn: time.Minute})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tok.Value)
	// Output: temp-token-123
}

func ExampleNew() {
	p := assemblyai.New(assemblyai.WithAPIKey("aai-key"))
	fmt.Println(p.TranscriptionModel("universal").ModelID())
	// Output: universal
}

func ExampleNewTranscriptionModel() {
	m := assemblyai.NewTranscriptionModel("universal")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: assemblyai universal
}

func ExampleProvider_TranscriptionModel() {
	p := assemblyai.New(assemblyai.WithAPIKey("aai-key"), assemblyai.WithRegion(assemblyai.RegionEU))
	m := p.TranscriptionModel("universal")
	fmt.Println(m.Provider(), m.ModelID())
	// Output: assemblyai universal
}
