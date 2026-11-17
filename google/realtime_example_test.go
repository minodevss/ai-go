package google_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
	"github.com/minodevss/ai-go/realtime"
)

func ExampleRealtimeModel_MintClientSecret() {
	srv := exampleServer(map[string]string{"/v1beta/auth_tokens": `{"name":"auth_tokens/abc123"}`})
	defer srv.Close()
	p := google.New(google.WithAPIKey("key"), google.WithBaseURL(srv.URL+"/v1beta"))

	cfg := realtime.Config{Instructions: "Greet the visitor.", Voice: "Kore"}
	secret, err := realtime.MintClientSecret(context.Background(), p.RealtimeModel("gemini-3.8-live"), cfg, realtime.MintOptions{TTL: time.Minute})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(secret.Value, secret.Locked)
	// Output: auth_tokens/abc123 true
}

func ExampleRealtimeModel_Connect() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow() //nolint:errcheck // the example server ends with the connection
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		for _, msg := range []string{
			`{"setupComplete":{}}`,
			`{"serverContent":{"modelTurn":{"parts":[{"text":"Hi there"}]}}}`,
			`{"serverContent":{"turnComplete":true}}`,
		} {
			if err := c.Write(ctx, websocket.MessageText, []byte(msg)); err != nil {
				return
			}
		}
		c.Read(ctx) //nolint:errcheck // waits for the client to close
	}))
	defer srv.Close()
	p := google.New(google.WithAPIKey("key"), google.WithBaseURL(srv.URL+"/v1beta"))

	ctx := context.Background()
	sess, err := realtime.Connect(ctx, p.RealtimeModel("gemini-3.8-live"), realtime.Config{
		Instructions: "Answer in one sentence.",
		Provider:     ai.Providers(google.RealtimeOptions{ResponseModalities: []string{"TEXT"}}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()
	for ev := range sess.Events() {
		switch ev := ev.(type) {
		case realtime.TextDelta:
			fmt.Println(ev.Text)
		case realtime.TurnEnd:
			return
		}
	}
	// Output: Hi there
}
