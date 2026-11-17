package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
	"github.com/minodevss/ai-go/realtime"
)

type liveServer struct {
	*httptest.Server
	mu      sync.Mutex
	path    string
	headers http.Header
}

func newLiveServer(t *testing.T, script func(ctx context.Context, c *websocket.Conn)) *liveServer {
	t.Helper()
	s := &liveServer{}
	var handlers sync.WaitGroup
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers.Add(1)
		defer handlers.Done()
		s.mu.Lock()
		s.path, s.headers = r.URL.Path, r.Header.Clone()
		s.mu.Unlock()
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		script(t.Context(), c)
	}))
	t.Cleanup(func() {
		s.Close()
		handlers.Wait()
	})
	return s
}

func (s *liveServer) model(opts ...google.Option) *google.RealtimeModel {
	p := google.New(append([]google.Option{google.WithAPIKey("test-key"), google.WithBaseURL(s.URL + "/v1beta")}, opts...)...)
	return p.RealtimeModel("gemini-3.8-live")
}

func (s *liveServer) seen() (string, http.Header) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.path, s.headers
}

func readJSON(ctx context.Context, t *testing.T, c *websocket.Conn) map[string]any {
	t.Helper()
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Errorf("server read: %v", err)
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Errorf("client sent invalid JSON %q: %v", data, err)
	}
	return m
}

func send(ctx context.Context, t *testing.T, c *websocket.Conn, messages ...string) {
	t.Helper()
	for _, m := range messages {
		if err := c.Write(ctx, websocket.MessageBinary, []byte(m)); err != nil {
			t.Errorf("server write: %v", err)
			return
		}
	}
}

func liveFixture(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "realtime", "server_messages.jsonl"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

func collect(t *testing.T, conn realtime.Conn, want int) []realtime.Event {
	t.Helper()
	var got []realtime.Event
	for ev, err := range conn.Events() {
		if err != nil {
			t.Fatalf("Events() error = %v", err)
		}
		got = append(got, ev)
		if len(got) == want {
			break
		}
	}
	return got
}

func TestRealtimeModel_connectSendsSetupAndWaitsForSetupComplete(t *testing.T) {
	setups := make(chan map[string]any, 1)
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		setups <- readJSON(ctx, t, c)
		send(ctx, t, c, `{"setupComplete":{}}`)
		_, _, _ = c.Read(ctx)
	})
	cfg := realtime.Config{
		Instructions:  "Be brief.",
		Voice:         "Kore",
		Resume:        "handle-0",
		TurnDetection: realtime.TurnDetection{Mode: realtime.TurnServerVAD, Threshold: new(0.8), SilenceDuration: 700 * time.Millisecond, Eagerness: "low"},
		Tools:         []ai.Tool{ai.NewClientTool[struct{}]("turn_on_the_lights", "Turn the lights on."), google.Search()},
	}
	conn, err := srv.model().Connect(t.Context(), &cfg)
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	defer conn.Close()
	path, header := srv.seen()
	if want := "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"; path != want {
		t.Errorf("path = %q, want %q", path, want)
	}
	if got := header.Get("X-Goog-Api-Key"); got != "test-key" {
		t.Errorf("x-goog-api-key = %q, want test-key", got)
	}
	want := `{"setup":{
		"model":"models/gemini-3.8-live",
		"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}},
		"systemInstruction":{"parts":[{"text":"Be brief."}]},
		"tools":[{"googleSearch":{}},{"functionDeclarations":[{"name":"turn_on_the_lights","description":"Turn the lights on."}]}],
		"realtimeInputConfig":{"automaticActivityDetection":{"startOfSpeechSensitivity":"START_SENSITIVITY_LOW","endOfSpeechSensitivity":"END_SENSITIVITY_LOW","silenceDurationMs":700}},
		"sessionResumption":{"handle":"handle-0"},
		"inputAudioTranscription":{},
		"outputAudioTranscription":{}
	}}`
	var wantMap map[string]any
	if err := json.Unmarshal([]byte(want), &wantMap); err != nil {
		t.Fatal(err)
	}
	got := <-setups
	if diff := cmp.Diff(wantMap, got); diff != "" {
		t.Errorf("setup mismatch (-want +got):\n%s", diff)
	}
}

func TestRealtimeModel_connectFailsWithoutSetupComplete(t *testing.T) {
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = readJSON(ctx, t, c)
		send(ctx, t, c, `{"goAway":{"timeLeft":"1s"}}`)
	})
	if _, err := srv.model().Connect(t.Context(), &realtime.Config{}); err == nil {
		t.Error("Connect() error = nil, want an error when setupComplete is missing")
	}
}

func TestRealtimeModel_connectRejects(t *testing.T) {
	srv := newLiveServer(t, func(context.Context, *websocket.Conn) {})
	tests := []struct {
		name  string
		model *google.RealtimeModel
		cfg   realtime.Config
	}{
		{name: "manual turns", model: srv.model(), cfg: realtime.Config{TurnDetection: realtime.TurnDetection{Mode: realtime.TurnManual}}},
		{name: "ulaw input", model: srv.model(), cfg: realtime.Config{InputAudio: realtime.AudioFormat{Encoding: "g711-ulaw"}}},
		{name: "output rate", model: srv.model(), cfg: realtime.Config{OutputAudio: realtime.AudioFormat{SampleRate: 16000}}},
		{name: "vertex", model: srv.model(google.WithVertex("p", "global"), google.WithTokenSource(func(context.Context) (string, error) { return "t", nil })), cfg: realtime.Config{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.model.Connect(t.Context(), &tt.cfg); err == nil {
				t.Error("Connect() error = nil, want an error")
			}
		})
	}
}

func TestRealtimeConn_sendsInput(t *testing.T) {
	got := make(chan map[string]any, 4)
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = readJSON(ctx, t, c)
		send(ctx, t, c, `{"setupComplete":{}}`, `{"toolCall":{"functionCalls":[{"id":"call_1","name":"lookup","args":{}}]}}`)
		for range 4 {
			got <- readJSON(ctx, t, c)
		}
	})
	conn, err := srv.model().Connect(t.Context(), &realtime.Config{InputAudio: realtime.AudioFormat{SampleRate: 24000}})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	defer conn.Close()
	collect(t, conn, 1)
	ctx := t.Context()
	steps := []error{
		conn.SendAudio(ctx, []byte{1, 2, 3}),
		conn.SendText(ctx, "hello"),
		conn.SendToolResult(ctx, "call_1", ai.ToolOutput{Text: "ok"}, false),
		conn.SendToolResult(ctx, "call_1", ai.ToolOutput{JSON: json.RawMessage(`{"lights":"on"}`)}, false),
	}
	for i, err := range steps {
		if err != nil {
			t.Fatalf("step %d error = %v", i, err)
		}
	}
	wants := []string{
		`{"realtimeInput":{"audio":{"data":"AQID","mimeType":"audio/pcm;rate=24000"}}}`,
		`{"realtimeInput":{"text":"hello"}}`,
		`{"toolResponse":{"functionResponses":[{"id":"call_1","name":"lookup","response":{"output":"ok"}}]}}`,
		`{"toolResponse":{"functionResponses":[{"id":"call_1","name":"lookup","response":{"lights":"on"}}]}}`,
	}
	for i, want := range wants {
		var w map[string]any
		if err := json.Unmarshal([]byte(want), &w); err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(w, <-got); diff != "" {
			t.Errorf("message %d mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestRealtimeConn_interruptIsUnsupported(t *testing.T) {
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = readJSON(ctx, t, c)
		send(ctx, t, c, `{"setupComplete":{}}`)
		_, _, _ = c.Read(ctx)
	})
	conn, err := srv.model().Connect(t.Context(), &realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	defer conn.Close()
	if err := conn.Interrupt(t.Context()); !errors.Is(err, realtime.ErrUnsupported) {
		t.Errorf("Interrupt() error = %v, want ErrUnsupported", err)
	}
}

func TestRealtimeConn_decodesServerMessages(t *testing.T) {
	lines := liveFixture(t)
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = readJSON(ctx, t, c)
		send(ctx, t, c, lines...)
		_, _, _ = c.Read(ctx)
	})
	conn, err := srv.model().Connect(t.Context(), &realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	defer conn.Close()
	got := collect(t, conn, 14)
	usage := ai.Usage{Input: 20, Output: 13, CacheRead: 4, Reasoning: 3, Raw: json.RawMessage(`{"promptTokenCount":20,"cachedContentTokenCount":4,"responseTokenCount":10,"thoughtsTokenCount":3,"totalTokenCount":33}`)}
	want := []realtime.Event{
		realtime.AudioDelta{Data: []byte{1, 2, 3}},
		realtime.TextDelta{Text: "Hello"},
		realtime.Transcript{Role: ai.RoleUser, Text: "turn on "},
		realtime.Transcript{Role: ai.RoleUser, Text: "the lights"},
		realtime.Transcript{Role: ai.RoleUser, Text: "turn on the lights", Final: true},
		realtime.Transcript{Role: ai.RoleAssistant, Text: "Sure, "},
		realtime.Transcript{Role: ai.RoleAssistant, Text: "done."},
		realtime.ToolCall{ID: "call_1", Name: "turn_on_the_lights", Input: json.RawMessage(`{"room":"kitchen"}`)},
		realtime.ToolCanceled{CallID: "call_1"},
		realtime.GoAway{After: 50 * time.Second},
		realtime.Resumable{Handle: "handle-1"},
		realtime.Transcript{Role: ai.RoleAssistant, Text: "Sure, done.", Final: true},
		realtime.Interrupted{},
		realtime.TurnEnd{Usage: usage, FinishReason: ai.FinishStop},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

func TestRealtimeSession_toolCancellationCancelsToolContext(t *testing.T) {
	started := make(chan struct{})
	canceled := make(chan error, 1)
	tool := ai.Tool{Name: "slow", Execute: func(ctx context.Context, _ json.RawMessage) (ai.ToolOutput, error) {
		close(started)
		<-ctx.Done()
		canceled <- context.Cause(ctx)
		return ai.ToolOutput{}, ctx.Err()
	}}
	srv := newLiveServer(t, func(ctx context.Context, c *websocket.Conn) {
		_ = readJSON(ctx, t, c)
		send(ctx, t, c, `{"setupComplete":{}}`, `{"toolCall":{"functionCalls":[{"id":"c1","name":"slow","args":{}}]}}`)
		<-started
		send(ctx, t, c, `{"toolCallCancellation":{"ids":["c1"]}}`)
		_, _, _ = c.Read(ctx)
	})
	sess, err := realtime.Connect(t.Context(), srv.model(), realtime.Config{Tools: []ai.Tool{tool}})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	defer sess.Close()
	go func() {
		for range sess.Events() {
		}
	}()
	select {
	case cause := <-canceled:
		if !errors.Is(cause, realtime.ErrInterrupted) {
			t.Errorf("tool context cause = %v, want ErrInterrupted", cause)
		}
	case <-time.After(5 * time.Second):
		t.Error("tool context was not canceled")
	}
}

func TestRealtimeModel_mintClientSecret(t *testing.T) {
	var body []byte
	var path string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		jsonReply(`{"name":"auth_tokens/abc123"}`)(w, r)
	})
	p := srv.provider(google.WithBaseURL(srv.URL + "/v1beta"))
	cfg := realtime.Config{Instructions: "Greet the visitor.", Voice: "Kore"}
	before := time.Now()
	secret, err := realtime.MintClientSecret(t.Context(), p.RealtimeModel("gemini-3.8-live"), cfg, realtime.MintOptions{TTL: 2 * time.Minute})
	if err != nil {
		t.Fatalf("MintClientSecret() error = %v", err)
	}
	req := srv.last(t)
	body, path = req.Body, req.Path
	if path != "/v1beta/auth_tokens" {
		t.Errorf("path = %q, want /v1beta/auth_tokens", path)
	}
	if got := req.Header.Get("X-Goog-Api-Key"); got != "test-key" {
		t.Errorf("x-goog-api-key = %q, want test-key", got)
	}
	if secret.Value != "auth_tokens/abc123" || !secret.Locked {
		t.Errorf("secret = %+v, want a locked secret with the token name", secret)
	}
	if lo, hi := before.Add(2*time.Minute-time.Second), time.Now().Add(2*time.Minute+time.Second); secret.ExpiresAt.Before(lo) || secret.ExpiresAt.After(hi) {
		t.Errorf("ExpiresAt = %v, want about two minutes ahead", secret.ExpiresAt)
	}
	var got struct {
		Uses                   int    `json:"uses"`
		ExpireTime             string `json:"expireTime"`
		NewSessionExpireTime   string `json:"newSessionExpireTime"`
		LiveConnectConstraints any    `json:"liveConnectConstraints"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	if got.Uses != 1 || got.ExpireTime == "" || got.NewSessionExpireTime == "" {
		t.Errorf("request = %+v, want uses 1 and both expiry times", got)
	}
	wantConstraints := map[string]any{
		"model": "models/gemini-3.8-live",
		"config": map[string]any{
			"responseModalities": []any{"AUDIO"},
			"speechConfig":       map[string]any{"voiceConfig": map[string]any{"prebuiltVoiceConfig": map[string]any{"voiceName": "Kore"}}},
			"systemInstruction":  map[string]any{"parts": []any{map[string]any{"text": "Greet the visitor."}}},
			"sessionResumption":  map[string]any{},
		},
	}
	if diff := cmp.Diff(wantConstraints, got.LiveConnectConstraints); diff != "" {
		t.Errorf("constraints mismatch (-want +got):\n%s", diff)
	}
}

func TestRealtimeModel_mintClientSecretRejects(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{}`))
	model := srv.provider().RealtimeModel("gemini-3.8-live")
	approved := realtime.Config{Tools: []ai.Tool{ai.NewClientTool[struct{}]("t", "").WithApproval(ai.AlwaysAsk)}}
	if _, err := realtime.MintClientSecret(t.Context(), model, approved, realtime.MintOptions{}); !errors.Is(err, realtime.ErrApprovalUnenforceable) {
		t.Errorf("MintClientSecret() error = %v, want ErrApprovalUnenforceable", err)
	}
	if _, err := realtime.MintClientSecret(t.Context(), model, realtime.Config{}, realtime.MintOptions{TTL: 20 * time.Hour}); err == nil {
		t.Error("MintClientSecret() error = nil, want an error for a TTL above the limit")
	}
	if n := srv.count(); n != 0 {
		t.Errorf("server saw %d requests, want 0", n)
	}
}
