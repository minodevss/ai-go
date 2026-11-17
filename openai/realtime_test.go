package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
	"github.com/minodevss/ai-go/realtime"
)

type wsServer struct {
	*httptest.Server
	mu      sync.Mutex
	url     string
	headers http.Header
}

func newWSServer(t *testing.T, script func(ctx context.Context, c *websocket.Conn)) *wsServer {
	t.Helper()
	s := &wsServer{}
	var handlers sync.WaitGroup
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers.Add(1)
		defer handlers.Done()
		s.mu.Lock()
		s.url, s.headers = r.URL.String(), r.Header.Clone()
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

func (s *wsServer) provider(opts ...openai.Option) *openai.Provider {
	return openai.New(append([]openai.Option{openai.WithAPIKey("test-key"), openai.WithBaseURL(s.URL + "/v1")}, opts...)...)
}

func (s *wsServer) seen() (string, http.Header) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.url, s.headers
}

type clientEvents struct {
	t   *testing.T
	ctx context.Context
	c   *websocket.Conn
	ch  chan map[string]any
}

func readClient(ctx context.Context, t *testing.T, c *websocket.Conn) *clientEvents {
	t.Helper()
	ce := &clientEvents{t: t, ctx: ctx, c: c, ch: make(chan map[string]any, 64)}
	go func() {
		defer close(ce.ch)
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var ev map[string]any
			if err := json.Unmarshal(data, &ev); err != nil {
				t.Errorf("client sent invalid JSON: %v", err)
				return
			}
			ce.ch <- ev
		}
	}()
	return ce
}

func (ce *clientEvents) next() map[string]any {
	select {
	case ev, ok := <-ce.ch:
		if ok {
			return ev
		}
		if ce.ctx.Err() == nil {
			ce.t.Error("client connection closed while waiting for an event")
		}
	case <-time.After(5 * time.Second):
		ce.t.Error("timed out waiting for a client event")
	}
	return map[string]any{}
}

func (ce *clientEvents) none(wait time.Duration) {
	select {
	case ev, ok := <-ce.ch:
		if ok {
			ce.t.Errorf("unexpected client event: %v", ev)
		}
	case <-time.After(wait):
	}
}

func field(ev map[string]any, keys ...string) string {
	var cur any = ev
	for _, k := range keys {
		m, ok := cur.(map[string]any)
		if !ok {
			return ""
		}
		cur = m[k]
	}
	s, _ := cur.(string)
	return s
}

func writeServer(ctx context.Context, t *testing.T, c *websocket.Conn, events ...string) {
	t.Helper()
	for _, ev := range events {
		if err := c.Write(ctx, websocket.MessageText, []byte(ev)); err != nil {
			t.Errorf("server write: %v", err)
			return
		}
	}
}

func fixtureLines(t *testing.T, name string) []string {
	t.Helper()
	return strings.Split(strings.TrimSpace(fixture(t, name)), "\n")
}

func nextEvent(t *testing.T, s *realtime.Session, events <-chan realtime.Event) realtime.Event {
	t.Helper()
	select {
	case ev, ok := <-events:
		if !ok {
			t.Fatalf("events ended early; Err() = %v", s.Err())
		}
		return ev
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for a session event")
		return nil
	}
}

func pump(s *realtime.Session) <-chan realtime.Event {
	ch := make(chan realtime.Event, 64)
	go func() {
		defer close(ch)
		for ev := range s.Events() {
			ch <- ev
		}
	}()
	return ch
}

func jsonMap(t *testing.T, ev map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestRealtimeModel_connectSendsTheSessionConfig(t *testing.T) {
	got := make(chan map[string]any, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		got <- readClient(ctx, t, c).next()
		<-ctx.Done()
	})
	tools := []ai.Tool{ai.NewTool("get_weather", "Weather by city", func(context.Context, struct {
		City string `json:"city"`
	},
	) (string, error) {
		return "sunny", nil
	})}
	sess, err := realtime.Connect(t.Context(), srv.provider(openai.WithHeaders(http.Header{"X-Trace": {"abc"}})).RealtimeModel("gpt-realtime-2.1"), realtime.Config{
		Instructions:  "Be brief.",
		Voice:         "marin",
		Tools:         tools,
		TurnDetection: realtime.TurnDetection{Mode: realtime.TurnServerVAD, Threshold: new(0.6), SilenceDuration: 500 * time.Millisecond},
		Provider:      ai.Providers(openai.RealtimeOptions{OutputModalities: []string{"audio"}, InputTranscription: "gpt-4o-mini-transcribe", NoiseReduction: "near_field", MaxOutputTokens: 400, SafetyIdentifier: "user-hash"}),
	})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	url, headers := srv.seen()
	if url != "/v1/realtime?model=gpt-realtime-2.1" {
		t.Errorf("url = %q, want /v1/realtime?model=gpt-realtime-2.1", url)
	}
	if headers.Get("Authorization") != "Bearer test-key" || headers.Get("X-Trace") != "abc" || headers.Get("OpenAI-Safety-Identifier") != "user-hash" {
		t.Errorf("headers = %v, want auth, trace and safety identifier", headers)
	}
	want := `{"type":"session.update","session":{"type":"realtime","instructions":"Be brief.","output_modalities":["audio"],"max_output_tokens":400,
	"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"gpt-4o-mini-transcribe"},"noise_reduction":{"type":"near_field"},"turn_detection":{"type":"server_vad","threshold":0.6,"silence_duration_ms":500}},
	"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"marin"}},
	"tools":[{"type":"function","name":"get_weather","description":"Weather by city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}],"tool_choice":"auto"}}`
	equalJSON(t, want, []byte(jsonMap(t, <-got)))
}

func TestRealtimeModel_audioAndTurnSettingsMapToTheWire(t *testing.T) {
	tests := []struct {
		name string
		cfg  realtime.Config
		want string
	}{
		{name: "defaults", cfg: realtime.Config{}, want: `{"type":"session.update","session":{"type":"realtime","audio":{"input":{"format":{"type":"audio/pcm","rate":24000}},"output":{"format":{"type":"audio/pcm","rate":24000}}}}}`},
		{name: "g711 and manual turns", cfg: realtime.Config{InputAudio: realtime.AudioFormat{Encoding: "g711-ulaw"}, OutputAudio: realtime.AudioFormat{Encoding: "g711-alaw"}, TurnDetection: realtime.TurnDetection{Mode: realtime.TurnManual}}, want: `{"type":"session.update","session":{"type":"realtime","audio":{"input":{"format":{"type":"audio/pcmu"},"turn_detection":null},"output":{"format":{"type":"audio/pcma"}}}}}`},
		{name: "semantic vad and custom voice", cfg: realtime.Config{Voice: "voice_9", TurnDetection: realtime.TurnDetection{Mode: realtime.TurnSemanticVAD, Eagerness: "low"}}, want: `{"type":"session.update","session":{"type":"realtime","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":{"type":"semantic_vad","eagerness":"low"}},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":{"id":"voice_9"}}}}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := make(chan map[string]any, 1)
			srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
				got <- readClient(ctx, t, c).next()
				<-ctx.Done()
			})
			sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), tt.cfg)
			if err != nil {
				t.Fatalf("Connect() error = %v", err)
			}
			t.Cleanup(func() { _ = sess.Close() })
			equalJSON(t, tt.want, []byte(jsonMap(t, <-got)))
		})
	}
}

func TestRealtimeModel_rejectsUnsupportedConfig(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, _ *websocket.Conn) { <-ctx.Done() })
	m := srv.provider().RealtimeModel("m")
	tests := []struct {
		name string
		cfg  realtime.Config
		want error
		text string
	}{
		{name: "resume", cfg: realtime.Config{Resume: "h"}, want: realtime.ErrUnsupported, text: "cannot be resumed"},
		{name: "audio encoding", cfg: realtime.Config{InputAudio: realtime.AudioFormat{Encoding: "mp3"}}, text: "unsupported realtime audio encoding"},
		{name: "turn mode", cfg: realtime.Config{TurnDetection: realtime.TurnDetection{Mode: "loud"}}, text: "unsupported turn detection"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := realtime.Connect(t.Context(), m, tt.cfg)
			if err == nil || !strings.Contains(err.Error(), tt.text) || (tt.want != nil && !errors.Is(err, tt.want)) {
				t.Errorf("Connect() error = %v, want %q", err, tt.text)
			}
		})
	}
}

func TestRealtimeModel_handshakeFailuresAreAPIErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"Incorrect API key","code":"invalid_api_key"}}`))
	}))
	t.Cleanup(srv.Close)
	p := openai.New(openai.WithAPIKey("bad"), openai.WithBaseURL(srv.URL+"/v1"))
	_, err := realtime.Connect(t.Context(), p.RealtimeModel("m"), realtime.Config{})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("Connect() error = %v, want an authentication APIError", err)
	}
	if _, err := realtime.Connect(t.Context(), openai.New(openai.WithBaseURL("http://127.0.0.1:1/v1")).RealtimeModel("m"), realtime.Config{}); err == nil || !strings.HasPrefix(err.Error(), "openai:") {
		t.Errorf("Connect() to a closed port error = %v, want an openai-prefixed error", err)
	}
}

func TestRealtimeModel_decodesServerEvents(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		client.next()
		writeServer(ctx, t, c, fixtureLines(t, "realtime_server_events.jsonl")...)
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{Tools: []ai.Tool{ai.NewClientTool[struct{}]("get_weather", ""), ai.NewClientTool[struct{}]("get_time", "")}})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	events := pump(sess)
	want := []realtime.Event{
		realtime.AudioDelta{Data: []byte{1, 2, 3}},
		realtime.Transcript{Role: ai.RoleAssistant, Text: "Hel"},
		realtime.Transcript{Role: ai.RoleAssistant, Text: "Hello", Final: true},
		realtime.TextDelta{Text: "Hi"},
		realtime.Transcript{Role: ai.RoleUser, Text: "Wh"},
		realtime.Transcript{Role: ai.RoleUser, Text: "What?", Final: true},
		realtime.ToolCall{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"Seoul"}`)},
		realtime.ToolCall{ID: "call_2", Name: "get_time", Input: json.RawMessage(`{}`)},
		realtime.TurnEnd{Usage: ai.Usage{Input: 20, Output: 10, Raw: json.RawMessage(`{"total_tokens":30,"input_tokens":20,"output_tokens":10}`)}, FinishReason: ai.FinishStop},
	}
	var got []realtime.Event
	for range want {
		got = append(got, nextEvent(t, sess, events))
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

func TestRealtimeModel_sendsInputs(t *testing.T) {
	received := make(chan []map[string]any, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		client.next()
		var evs []map[string]any
		for range 4 {
			evs = append(evs, client.next())
		}
		received <- evs
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	ctx := t.Context()
	if err := errors.Join(sess.SendAudio(ctx, []byte{1, 2, 3, 4}), sess.SendText(ctx, "hello"), sess.Interrupt(ctx)); err != nil {
		t.Fatalf("send error = %v", err)
	}
	evs := <-received
	want := []string{
		`{"type":"input_audio_buffer.append","audio":"` + base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4}) + `"}`,
		`{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}`,
		`{"type":"response.create"}`,
		`{"type":"response.cancel"}`,
	}
	for i, w := range want {
		equalJSON(t, w, []byte(jsonMap(t, evs[i])))
	}
}

func weatherTool(runs *int, mu *sync.Mutex) ai.Tool {
	return ai.NewTool("get_weather", "Weather", func(_ context.Context, in struct {
		City string `json:"city"`
	},
	) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		*runs++
		return "sunny in " + in.City, nil
	})
}

const parallelCalls = `{"type":"response.done","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Seoul\"}"},{"type":"function_call","call_id":"call_2","name":"get_weather","arguments":"{\"city\":\"Busan\"}"}]}}`

func TestRealtimeModel_serverSideToolsAnswerOnceAfterAllCalls(t *testing.T) {
	done := make(chan []map[string]any, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		client.next()
		writeServer(ctx, t, c, `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`, parallelCalls)
		var evs []map[string]any
		for range 3 {
			evs = append(evs, client.next())
		}
		client.none(100 * time.Millisecond)
		done <- evs
		<-ctx.Done()
	})
	var runs int
	var mu sync.Mutex
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{Tools: []ai.Tool{weatherTool(&runs, &mu)}})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	go func() {
		for range sess.Events() {
		}
	}()
	evs := <-done
	outputs := map[string]string{}
	for _, ev := range evs[:2] {
		if field(ev, "type") != "conversation.item.create" || field(ev, "item", "type") != "function_call_output" {
			t.Fatalf("event %v, want a function_call_output item", ev)
		}
		outputs[field(ev, "item", "call_id")] = field(ev, "item", "output")
	}
	if diff := cmp.Diff(map[string]string{"call_1": `"sunny in Seoul"`, "call_2": `"sunny in Busan"`}, outputs); diff != "" {
		t.Errorf("outputs mismatch (-want +got):\n%s", diff)
	}
	if evs[2]["type"] != "response.create" {
		t.Errorf("third event = %v, want exactly one response.create after both results", evs[2])
	}
}

func TestRealtimeModel_approvalGatesTheToolBeforeAnythingIsSent(t *testing.T) {
	tests := []struct {
		name     string
		approve  bool
		wantRuns int
		wantText string
	}{
		{name: "approved", approve: true, wantRuns: 1, wantText: `"sunny in Seoul"`},
		{name: "denied", approve: false, wantRuns: 0, wantText: "denied"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputs := make(chan string, 1)
			srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
				client := readClient(ctx, t, c)
				client.next()
				writeServer(ctx, t, c, `{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Seoul\"}"}]}}`)
				client.none(150 * time.Millisecond)
				ev := client.next()
				outputs <- field(ev, "item", "output")
				<-ctx.Done()
			})
			var runs int
			var mu sync.Mutex
			sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{Tools: []ai.Tool{weatherTool(&runs, &mu).WithApproval(ai.AlwaysAsk)}})
			if err != nil {
				t.Fatalf("Connect() error = %v", err)
			}
			t.Cleanup(func() { _ = sess.Close() })
			events := pump(sess)
			var req realtime.ApprovalRequest
			for req.ApprovalID == "" {
				if r, ok := nextEvent(t, sess, events).(realtime.ApprovalRequest); ok {
					req = r
				}
			}
			time.Sleep(200 * time.Millisecond)
			if err := sess.Approve(t.Context(), req.ApprovalID, tt.approve); err != nil {
				t.Fatalf("Approve() error = %v", err)
			}
			if out := <-outputs; !strings.Contains(out, tt.wantText) {
				t.Errorf("tool output sent to the model = %q, want %q", out, tt.wantText)
			}
			mu.Lock()
			defer mu.Unlock()
			if runs != tt.wantRuns {
				t.Errorf("tool ran %d times, want %d", runs, tt.wantRuns)
			}
		})
	}
}

func TestRealtimeModel_interruptionIsReportedOnce(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readClient(ctx, t, c).next()
		writeServer(ctx, t, c,
			`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`,
			`{"type":"response.output_audio.delta","delta":"AQ=="}`,
			`{"type":"input_audio_buffer.speech_started","audio_start_ms":120,"item_id":"i"}`,
			`{"type":"response.done","response":{"id":"r1","status":"canceled","status_details":{"type":"canceled","reason":"turn_detected"},"output":[{"type":"function_call","call_id":"stale","name":"get_weather","arguments":"{}"}]}}`,
			`{"type":"input_audio_buffer.speech_started","audio_start_ms":900,"item_id":"j"}`,
			`{"type":"response.created","response":{"id":"r2","status":"in_progress"}}`,
			`{"type":"response.done","response":{"id":"r2","status":"completed","output":[]}}`,
		)
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	events := pump(sess)
	want := []realtime.Event{
		realtime.AudioDelta{Data: []byte{1}},
		realtime.Interrupted{},
		realtime.TurnEnd{FinishReason: ai.FinishOther},
		realtime.TurnEnd{FinishReason: ai.FinishStop},
	}
	var got []realtime.Event
	for range want {
		got = append(got, nextEvent(t, sess, events))
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

func TestRealtimeModel_canceledResponseWithoutSpeechStillInterrupts(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readClient(ctx, t, c).next()
		writeServer(ctx, t, c,
			`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`,
			`{"type":"response.done","response":{"id":"r1","status":"canceled","status_details":{"type":"canceled","reason":"client_canceled"},"output":[]}}`)
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	events := pump(sess)
	if _, ok := nextEvent(t, sess, events).(realtime.Interrupted); !ok {
		t.Error("first event is not Interrupted")
	}
	if end, ok := nextEvent(t, sess, events).(realtime.TurnEnd); !ok || end.FinishReason != ai.FinishOther {
		t.Errorf("second event = %+v, want a TurnEnd with FinishOther", end)
	}
}

func TestRealtimeModel_incompleteResponsesMapFinishReasons(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readClient(ctx, t, c).next()
		writeServer(ctx, t, c,
			`{"type":"response.done","response":{"id":"a","status":"incomplete","status_details":{"type":"incomplete","reason":"max_output_tokens"},"output":[]}}`,
			`{"type":"response.done","response":{"id":"b","status":"incomplete","status_details":{"type":"incomplete","reason":"content_filter"},"output":[]}}`,
			`{"type":"response.done","response":{"id":"c","status":"failed","status_details":{"type":"failed","error":{"type":"server_error","code":"server_error","message":"x"}},"output":[]}}`)
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	events := pump(sess)
	for _, want := range []ai.FinishReason{ai.FinishLength, ai.FinishContentFilter, ai.FinishError} {
		if end, ok := nextEvent(t, sess, events).(realtime.TurnEnd); !ok || end.FinishReason != want {
			t.Errorf("TurnEnd = %+v, want %s", end, want)
		}
	}
}

func TestRealtimeModel_interruptCancelsToolsOfTheCanceledResponse(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		client.next()
		writeServer(ctx, t, c,
			`{"type":"response.created","response":{"id":"r1","status":"in_progress"}}`,
			`{"type":"response.done","response":{"id":"r1","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"slow","arguments":"{}"}]}}`,
			`{"type":"response.created","response":{"id":"r2","status":"in_progress"}}`,
			`{"type":"input_audio_buffer.speech_started","audio_start_ms":1,"item_id":"i"}`)
		client.none(200 * time.Millisecond)
	})
	canceled := make(chan error, 1)
	slow := ai.NewTool("slow", "", func(ctx context.Context, _ struct{}) (string, error) {
		<-ctx.Done()
		canceled <- context.Cause(ctx)
		return "", ctx.Err()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{Tools: []ai.Tool{slow}})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	go func() {
		for range sess.Events() {
		}
	}()
	select {
	case cause := <-canceled:
		if !errors.Is(cause, realtime.ErrInterrupted) {
			t.Errorf("tool ctx cause = %v, want ErrInterrupted", cause)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the tool was not canceled when the user started speaking")
	}
}

func TestRealtimeModel_errorEvents(t *testing.T) {
	tests := []struct {
		name    string
		event   string
		wantErr bool
	}{
		{name: "fatal error ends the session", event: `{"type":"error","error":{"type":"server_error","code":"server_error","message":"The server had an error"}}`, wantErr: true},
		{name: "a cancel race is not an error", event: `{"type":"error","error":{"type":"invalid_request_error","code":"response_cancel_not_active","message":"no active response"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
				readClient(ctx, t, c).next()
				writeServer(ctx, t, c, tt.event, `{"type":"response.output_text.delta","delta":"after"}`)
				<-ctx.Done()
			})
			sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
			if err != nil {
				t.Fatalf("Connect() error = %v", err)
			}
			t.Cleanup(func() { _ = sess.Close() })
			events := pump(sess)
			if tt.wantErr {
				for range events {
				}
				apiErr, ok := errors.AsType[*ai.APIError](sess.Err())
				if !ok || apiErr.Code != "server_error" || apiErr.StatusCode != http.StatusInternalServerError {
					t.Errorf("Err() = %v, want a server_error APIError", sess.Err())
				}
				return
			}
			if ev, ok := nextEvent(t, sess, events).(realtime.TextDelta); !ok || ev.Text != "after" {
				t.Errorf("first event = %+v, want the text after the ignored error", ev)
			}
		})
	}
}

func TestRealtimeModel_connectionLoss(t *testing.T) {
	tests := []struct {
		name    string
		close   func(c *websocket.Conn)
		wantErr bool
	}{
		{name: "normal close", close: func(c *websocket.Conn) { _ = c.Close(websocket.StatusNormalClosure, "bye") }},
		{name: "abrupt close", close: func(c *websocket.Conn) { _ = c.CloseNow() }, wantErr: true},
		{name: "protocol error", close: func(c *websocket.Conn) { _ = c.Close(websocket.StatusInternalError, "boom") }, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
				readClient(ctx, t, c).next()
				tt.close(c)
			})
			sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
			if err != nil {
				t.Fatalf("Connect() error = %v", err)
			}
			for range sess.Events() {
			}
			if (sess.Err() != nil) != tt.wantErr {
				t.Errorf("Err() = %v, want error=%v", sess.Err(), tt.wantErr)
			}
			if err := sess.Close(); err != nil && !tt.wantErr {
				t.Errorf("Close() error = %v", err)
			}
		})
	}
}

func TestRealtimeModel_garbageFromTheServerEndsTheSession(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readClient(ctx, t, c).next()
		writeServer(ctx, t, c, `not json`)
		<-ctx.Done()
	})
	sess, err := realtime.Connect(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	for range sess.Events() {
	}
	if err := sess.Err(); err == nil || !strings.Contains(err.Error(), "decode realtime event") {
		t.Errorf("Err() = %v, want a decode error", err)
	}
}

func TestRealtimeModel_limiterAppliesToConnect(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		readClient(ctx, t, c).next()
		<-ctx.Done()
	})
	lim := &countingLimiter{}
	sess, err := realtime.Connect(t.Context(), srv.provider(openai.WithLimiter(lim)).RealtimeModel("m"), realtime.Config{})
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	if lim.n != 1 {
		t.Errorf("limiter waited %d times, want 1", lim.n)
	}
}

func TestRealtimeModel_mintClientSecret(t *testing.T) {
	tests := []struct {
		name     string
		ttl      time.Duration
		wantSecs int
	}{
		{name: "default one minute", wantSecs: 60},
		{name: "rounds up to whole seconds", ttl: 10500 * time.Millisecond, wantSecs: 11},
		{name: "lower bound", ttl: 10 * time.Second, wantSecs: 10},
		{name: "one hour", ttl: time.Hour, wantSecs: 3600},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(fixture(t, "realtime_client_secret.json")))
			secret, err := realtime.MintClientSecret(t.Context(), srv.provider().RealtimeModel("gpt-realtime-2.1"), realtime.Config{
				Instructions: "You are a friendly assistant.",
				Voice:        "marin",
				Tools:        []ai.Tool{ai.NewClientTool[struct{ City string }]("lookup", "Look up a city")},
			}, realtime.MintOptions{TTL: tt.ttl})
			if err != nil {
				t.Fatalf("MintClientSecret() error = %v", err)
			}
			got := srv.last(t)
			if got.Path != "/v1/realtime/client_secrets" || got.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("request = %s auth %q, want the client_secrets endpoint with the server key", got.Path, got.Header.Get("Authorization"))
			}
			want := `{"expires_after":{"anchor":"created_at","seconds":` + itoa(tt.wantSecs) + `},"session":{"type":"realtime","model":"gpt-realtime-2.1","instructions":"You are a friendly assistant.","audio":{"input":{"format":{"type":"audio/pcm","rate":24000}},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"marin"}},"tools":[{"type":"function","name":"lookup","description":"Look up a city","parameters":{"type":"object","properties":{"City":{"type":"string"}},"required":["City"],"additionalProperties":false}}],"tool_choice":"auto"}}`
			equalJSON(t, want, got.Body)
			if secret.Value != "ek_68af296e8e408191a1120ab6383263c2" || !secret.ExpiresAt.Equal(time.Unix(1756310470, 0)) {
				t.Errorf("secret = %+v, want the fixture value and expiry", secret)
			}
			if secret.Locked || len(secret.Warnings) != 1 || !strings.Contains(secret.Warnings[0].Message, "override") {
				t.Errorf("secret = %+v, want it reported as not locked with a warning", secret)
			}
		})
	}
}

func TestRealtimeModel_mintRejectsTTLOutsideTheAPIRange(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "realtime_client_secret.json")))
	m := srv.provider().RealtimeModel("m")
	for _, ttl := range []time.Duration{time.Second, 9999 * time.Millisecond, 2*time.Hour + time.Second} {
		model := openaiMinter{m}
		if _, err := model.MintClientSecret(t.Context(), &realtime.Config{}, realtime.MintOptions{TTL: ttl}); err == nil {
			t.Errorf("MintClientSecret(TTL %v) succeeded", ttl)
		}
	}
	if srv.count() != 0 {
		t.Errorf("%d requests were sent for an invalid TTL", srv.count())
	}
}

type openaiMinter struct{ *openai.RealtimeModel }

func TestRealtimeModel_mintFailuresAreAPIErrors(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"message":"no access to realtime","code":"model_not_found"}}`))
	})
	_, err := realtime.MintClientSecret(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{}, realtime.MintOptions{})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("MintClientSecret() error = %v, want a 403 APIError", err)
	}
}

func TestRealtimeModel_mintRefusesApprovalTools(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "realtime_client_secret.json")))
	tool := ai.NewClientTool[struct{}]("pay", "").WithApproval(ai.AlwaysAsk)
	_, err := realtime.MintClientSecret(t.Context(), srv.provider().RealtimeModel("m"), realtime.Config{Tools: []ai.Tool{tool}}, realtime.MintOptions{})
	if !errors.Is(err, realtime.ErrApprovalUnenforceable) || srv.count() != 0 {
		t.Errorf("MintClientSecret() error = %v with %d requests, want a refusal before any request", err, srv.count())
	}
}

func TestTranscriptionModel_liveStreamOverRealtime(t *testing.T) {
	type result struct {
		session map[string]any
		appends int
		bytes   int
		commit  bool
	}
	got := make(chan result, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		var res result
		res.session = client.next()
		for {
			ev := client.next()
			switch ev["type"] {
			case "input_audio_buffer.append":
				raw, err := base64.StdEncoding.DecodeString(field(ev, "audio"))
				if err != nil {
					t.Errorf("bad audio: %v", err)
				}
				res.appends++
				res.bytes += len(raw)
				continue
			case "input_audio_buffer.commit":
				res.commit = true
			}
			break
		}
		got <- res
		writeServer(ctx, t, c,
			`{"type":"input_audio_buffer.committed","item_id":"i"}`,
			`{"type":"conversation.item.input_audio_transcription.delta","item_id":"i","delta":"Hel"}`,
			`{"type":"conversation.item.input_audio_transcription.delta","item_id":"i","delta":"lo"}`,
			`{"type":"conversation.item.input_audio_transcription.completed","item_id":"i","transcript":"Hello"}`,
			`{"type":"error","error":{"type":"invalid_request_error","code":"input_audio_buffer_commit_empty","message":"empty"}}`)
		<-ctx.Done()
	})
	audio := strings.NewReader(strings.Repeat("\x01\x02", 12000))
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("gpt-live-transcribe"), ai.TranscribeStreamRequest{Audio: audio, MediaType: "audio/pcm;rate=24000", Language: "en"})
	var segs []ai.Segment
	for seg := range s.Segments() {
		segs = append(segs, seg)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	want := []ai.Segment{{Text: "Hel", Partial: true}, {Text: "lo", Partial: true}, {Text: "Hello"}}
	if diff := cmp.Diff(want, segs); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	res := <-got
	equalJSON(t, `{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"gpt-live-transcribe","language":"en"},"turn_detection":null}}}}`, []byte(jsonMap(t, res.session)))
	if res.appends != 3 || res.bytes != 24000 || !res.commit {
		t.Errorf("appends = %d (%d bytes), commit = %v, want 3 chunks of at most 9600 bytes then a commit", res.appends, res.bytes, res.commit)
	}
	if url, _ := srv.seen(); url != "/v1/realtime?intent=transcription" {
		t.Errorf("url = %q, want the transcription intent without a model", url)
	}
}

func TestTranscriptionModel_liveStreamServerVAD(t *testing.T) {
	sessions := make(chan map[string]any, 1)
	srv := newWSServer(t, func(ctx context.Context, c *websocket.Conn) {
		client := readClient(ctx, t, c)
		sessions <- client.next()
		for client.next()["type"] != "input_audio_buffer.commit" {
			continue
		}
		writeServer(ctx, t, c,
			`{"type":"input_audio_buffer.committed","item_id":"a"}`,
			`{"type":"conversation.item.input_audio_transcription.completed","item_id":"a","transcript":"first turn"}`,
			`{"type":"input_audio_buffer.committed","item_id":"b"}`,
			`{"type":"conversation.item.input_audio_transcription.completed","item_id":"b","transcript":"second turn"}`,
			`{"type":"error","error":{"type":"invalid_request_error","code":"input_audio_buffer_commit_empty","message":"empty"}}`)
		<-ctx.Done()
	})
	s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("gpt-4o-transcribe"), ai.TranscribeStreamRequest{Audio: strings.NewReader("ab"), MediaType: "audio/pcm"})
	var texts []string
	for seg := range s.Segments() {
		texts = append(texts, seg.Text)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if diff := cmp.Diff([]string{"first turn", "second turn"}, texts); diff != "" {
		t.Errorf("segments mismatch (-want +got):\n%s", diff)
	}
	session := <-sessions
	equalJSON(t, `{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"gpt-4o-transcribe"},"turn_detection":{"type":"server_vad"}}}}}`, []byte(jsonMap(t, session)))
}

func TestTranscriptionModel_liveStreamRejectsOtherRates(t *testing.T) {
	srv := newWSServer(t, func(ctx context.Context, _ *websocket.Conn) { <-ctx.Done() })
	tests := []struct {
		name      string
		mediaType string
	}{
		{name: "16 kHz", mediaType: "audio/pcm;rate=16000"},
		{name: "invalid rate", mediaType: "audio/pcm;rate=fast"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("m"), ai.TranscribeStreamRequest{Audio: strings.NewReader("ab"), MediaType: tt.mediaType})
			for range s.Segments() {
			}
			if s.Err() == nil {
				t.Error("Err() = nil, want a rate error")
			}
		})
	}
	cfg := realtime.Config{InputAudio: realtime.AudioFormat{SampleRate: 16000}}
	if _, err := srv.provider().RealtimeModel("m").Connect(t.Context(), &cfg); err == nil {
		t.Error("Connect() with a 16 kHz input succeeded, want an error")
	}
}

func TestTranscriptionModel_liveStreamFailures(t *testing.T) {
	tests := []struct {
		name   string
		script func(ctx context.Context, c *websocket.Conn)
		audio  func() *failingReader
		want   string
	}{
		{name: "transcription failed", script: func(ctx context.Context, c *websocket.Conn) {
			readClient(ctx, t, c).next()
			writeServer(ctx, t, c, `{"type":"conversation.item.input_audio_transcription.failed","item_id":"i","error":{"message":"bad"}}`)
			<-ctx.Done()
		}, audio: func() *failingReader { return newFailingReader("ab", nil) }, want: "transcription failed"},
		{name: "audio read error", script: func(ctx context.Context, c *websocket.Conn) {
			readClient(ctx, t, c).next()
			<-ctx.Done()
		}, audio: func() *failingReader { return newFailingReader("ab", errors.New("mic unplugged")) }, want: "mic unplugged"},
		{name: "server error event", script: func(ctx context.Context, c *websocket.Conn) {
			readClient(ctx, t, c).next()
			writeServer(ctx, t, c, `{"type":"error","error":{"code":"invalid_request_error","message":"no"}}`)
			<-ctx.Done()
		}, audio: func() *failingReader { return newFailingReader("ab", nil) }, want: "invalid_request_error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newWSServer(t, tt.script)
			s := ai.StreamTranscribe(t.Context(), srv.provider().TranscriptionModel("m"), ai.TranscribeStreamRequest{Audio: tt.audio(), MediaType: "audio/pcm"})
			for range s.Segments() {
			}
			if err := s.Err(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Err() = %v, want %q", err, tt.want)
			}
		})
	}
}

type failingReader struct {
	data    string
	then    error
	done    bool
	release chan struct{}
	once    sync.Once
}

func newFailingReader(data string, then error) *failingReader {
	return &failingReader{data: data, then: then, release: make(chan struct{})}
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.done {
		if r.then == nil {
			<-r.release
			return 0, io.ErrClosedPipe
		}
		return 0, r.then
	}
	r.done = true
	return copy(p, r.data), nil
}

func (r *failingReader) Close() error {
	r.once.Do(func() { close(r.release) })
	return nil
}
