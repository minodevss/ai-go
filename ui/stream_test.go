package ui_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/ui"
)

func TestRespond_textStream(t *testing.T) {
	m := scripted(aitest.Turn(aitest.TextChunks("Hel", "lo")))
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}), ui.Options{NewID: sequentialIDs()})

	wantHeaders := map[string]string{
		"Content-Type":                  "text/event-stream",
		"Cache-Control":                 "no-cache",
		"Connection":                    "keep-alive",
		"X-Vercel-Ai-Ui-Message-Stream": "v1",
		"X-Accel-Buffering":             "no",
	}
	for k, want := range wantHeaders {
		if got := rec.Header().Get(k); got != want {
			t.Errorf("header %s = %q, want %q", k, got, want)
		}
	}
	want := `data: {"type":"start","messageId":"msg_1"}

data: {"type":"start-step"}

data: {"type":"text-start","id":"text_1"}

data: {"type":"text-delta","id":"text_1","delta":"Hel"}

data: {"type":"text-delta","id":"text_1","delta":"lo"}

data: {"type":"text-end","id":"text_1"}

data: {"type":"finish-step"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]

`
	if diff := cmp.Diff(want, rec.Body.String()); diff != "" {
		t.Errorf("body mismatch (-want +got):\n%s", diff)
	}
}

func echoTools() []ai.Tool {
	return []ai.Tool{
		ai.NewTool("echo", "echo", func(_ context.Context, in pathInput) (string, error) { return in.Path, nil }),
		ai.NewTool("fail", "visible to the model only", func(_ context.Context, in pathInput) (string, error) {
			return "", ai.ToolErrorf("secret detail %s", in.Path)
		}),
		ai.NewTool("boom", "panics", func(context.Context, pathInput) (string, error) { panic("kaboom with secrets") }),
		ai.NewTool("plain", "plain error", func(context.Context, pathInput) (string, error) { return "", errors.New("pg password") }),
	}
}

func TestRespond_mapsEveryPartKind(t *testing.T) {
	m := scripted(
		aitest.Turn(
			aitest.Reasoning("think"),
			aitest.TextChunks("a", "b"),
			aitest.Part(ai.File{MediaType: "image/png", Data: []byte{1}}),
			aitest.StreamPart(ai.Citation{URL: "https://e.com", Title: "E"}),
			aitest.StreamPart(ai.Citation{Title: "Doc", DocIndex: new(0)}),
			aitest.ToolCall("echo", `{"path":"p"}`),
		),
		aitest.Turn(aitest.Text("final")),
	)
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, Tools: echoTools()}), ui.Options{NewID: sequentialIDs()})
	got := chunksOf(t, rec.Body.String())
	wantTypes := []string{
		"start", "start-step",
		"reasoning-start", "reasoning-delta", "reasoning-end",
		"text-start", "text-delta", "text-delta", "text-end",
		"file", "source-url", "source-document",
		"tool-input-start", "tool-input-delta", "tool-input-available", "tool-output-available",
		"finish-step", "start-step", "text-start", "text-delta", "text-end", "finish-step", "finish",
	}
	if diff := cmp.Diff(wantTypes, typesOf(got)); diff != "" {
		t.Fatalf("chunk types mismatch (-want +got):\n%s", diff)
	}
	byType := map[string]map[string]any{}
	for _, c := range got {
		kind, _ := c["type"].(string)
		byType[kind] = c
	}
	checks := []struct {
		typ  string
		key  string
		want any
	}{
		{"file", "url", "data:image/png;base64,AQ=="},
		{"file", "mediaType", "image/png"},
		{"source-url", "url", "https://e.com"},
		{"source-url", "title", "E"},
		{"source-document", "title", "Doc"},
		{"tool-input-available", "toolName", "echo"},
		{"tool-output-available", "output", "p"},
		{"tool-output-available", "toolCallId", "call_1"},
		{"finish", "finishReason", "stop"},
	}
	for _, c := range checks {
		if got := byType[c.typ][c.key]; got != c.want {
			t.Errorf("%s.%s = %v, want %v", c.typ, c.key, got, c.want)
		}
	}
}

func TestRespond_toolErrorsAreMaskedByDefault(t *testing.T) {
	m := scripted(
		aitest.Turn(
			aitest.ToolCall("fail", `{"path":"p"}`),
			aitest.ToolCall("boom", `{"path":"p"}`),
			aitest.ToolCall("plain", `{"path":"p"}`),
			aitest.ToolCall("missing", `{}`),
		),
		aitest.Turn(aitest.Text("done")),
	)
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, Tools: echoTools()}), ui.Options{})
	body := rec.Body.String()
	for _, secret := range []string{"secret detail", "kaboom", "pg password", "available tools"} {
		if strings.Contains(body, secret) {
			t.Errorf("stream leaks %q:\n%s", secret, body)
		}
	}
	var errorTexts []string
	for _, c := range chunksOf(t, body) {
		if c["type"] == "tool-output-error" {
			text, _ := c["errorText"].(string)
			errorTexts = append(errorTexts, text)
		}
	}
	if diff := cmp.Diff([]string{"An error occurred.", "An error occurred.", "An error occurred.", "An error occurred."}, errorTexts); diff != "" {
		t.Errorf("tool error texts mismatch (-want +got):\n%s", diff)
	}
}

func TestRespond_onErrorCustomizesToolStreamAndPanicErrors(t *testing.T) {
	var seen []string
	onError := func(err error) string {
		seen = append(seen, err.Error())
		if tool, ok := errors.AsType[*ai.ToolError](err); ok {
			return "tool said: " + tool.Message
		}
		return "custom"
	}
	m := scripted(aitest.Turn(aitest.ToolCall("fail", `{"path":"p"}`), aitest.ToolCall("boom", `{"path":"p"}`)), aitest.Turn(aitest.Text("x"), aitest.Err(errors.New("stream broke"))))
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, Tools: echoTools(), MaxRetries: new(0)}), ui.Options{OnError: onError})
	var texts []string
	for _, c := range chunksOf(t, rec.Body.String()) {
		if s, ok := c["errorText"].(string); ok {
			texts = append(texts, s)
		}
	}
	if diff := cmp.Diff([]string{"tool said: secret detail p", "custom", "custom"}, texts); diff != "" {
		t.Errorf("error texts mismatch (-want +got):\n%s", diff)
	}
	if len(seen) != 3 {
		t.Errorf("OnError saw %v, want the tool error, the panic and the stream error", seen)
	}
}

func TestStream_panicsAreRecoveredAndMasked(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	var finish ui.Finish
	ui.Stream(rec, req, func(context.Context, *ui.Writer) error { panic("secret panic value") }, ui.Options{
		NewID:    sequentialIDs(),
		OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil },
	})
	body := rec.Body.String()
	if strings.Contains(body, "secret panic value") {
		t.Errorf("panic value leaked:\n%s", body)
	}
	want := []string{"start", "error", "finish"}
	if diff := cmp.Diff(want, typesOf(chunksOf(t, body))); diff != "" {
		t.Errorf("chunk types mismatch (-want +got):\n%s", diff)
	}
	if finish.Err == nil || finish.Aborted {
		t.Errorf("Finish = %+v, want the panic as an error and no abort", finish)
	}
}

func TestRespond_streamErrorEndsWithErrorAndFinish(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("partial"), aitest.Err(errors.New("db password hunter2"))))
	var finish ui.Finish
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, MaxRetries: new(0)}), ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	body := rec.Body.String()
	if strings.Contains(body, "hunter2") {
		t.Errorf("error leaked:\n%s", body)
	}
	types := typesOf(chunksOf(t, body))
	if types[len(types)-2] != "error" || types[len(types)-1] != "finish" || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Errorf("chunk types = %v, want error, finish, [DONE] at the end", types)
	}
	if len(finish.New) != 0 || finish.Err == nil || finish.FinishReason != ai.FinishError {
		t.Errorf("Finish = %+v, want no completed steps and the error", finish)
	}
}

func TestRespond_onFinishReceivesCompletedStepsOnly(t *testing.T) {
	m := scripted(aitest.Turn(aitest.ToolCall("echo", `{"path":"p"}`)), aitest.Turn(aitest.Text("partial"), aitest.Err(errors.New("boom"))))
	var finish ui.Finish
	respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, Tools: echoTools(), MaxRetries: new(0)}), ui.Options{
		NewID:    sequentialIDs(),
		OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil },
	})
	if len(finish.New) != 2 || finish.New[0].Role != ai.RoleAssistant || finish.New[1].Role != ai.RoleTool {
		t.Fatalf("Finish.New = %+v, want the first step's assistant and tool messages", finish.New)
	}
	if finish.UI.ID != "msg_1" || finish.UI.Role != ui.RoleAssistant || len(toolParts(finish.UI)) != 1 || textOf(finish.UI) != "" {
		t.Errorf("Finish.UI = %s, want the completed step only", mustJSON(t, finish.UI))
	}
}

func TestRespond_onFinishErrorIsReportedToTheClientMasked(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("ok")))
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(context.Context, ui.Finish) error { return errors.New("database is down") }})
	body := rec.Body.String()
	if strings.Contains(body, "database") || !strings.Contains(body, `"type":"error","errorText":"An error occurred."`) {
		t.Errorf("body = %s, want a masked error chunk", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Error("stream did not end with [DONE]")
	}
}

func TestRespond_onFinishRunsBeforeTheFinishChunk(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("ok")))
	var bodyAtFinish string
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	ui.Respond(rec, req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(context.Context, ui.Finish) error {
		bodyAtFinish = rec.Body.String()
		return nil
	}})
	if strings.Contains(bodyAtFinish, `"type":"finish"`) || strings.Contains(bodyAtFinish, "[DONE]") {
		t.Errorf("OnFinish ran after the finish chunk was written:\n%s", bodyAtFinish)
	}
}

func TestRespond_exposesOnlyWhatIsEnabled(t *testing.T) {
	reasoning := ai.Reasoning{Text: "r", Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"signature":"S3CRET"}`)}}
	script := func() *aitest.Model {
		return scripted(aitest.Turn(aitest.Part(reasoning), aitest.StreamPart(ai.StepEnd{}), aitest.Usage(7, 9)))
	}
	t.Run("nothing by default", func(t *testing.T) {
		rec := respond(t, streamOf(script(), ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{})
		body := rec.Body.String()
		for _, hidden := range []string{"S3CRET", "usage", "warnings", "providerMetadata"} {
			if strings.Contains(body, hidden) {
				t.Errorf("default stream contains %q:\n%s", hidden, body)
			}
		}
	})
	t.Run("everything when enabled", func(t *testing.T) {
		m := aitest.NewModel(aitest.Turn(aitest.Part(reasoning), aitest.Usage(7, 9)))
		rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{Expose: ui.Expose{Provider: true, Usage: true}})
		body := rec.Body.String()
		if !strings.Contains(body, "S3CRET") || !strings.Contains(body, `"usage":{"cacheRead":0,"cacheWrite":0,"input":7,"output":9,"reasoning":0}`) {
			t.Errorf("enabled stream is missing provider data or usage:\n%s", body)
		}
	})
}

func TestRespond_abortedStreamPersistsCompletedStepsWithSyntheticResults(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("hi"), aitest.ToolCall("location", `{}`)))
	tools := []ai.Tool{ai.NewClientTool[struct{}]("location", "browser")}
	var finish ui.Finish
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	ctx, cancel := context.WithCancel(req.Context())
	ui.Stream(rec, req.WithContext(ctx), func(sctx context.Context, uw *ui.Writer) error {
		err := uw.Merge(ai.StreamText(sctx, m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}, Tools: tools}))
		cancel()
		return err
	}, ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	if !finish.Aborted || len(finish.New) != 2 {
		t.Fatalf("Finish = %+v, want an aborted response with the step and synthetic results", finish)
	}
	res := resultsIn(finish.New)
	if len(res) != 1 || !res[0].IsError || res[0].CallID != "call_1" {
		t.Errorf("synthetic results = %+v, want one error result for call_1", res)
	}
	if strings.Contains(rec.Body.String(), `"type":"error"`) {
		t.Errorf("aborted stream wrote an error chunk:\n%s", rec.Body.String())
	}
}

func TestRespond_clientDisconnectCancelsGeneration(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := scripted(aitest.Turn(aitest.Text("a"), aitest.Delay(time.Hour), aitest.Text("b")))
		req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
		ctx, cancel := context.WithCancel(req.Context())
		time.AfterFunc(time.Second, cancel)
		var finish ui.Finish
		rec := httptest.NewRecorder()
		ui.Respond(rec, req.WithContext(ctx), streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
		if !finish.Aborted || len(finish.New) != 0 {
			t.Errorf("Finish = %+v, want an aborted response without completed steps", finish)
		}
	})
}

type failingWriter struct {
	header http.Header
	writes int
}

func (w *failingWriter) Header() http.Header { return w.header }

func (w *failingWriter) WriteHeader(int) {}

func (w *failingWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes > 2 {
		return 0, errors.New("broken pipe")
	}
	return len(p), nil
}

func TestRespond_writeFailureStopsTheModel(t *testing.T) {
	m := scripted(aitest.Turn(aitest.TextChunks("a", "b", "c", "d", "e", "f")))
	var finish ui.Finish
	w := &failingWriter{header: http.Header{}}
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	ui.Respond(w, req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	if !finish.Aborted {
		t.Errorf("Finish = %+v, want an aborted response after the write error", finish)
	}
	if w.writes > 6 {
		t.Errorf("wrote %d frames after the client was gone, want the stream to stop", w.writes)
	}
}

func TestRespond_continuesTheLastAssistantMessage(t *testing.T) {
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.ToolCall{ID: "c1", Name: "echo", Input: json.RawMessage(`{"path":"p"}`)},
			ai.ToolApprovalRequest{ApprovalID: "a1", CallID: "c1"},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolApprovalResponse{ApprovalID: "a1", CallID: "c1", Approved: false}}},
	}
	m := scripted(aitest.Turn(aitest.Text("continued")))
	var finish ui.Finish
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: history, Tools: echoTools()}), ui.Options{History: history, OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	if start := chunksOf(t, rec.Body.String())[0]; start["messageId"] != nil {
		t.Errorf("start chunk = %v, want no messageId so the client keeps appending to its message", start)
	}
	if !slices.Contains(typesOf(chunksOf(t, rec.Body.String())), "tool-output-denied") {
		t.Errorf("stream = %s, want the denied call reported as tool-output-denied", rec.Body.String())
	}
	if got := textOf(finish.UI); got != "continued" || len(toolParts(finish.UI)) != 1 {
		t.Errorf("Finish.UI = %s, want the earlier tool part and the new text", mustJSON(t, finish.UI))
	}
}

func TestStream_writerDataAndMerge(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("hello")))
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	var finish ui.Finish
	ui.Stream(rec, req, func(ctx context.Context, uw *ui.Writer) error {
		if err := uw.Data("note", "first", ui.DataID("n")); err != nil {
			return err
		}
		if err := uw.Data("note", "second", ui.DataID("n")); err != nil {
			return err
		}
		if err := uw.Data("tick", 1, ui.Transient()); err != nil {
			return err
		}
		return uw.Merge(ai.StreamText(ctx, m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}))
	}, ui.Options{NewID: sequentialIDs(), OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	got := chunksOf(t, rec.Body.String())
	wantTypes := []string{"start", "data-note", "data-note", "data-tick", "start-step", "text-start", "text-delta", "text-end", "finish-step", "finish"}
	if diff := cmp.Diff(wantTypes, typesOf(got)); diff != "" {
		t.Errorf("chunk types mismatch (-want +got):\n%s", diff)
	}
	if got[3]["transient"] != true || got[1]["id"] != "n" || got[2]["data"] != "second" {
		t.Errorf("data chunks = %v", got[1:4])
	}
	var notes []ui.DataPart
	for _, p := range finish.UI.Parts {
		if dp, ok := p.(ui.DataPart); ok {
			notes = append(notes, dp)
		}
	}
	if len(notes) != 1 || string(notes[0].Data) != `"second"` {
		t.Errorf("Finish.UI data parts = %+v, want the replaced note only", notes)
	}
}

func TestStream_unencodableDataIsAnError(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	var dataErr error
	ui.Stream(rec, req, func(_ context.Context, uw *ui.Writer) error {
		dataErr = uw.Data("bad", func() {})
		return nil
	}, ui.Options{})
	if dataErr == nil {
		t.Error("Data(func) error = nil, want an encoding error")
	}
}

func TestStream_weakKeyIsAConfigurationError(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	var seen error
	ui.Stream(rec, req, func(context.Context, *ui.Writer) error { return nil }, ui.Options{Key: []byte("short"), OnError: func(err error) string { seen = err; return "nope" }})
	if rec.Code != http.StatusInternalServerError || !errors.Is(seen, ui.ErrWeakKey) || strings.Contains(rec.Body.String(), "short") {
		t.Errorf("status = %d, error = %v, body = %q, want a masked 500 for a weak key", rec.Code, seen, rec.Body.String())
	}
}

func TestRespond_signedMessageMetadataIsInTheFinishChunk(t *testing.T) {
	m := scripted(aitest.Turn(aitest.Text("hello")))
	rec := respond(t, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{Key: testKey()})
	chunks := chunksOf(t, rec.Body.String())
	meta, ok := chunks[len(chunks)-1]["messageMetadata"].(map[string]any)
	if !ok || meta["aigo"] == nil {
		t.Fatalf("finish chunk = %v, want signature metadata", chunks[len(chunks)-1])
	}
}

func TestRespondText_streamsPlainText(t *testing.T) {
	m := scripted(aitest.Turn(aitest.TextChunks("Hel", "lo")))
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	var finish ui.Finish
	ui.RespondText(rec, req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
	if rec.Body.String() != "Hello" || rec.Header().Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Errorf("body = %q, content type = %q", rec.Body.String(), rec.Header().Get("Content-Type"))
	}
	if len(finish.New) != 1 || textOf(finish.UI) != "Hello" {
		t.Errorf("Finish = %+v", finish)
	}
}
