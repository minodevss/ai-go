package ui_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/ui"
)

const compatTimeout = 90 * time.Second

var installOnce sync.Once

func compatDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(filepath.Join("testdata", "compat"))
	if err != nil {
		t.Fatalf("Abs() error = %v", err)
	}
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node is not installed")
	}
	npm, err := exec.LookPath("npm")
	if err != nil {
		t.Skip("npm is not installed")
	}
	var installErr error
	installOnce.Do(func() {
		if _, err := os.Stat(filepath.Join(dir, "node_modules", "ai", "package.json")); err == nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), compatTimeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, npm, "install", "--no-audit", "--no-fund")
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			installErr = errors.New("npm install failed: " + string(out))
		}
	})
	if installErr != nil {
		t.Skip(installErr)
	}
	return dir
}

type harnessResult struct {
	Status   string            `json:"status"`
	Messages []ui.Message      `json:"messages"`
	Errors   []string          `json:"errors"`
	Received []json.RawMessage `json:"received"`
	Message  *ui.Message       `json:"message"`
}

func runHarness(t *testing.T, base, scenario string, extra ...string) harnessResult {
	t.Helper()
	dir := compatDir(t)
	ctx, cancel := context.WithTimeout(t.Context(), compatTimeout)
	defer cancel()
	args := append([]string{"harness.mjs", base, scenario}, extra...)
	cmd := exec.CommandContext(ctx, "node", args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("node harness %s: %v\nstderr: %s\nstdout: %s", scenario, err, stderr.String(), stdout.String())
	}
	var res harnessResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("decode harness output: %v\n%s", err, stdout.String())
	}
	return res
}

type deleteInput struct {
	Path string `json:"path"`
}

type compatApp struct {
	t      *testing.T
	mu     sync.Mutex
	stored map[string][]ai.Message
	ran    []string
	seen   []ui.Trigger
	tools  []ai.Tool
	model  *aitest.Model
	key    []byte
	store  *ui.MemoryStore
	newID  func() string
}

func newCompatApp(t *testing.T, turns ...aitest.ModelOption) *compatApp {
	t.Helper()
	a := &compatApp{t: t, stored: map[string][]ai.Message{}, model: aitest.NewModel(turns...), key: testKey(), store: ui.NewMemoryStore(), newID: sequentialIDs()}
	a.tools = []ai.Tool{
		ai.NewTool("delete", "delete a file", func(_ context.Context, in deleteInput) (map[string]bool, error) {
			a.mu.Lock()
			defer a.mu.Unlock()
			a.ran = append(a.ran, in.Path)
			return map[string]bool{"deleted": true}, nil
		}).WithApproval(ai.AlwaysAsk),
		ai.NewClientTool[struct{}]("location", "ask the browser for the location"),
	}
	return a
}

func (a *compatApp) serve() *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /text", a.stateless)
	mux.HandleFunc("POST /history", a.stateless)
	mux.HandleFunc("POST /file", a.stateless)
	mux.HandleFunc("POST /approval", a.serverOwned)
	mux.HandleFunc("POST /clienttool", a.serverOwned)
	mux.HandleFunc("POST /approval-signed", a.signed)
	mux.HandleFunc("POST /error", a.stateless)
	mux.HandleFunc("POST /data", a.data)
	mux.HandleFunc("POST /plain", a.plain)
	mux.HandleFunc("POST /resume", a.resumable)
	mux.Handle("GET /resume/{id}/stream", ui.ResumeHandler(a.store, func(*http.Request) (string, error) { return "u1", nil }))
	srv := httptest.NewServer(mux)
	a.t.Cleanup(srv.Close)
	return srv
}

func (a *compatApp) request(w http.ResponseWriter, r *http.Request) (ui.Request, bool) {
	req, err := ui.ReadRequest(r, ui.Limits{})
	if err != nil {
		status := http.StatusBadRequest
		if re, ok := errors.AsType[*ui.RequestError](err); ok {
			status = re.Status
		}
		http.Error(w, err.Error(), status)
		return ui.Request{}, false
	}
	return req, true
}

func (a *compatApp) respond(w http.ResponseWriter, r *http.Request, msgs []ai.Message, opts ui.Options) {
	opts.History = msgs
	opts.NewID = a.newID
	ui.Respond(w, r, func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, a.model, ai.TextRequest{Messages: msgs, Tools: a.tools})
	}, opts)
}

func (a *compatApp) stateless(w http.ResponseWriter, r *http.Request) {
	req, ok := a.request(w, r)
	if !ok {
		return
	}
	a.mu.Lock()
	a.seen = append(a.seen, req.Trigger)
	a.mu.Unlock()
	msgs, err := ui.ToMessages(req, a.tools)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	a.respond(w, r, msgs, ui.Options{})
}

func (a *compatApp) save(id string, msgs []ai.Message) func(context.Context, ui.Finish) error {
	return func(_ context.Context, f ui.Finish) error {
		a.mu.Lock()
		defer a.mu.Unlock()
		a.stored[id] = slices.Concat(msgs, f.New)
		return nil
	}
}

func (a *compatApp) serverOwned(w http.ResponseWriter, r *http.Request) {
	req, ok := a.request(w, r)
	if !ok {
		return
	}
	a.mu.Lock()
	stored := a.stored[req.ID]
	a.mu.Unlock()
	msgs, err := ui.Merge(stored, req, a.tools)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	a.respond(w, r, msgs, ui.Options{OnFinish: a.save(req.ID, msgs)})
}

func (a *compatApp) signed(w http.ResponseWriter, r *http.Request) {
	req, ok := a.request(w, r)
	if !ok {
		return
	}
	msgs, err := ui.ToMessages(req, a.tools, ui.VerifyKey(a.key))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	a.respond(w, r, msgs, ui.Options{Key: a.key})
}

func (a *compatApp) data(w http.ResponseWriter, r *http.Request) {
	ui.Stream(w, r, func(ctx context.Context, uw *ui.Writer) error {
		if err := uw.Data("status", map[string]string{"phase": "thinking"}, ui.Transient()); err != nil {
			return err
		}
		if err := uw.Data("weather", map[string]any{"temp": 21}, ui.DataID("w1")); err != nil {
			return err
		}
		return uw.Merge(ai.StreamText(ctx, a.model, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}))
	}, ui.Options{NewID: a.newID})
}

func (a *compatApp) plain(w http.ResponseWriter, r *http.Request) {
	ui.RespondText(w, r, func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, a.model, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	}, ui.Options{})
}

func (a *compatApp) resumable(w http.ResponseWriter, r *http.Request) {
	req, ok := a.request(w, r)
	if !ok {
		return
	}
	a.respond(w, r, []ai.Message{ai.User("hi")}, ui.Options{Resume: &ui.Resume{Store: a.store, ChatID: req.ID, Owner: "u1"}})
}

func textOf(m ui.Message) string {
	var out strings.Builder
	for _, p := range m.Parts {
		if tp, ok := p.(ui.TextPart); ok {
			out.WriteString(tp.Text)
		}
	}
	return out.String()
}

func toolParts(m ui.Message) []ui.ToolPart {
	var out []ui.ToolPart
	for _, p := range m.Parts {
		if tp, ok := p.(ui.ToolPart); ok {
			out = append(out, tp)
		}
	}
	return out
}

func lastMessage(t *testing.T, res harnessResult) ui.Message {
	t.Helper()
	if len(res.Messages) == 0 {
		t.Fatalf("harness returned no messages: %+v", res)
	}
	return res.Messages[len(res.Messages)-1]
}

func TestCompat_text(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.TextChunks("Hel", "lo")))
	res := runHarness(t, app.serve().URL, "text")
	if got := textOf(lastMessage(t, res)); got != "Hello" || res.Status != "ready" || len(res.Errors) != 0 {
		t.Errorf("text = %q, status %q, errors %v, want Hello/ready", got, res.Status, res.Errors)
	}
}

func TestCompat_historyWithoutKeyDropsAssistantMessages(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("first")), aitest.Turn(aitest.Text("second")))
	res := runHarness(t, app.serve().URL, "history")
	if got := textOf(lastMessage(t, res)); got != "second" || len(res.Errors) != 0 {
		t.Errorf("history text = %q, errors %v, want second", got, res.Errors)
	}
	got := app.model.Calls()[1].Messages
	if len(got) != 2 || got[0].Role != ai.RoleUser || got[1].Role != ai.RoleUser {
		t.Errorf("second call saw %+v, want only the two user messages", got)
	}
}

func TestCompat_readerAcceptsOurStream(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Reasoning("hmm"), aitest.Text("done")))
	srv := app.serve()
	res := runHarness(t, srv.URL, "readMessages")
	if res.Message == nil || textOf(*res.Message) != "done" {
		t.Fatalf("readUIMessageStream message = %+v, want text done", res.Message)
	}
	var reasoning bool
	for _, p := range res.Message.Parts {
		if rp, ok := p.(ui.ReasoningPart); ok && rp.Text == "hmm" && rp.State == ui.StateDone {
			reasoning = true
		}
	}
	if !reasoning {
		t.Errorf("parts = %+v, want a finished reasoning part", res.Message.Parts)
	}
}

func approvalTurns() []aitest.ModelOption {
	return []aitest.ModelOption{
		aitest.Turn(aitest.ToolCall("delete", `{"path":"x"}`)),
		aitest.Turn(aitest.Text("Done")),
	}
}

func TestCompat_approval(t *testing.T) {
	tests := []struct {
		name     string
		scenario string
		extra    string
		wantRan  []string
		wantTool ui.ToolState
	}{
		{name: "server-owned approve", scenario: "approval", wantRan: []string{"x"}, wantTool: ui.ToolOutputAvailable},
		{name: "server-owned deny", scenario: "approval", extra: "deny", wantTool: ui.ToolOutputDenied},
		{name: "signed approve", scenario: "approval", extra: "signed", wantRan: []string{"x"}, wantTool: ui.ToolOutputAvailable},
		{name: "signed deny", scenario: "approval", extra: "signed-deny", wantTool: ui.ToolOutputDenied},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			app := newCompatApp(t, approvalTurns()...)
			res := runHarness(t, app.serve().URL, tt.scenario, tt.extra)
			if len(res.Errors) != 0 || res.Status != "ready" {
				t.Fatalf("errors = %v, status = %q", res.Errors, res.Status)
			}
			msg := lastMessage(t, res)
			tools := toolParts(msg)
			if len(tools) != 1 || tools[0].State != tt.wantTool || tools[0].Name != "delete" {
				t.Fatalf("tool parts = %+v, want one delete part in state %s; message = %s", tools, tt.wantTool, mustJSON(t, msg))
			}
			if tools[0].Approval == nil || tools[0].Approval.Approved == nil {
				t.Errorf("approval = %+v, want the client's decision kept", tools[0].Approval)
			}
			if got := textOf(msg); got != "Done" {
				t.Errorf("text = %q, want Done", got)
			}
			if !slices.Equal(app.ran, tt.wantRan) {
				t.Errorf("tool ran with %v, want %v", app.ran, tt.wantRan)
			}
			if len(res.Messages) != 2 {
				t.Errorf("client holds %d messages, want one user and one assistant message", len(res.Messages))
			}
		})
	}
}

func TestCompat_clientTool(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.ToolCall("location", `{}`)), aitest.Turn(aitest.Text("Paris")))
	res := runHarness(t, app.serve().URL, "clientTool")
	if len(res.Errors) != 0 {
		t.Fatalf("errors = %v", res.Errors)
	}
	msg := lastMessage(t, res)
	tools := toolParts(msg)
	if len(tools) != 1 || tools[0].State != ui.ToolOutputAvailable || textOf(msg) != "Paris" {
		t.Fatalf("message = %s, want the location output and the final text", mustJSON(t, msg))
	}
	calls := app.model.Calls()
	if len(calls) != 2 || len(resultsIn(calls[1].Messages)) != 1 {
		t.Errorf("second call = %+v, want the client's tool result", calls)
	}
}

func resultsIn(msgs []ai.Message) []ai.ToolResult {
	var out []ai.ToolResult
	for _, m := range msgs {
		for _, p := range m.Content {
			if r, ok := p.(ai.ToolResult); ok {
				out = append(out, r)
			}
		}
	}
	return out
}

func TestCompat_fileDataURL(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("seen")))
	res := runHarness(t, app.serve().URL, "file")
	if len(res.Errors) != 0 || textOf(lastMessage(t, res)) != "seen" {
		t.Fatalf("errors = %v, messages = %s", res.Errors, mustJSON(t, res.Messages))
	}
	user := app.model.Calls()[0].Messages[0]
	var file ai.File
	for _, p := range user.Content {
		if f, ok := p.(ai.File); ok {
			file = f
		}
	}
	if string(file.Data) != "hello" || file.MediaType != "text/plain" || file.Filename != "a.txt" {
		t.Errorf("model saw %+v, want the decoded text file", user.Content)
	}
}

func TestCompat_dataParts(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("ok")))
	res := runHarness(t, app.serve().URL, "data")
	msg := lastMessage(t, res)
	var stored *ui.DataPart
	for _, p := range msg.Parts {
		if dp, ok := p.(ui.DataPart); ok {
			stored = &dp
		}
	}
	if stored == nil || stored.Name != "weather" || stored.ID != "w1" || string(stored.Data) != `{"temp":21}` {
		t.Errorf("parts = %s, want the persistent weather data part", mustJSON(t, msg.Parts))
	}
	if len(res.Received) != 2 {
		t.Errorf("onData received %d chunks, want the transient and the persistent one", len(res.Received))
	}
	for _, p := range msg.Parts {
		if dp, ok := p.(ui.DataPart); ok && dp.Name == "status" {
			t.Error("transient data part was stored in the message")
		}
	}
}

func TestCompat_errorsAreMasked(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("partial"), aitest.Err(errors.New("db password hunter2"))))
	res := runHarness(t, app.serve().URL, "error")
	if len(res.Errors) != 1 || res.Errors[0] != "An error occurred." {
		t.Errorf("client errors = %v, want the generic message", res.Errors)
	}
}

func TestCompat_textStreamTransport(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.TextChunks("plain ", "text")))
	res := runHarness(t, app.serve().URL, "textStream")
	if got := textOf(lastMessage(t, res)); got != "plain text" || len(res.Errors) != 0 {
		t.Errorf("text = %q, errors %v, want plain text", got, res.Errors)
	}
}

func TestCompat_resume(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("part one "), aitest.Delay(600*time.Millisecond), aitest.Text("part two")))
	res := runHarness(t, app.serve().URL, "resume")
	if got := textOf(lastMessage(t, res)); got != "part one part two" || len(res.Errors) != 0 {
		t.Errorf("resumed text = %q, errors %v, want the whole response", got, res.Errors)
	}
}

func TestCompat_resumeWithoutStreamIsNoContent(t *testing.T) {
	app := newCompatApp(t)
	res := runHarness(t, app.serve().URL, "resumeNone")
	if len(res.Messages) != 0 || res.Status != "ready" || len(res.Errors) != 0 {
		t.Errorf("result = %+v, want an idle chat", res)
	}
}

func TestCompat_regenerate(t *testing.T) {
	app := newCompatApp(t, aitest.Turn(aitest.Text("first")), aitest.Turn(aitest.Text("second")))
	res := runHarness(t, app.serve().URL, "regenerate")
	if len(res.Messages) != 2 || textOf(lastMessage(t, res)) != "second" || len(res.Errors) != 0 {
		t.Errorf("messages = %s, errors %v, want the user message and the regenerated reply", mustJSON(t, res.Messages), res.Errors)
	}
	if diff := cmp.Diff([]ui.Trigger{ui.TriggerSubmit, ui.TriggerRegenerate}, app.seen); diff != "" {
		t.Errorf("triggers mismatch (-want +got):\n%s", diff)
	}
	if got := app.model.Calls()[1].Messages; len(got) != 1 {
		t.Errorf("regenerate call saw %d messages, want only the user message", len(got))
	}
}
