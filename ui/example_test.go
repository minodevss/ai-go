package ui_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/ui"
)

const exampleBody = `{"id":"chat-1","trigger":"submit-message","messages":[{"id":"u1","role":"user","parts":[{"type":"text","text":"Where is my order?"}]}]}`

func examplePost(body string) *http.Request {
	req := httptest.NewRequestWithContext(context.Background(), http.MethodPost, "/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func exampleChunkTypes(body string) []string {
	var out []string
	for frame := range strings.SplitSeq(body, "\n\n") {
		if frame == "" {
			continue
		}
		payload := strings.TrimPrefix(frame, "data: ")
		if payload == "[DONE]" {
			out = append(out, payload)
			continue
		}
		kind, _, _ := strings.Cut(strings.TrimPrefix(payload, `{"type":"`), `"`)
		out = append(out, kind)
	}
	return out
}

func ExampleReadRequest() {
	req, err := ui.ReadRequest(examplePost(exampleBody), ui.Limits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(req.ID, req.Trigger, len(req.Messages))

	_, err = ui.ReadRequest(examplePost(`{"messages":[{"id":"u","role":"user","parts":[{"type":"file","mediaType":"image/png","url":"https://internal.example/x.png"}]}]}`), ui.Limits{})
	if reqErr, ok := errors.AsType[*ui.RequestError](err); ok {
		fmt.Println(reqErr.Status, err)
	}
	// Output:
	// chat-1 submit-message 1
	// 400 ui: file not allowed: file urls are not accepted
}

func ExampleMerge() {
	req, _ := ui.ReadRequest(examplePost(exampleBody), ui.Limits{})
	stored := []ai.Message{ai.User("Hi"), ai.Assistant("Hello! How can I help?")}
	msgs, err := ui.Merge(stored, req, nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, m := range msgs {
		text, _ := m.Content[0].(ai.Text)
		fmt.Println(m.Role, text.Text)
	}
	// Output:
	// user Hi
	// assistant Hello! How can I help?
	// user Where is my order?
}

func ExampleToMessages() {
	req, _ := ui.ReadRequest(examplePost(exampleBody), ui.Limits{})
	msgs, err := ui.ToMessages(req, nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(msgs), msgs[0].Role)
	// Output: 1 user
}

func ExampleVerifyKey() {
	key := []byte(strings.Repeat("k", 32))
	model := aitest.NewModel(aitest.Turn(aitest.Text("It shipped yesterday.")))
	var reply ui.Message
	rec := httptest.NewRecorder()
	ui.Respond(rec, examplePost(exampleBody), func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Where is my order?")}})
	}, ui.Options{Key: key, OnFinish: func(_ context.Context, f ui.Finish) error {
		reply = f.UI
		return nil
	}})

	next := ui.Request{Messages: []ui.Message{
		{ID: "u1", Role: ui.RoleUser, Parts: []ui.Part{ui.TextPart{Text: "Where is my order?"}}},
		reply,
		{ID: "u2", Role: ui.RoleUser, Parts: []ui.Part{ui.TextPart{Text: "Thanks"}}},
	}}
	msgs, err := ui.ToMessages(next, nil, ui.VerifyKey(key))
	fmt.Println(len(msgs), err)

	next.Messages[1].Metadata = nil
	_, err = ui.ToMessages(next, nil, ui.VerifyKey(key))
	fmt.Println(errors.Is(err, ui.ErrBadSignature))
	// Output:
	// 3 <nil>
	// true
}

func ExampleMaxAge() {
	key := []byte(strings.Repeat("k", 32))
	req := ui.Request{}
	_, err := ui.ToMessages(req, nil, ui.VerifyKey(key), ui.MaxAge(10*time.Minute))
	fmt.Println(err)
	// Output: <nil>
}

func ExampleFromMessages() {
	history := []ai.Message{ai.User("Hi"), ai.Assistant("Hello")}
	for _, m := range ui.FromMessages(history) {
		fmt.Println(m.Role, len(m.Parts))
	}
	// Output:
	// user 1
	// assistant 2
}

func ExampleToAI() {
	history := []ai.Message{ai.User("Hi"), ai.Assistant("Hello")}
	back, err := ui.ToAI(ui.FromMessages(history))
	fmt.Println(len(back), err)
	// Output: 2 <nil>
}

func ExampleMessage_WithoutProvider() {
	history := []ai.Message{{Role: ai.RoleAssistant, Content: ai.Parts{ai.Reasoning{Text: "hmm", Provider: ai.ProviderData{"anthropic": []byte(`{"signature":"s"}`)}}}}}
	m := ui.FromMessages(history)[0].WithoutProvider()
	reasoning, _ := m.Parts[1].(ui.ReasoningPart)
	fmt.Println(reasoning.Provider == nil)
	// Output: true
}

func ExampleRespond() {
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks("It ", "shipped.")))
	rec := httptest.NewRecorder()
	ui.Respond(rec, examplePost(exampleBody), func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Where is my order?")}})
	}, ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error {
		fmt.Println("persist", len(f.New), "message")
		return nil
	}})
	fmt.Println(rec.Header().Get("x-vercel-ai-ui-message-stream"))
	fmt.Println(strings.Join(exampleChunkTypes(rec.Body.String()), " "))
	// Output:
	// persist 1 message
	// v1
	// start start-step text-start text-delta text-delta text-end finish-step finish [DONE]
}

func ExampleStream() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("Sunny.")))
	rec := httptest.NewRecorder()
	ui.Stream(rec, examplePost(exampleBody), func(ctx context.Context, uw *ui.Writer) error {
		if err := uw.Data("status", "looking up the forecast", ui.Transient()); err != nil {
			return err
		}
		return uw.Merge(ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Weather?")}}))
	}, ui.Options{})
	fmt.Println(strings.Join(exampleChunkTypes(rec.Body.String()), " "))
	// Output: start data-status start-step text-start text-delta text-end finish-step finish [DONE]
}

func ExampleWriter_Data() {
	rec := httptest.NewRecorder()
	ui.Stream(rec, examplePost(exampleBody), func(_ context.Context, uw *ui.Writer) error {
		return uw.Data("weather", map[string]int{"celsius": 21}, ui.DataID("w1"))
	}, ui.Options{})
	fmt.Println(strings.Split(rec.Body.String(), "\n\n")[1])
	// Output: data: {"type":"data-weather","id":"w1","data":{"celsius":21}}
}

func ExampleWriter_Merge() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("Done.")))
	rec := httptest.NewRecorder()
	ui.Stream(rec, examplePost(exampleBody), func(ctx context.Context, uw *ui.Writer) error {
		return uw.Merge(ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Go")}}))
	}, ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error {
		fmt.Println(f.FinishReason, len(f.New))
		return nil
	}})
	// Output: stop 1
}

func ExampleTransient() {
	rec := httptest.NewRecorder()
	ui.Stream(rec, examplePost(exampleBody), func(_ context.Context, uw *ui.Writer) error {
		return uw.Data("tick", 1, ui.Transient())
	}, ui.Options{})
	fmt.Println(strings.Contains(rec.Body.String(), `"transient":true`))
	// Output: true
}

func ExampleDataID() {
	rec := httptest.NewRecorder()
	ui.Stream(rec, examplePost(exampleBody), func(_ context.Context, uw *ui.Writer) error {
		return uw.Data("progress", 50, ui.DataID("job-1"))
	}, ui.Options{})
	fmt.Println(strings.Contains(rec.Body.String(), `"id":"job-1"`))
	// Output: true
}

func ExampleRespondText() {
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks("plain ", "text")))
	rec := httptest.NewRecorder()
	ui.RespondText(rec, examplePost(exampleBody), func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Hi")}})
	}, ui.Options{})
	fmt.Println(rec.Body.String())
	// Output: plain text
}

func ExampleNewMemoryStore() {
	store := ui.NewMemoryStore()
	model := aitest.NewModel(aitest.Turn(aitest.Text("Resumable.")))
	rec := httptest.NewRecorder()
	ui.Respond(rec, examplePost(exampleBody), func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Hi")}})
	}, ui.Options{Resume: &ui.Resume{Store: store, ChatID: "chat-1", Owner: "user-1"}})
	fmt.Println(strings.HasSuffix(rec.Body.String(), "data: [DONE]\n\n"))
	// Output: true
}

func ExampleWithTTL() {
	store := ui.NewMemoryStore(ui.WithTTL(5 * time.Minute))
	_, err := store.Attach(context.Background(), "chat-1", "user-1")
	fmt.Println(errors.Is(err, ui.ErrNoStream))
	// Output: true
}

func ExampleResumeHandler() {
	store := ui.NewMemoryStore()
	mux := http.NewServeMux()
	mux.Handle("GET /chat/{id}/stream", ui.ResumeHandler(store, func(r *http.Request) (string, error) {
		return r.Header.Get("X-User"), nil
	}))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequestWithContext(context.Background(), http.MethodGet, "/chat/chat-1/stream", nil))
	fmt.Println(rec.Code)
	// Output: 204
}

func ExampleStop() {
	store := ui.NewMemoryStore()
	err := ui.Stop(context.Background(), store, "chat-1", "user-1")
	fmt.Println(errors.Is(err, ui.ErrNoStream))
	// Output: true
}

func ExampleChat() {
	key := []byte(strings.Repeat("k", 32))
	model := aitest.NewModel(aitest.Turn(aitest.Text("It shipped yesterday.")))
	var reply ui.Message
	rec := httptest.NewRecorder()
	ui.Respond(rec, examplePost(exampleBody), func(ctx context.Context) *ai.TextStream {
		return ai.StreamText(ctx, model, ai.TextRequest{Messages: []ai.Message{ai.User("Where is my order?")}})
	}, ui.Options{Key: key, ChatID: "chat-1", Owner: "user-1", OnFinish: func(_ context.Context, f ui.Finish) error {
		reply = f.UI
		return nil
	}})

	next := ui.Request{Messages: []ui.Message{
		{ID: "u1", Role: ui.RoleUser, Parts: []ui.Part{ui.TextPart{Text: "Where is my order?"}}},
		reply,
		{ID: "u2", Role: ui.RoleUser, Parts: []ui.Part{ui.TextPart{Text: "Thanks"}}},
	}}
	msgs, err := ui.ToMessages(next, nil, ui.VerifyKey(key), ui.Chat("chat-1", "user-1"))
	fmt.Println(len(msgs), err)

	_, err = ui.ToMessages(next, nil, ui.VerifyKey(key), ui.Chat("chat-1", "user-2"))
	fmt.Println(errors.Is(err, ui.ErrBadSignature))
	// Output:
	// 3 <nil>
	// true
}

func ExampleMemoryStore_Begin() {
	store := ui.NewMemoryStore()
	ctx := context.Background()
	spec := ui.StreamSpec{ChatID: "chat-1", Owner: "user-1", Stop: func() {}}
	w, err := store.Begin(ctx, spec)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer w.Close(ctx)
	_, err = store.Begin(ctx, spec)
	fmt.Println(errors.Is(err, ui.ErrBusy))
	// Output: true
}

func ExampleMemoryStore_Attach() {
	store := ui.NewMemoryStore()
	ctx := context.Background()
	w, err := store.Begin(ctx, ui.StreamSpec{ChatID: "chat-1", Owner: "user-1", Stop: func() {}})
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := w.Append(ctx, []byte("frame")); err != nil {
		fmt.Println(err)
		return
	}
	r, err := store.Attach(ctx, "chat-1", "user-1")
	if err != nil {
		fmt.Println(err)
		return
	}
	frame, err := r.Next(ctx)
	fmt.Println(string(frame), err)
	_, err = store.Attach(ctx, "chat-1", "user-2")
	fmt.Println(errors.Is(err, ui.ErrNoStream))
	// Output:
	// frame <nil>
	// true
}

func ExampleMemoryStore_Stop() {
	store := ui.NewMemoryStore()
	ctx := context.Background()
	stopped := false
	if _, err := store.Begin(ctx, ui.StreamSpec{ChatID: "chat-1", Owner: "user-1", Stop: func() { stopped = true }}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(store.Stop(ctx, "chat-1", "user-1"), stopped)
	// Output: <nil> true
}
