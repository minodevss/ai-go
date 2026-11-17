package ui_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/ui"
)

func drain(t *testing.T, r ui.StreamReader) []string {
	t.Helper()
	var out []string
	for {
		f, err := r.Next(t.Context())
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		out = append(out, string(f))
	}
}

func spec(owner, chat string, stop context.CancelFunc) ui.StreamSpec {
	return ui.StreamSpec{ChatID: chat, Owner: owner, MaxConcurrent: 2, MaxBytes: 1 << 20, Stop: stop}
}

func TestMemoryStore_tailsFramesFromTheStart(t *testing.T) {
	s := ui.NewMemoryStore()
	w, err := s.Begin(t.Context(), spec("u1", "c1", func() {}))
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	if err := w.Append(t.Context(), []byte("one")); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	r, err := s.Attach(t.Context(), "c1", "u1")
	if err != nil {
		t.Fatalf("Attach() error = %v", err)
	}
	if err := w.Append(t.Context(), []byte("two")); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if err := w.Close(t.Context(), []byte("end")); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if diff := cmp.Diff([]string{"one", "two", "end"}, drain(t, r)); diff != "" {
		t.Errorf("frames mismatch (-want +got):\n%s", diff)
	}
	if _, err := s.Attach(t.Context(), "c1", "u1"); !errors.Is(err, ui.ErrNoStream) {
		t.Errorf("Attach() after Close error = %v, want %v", err, ui.ErrNoStream)
	}
	if err := w.Append(t.Context(), []byte("late")); !errors.Is(err, ui.ErrNoStream) {
		t.Errorf("Append() after Close error = %v, want %v", err, ui.ErrNoStream)
	}
}

func TestMemoryStore_readerBlocksUntilData(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := ui.NewMemoryStore()
		w, _ := s.Begin(t.Context(), spec("u1", "c1", func() {}))
		r, _ := s.Attach(t.Context(), "c1", "u1")
		var got []byte
		var wg sync.WaitGroup
		wg.Go(func() { got, _ = r.Next(t.Context()) })
		time.Sleep(time.Second)
		synctest.Wait()
		if got != nil {
			t.Fatalf("Next() returned %q before any frame", got)
		}
		if err := w.Append(t.Context(), []byte("x")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		wg.Wait()
		if string(got) != "x" {
			t.Errorf("Next() = %q, want x", got)
		}
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestMemoryStore_readerHonorsContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := ui.NewMemoryStore()
		w, _ := s.Begin(t.Context(), spec("u1", "c1", func() {}))
		r, _ := s.Attach(t.Context(), "c1", "u1")
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		defer cancel()
		if _, err := r.Next(ctx); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Next() error = %v, want %v", err, context.DeadlineExceeded)
		}
		if err := r.Close(); err != nil {
			t.Errorf("reader Close() error = %v", err)
		}
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("writer Close() error = %v", err)
		}
	})
}

func TestMemoryStore_ownersAreIsolated(t *testing.T) {
	s := ui.NewMemoryStore()
	stopped := false
	w, err := s.Begin(t.Context(), spec("alice", "c1", func() { stopped = true }))
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	if _, err := s.Attach(t.Context(), "c1", "mallory"); !errors.Is(err, ui.ErrNoStream) {
		t.Errorf("Attach(other owner) error = %v, want %v", err, ui.ErrNoStream)
	}
	if err := ui.Stop(t.Context(), s, "c1", "mallory"); !errors.Is(err, ui.ErrNoStream) || stopped {
		t.Errorf("Stop(other owner) error = %v, stopped = %v, want ErrNoStream and no stop", err, stopped)
	}
	if _, err := s.Begin(t.Context(), spec("mallory", "c1", func() {})); err != nil {
		t.Errorf("Begin() for another owner with the same chat id error = %v, want an independent stream", err)
	}
	if err := ui.Stop(t.Context(), s, "c1", "alice"); err != nil || !stopped {
		t.Errorf("Stop(owner) error = %v, stopped = %v, want the stream stopped", err, stopped)
	}
	if err := w.Close(t.Context()); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestMemoryStore_limits(t *testing.T) {
	t.Run("one stream per chat", func(t *testing.T) {
		s := ui.NewMemoryStore()
		if _, err := s.Begin(t.Context(), spec("u1", "c1", func() {})); err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if _, err := s.Begin(t.Context(), spec("u1", "c1", func() {})); !errors.Is(err, ui.ErrBusy) {
			t.Errorf("second Begin() error = %v, want %v", err, ui.ErrBusy)
		}
	})
	t.Run("streams per owner", func(t *testing.T) {
		s := ui.NewMemoryStore()
		var writers []ui.StreamWriter
		for _, chat := range []string{"a", "b"} {
			w, err := s.Begin(t.Context(), spec("u1", chat, func() {}))
			if err != nil {
				t.Fatalf("Begin(%s) error = %v", chat, err)
			}
			writers = append(writers, w)
		}
		if _, err := s.Begin(t.Context(), spec("u1", "c", func() {})); !errors.Is(err, ui.ErrTooManyStreams) {
			t.Errorf("third Begin() error = %v, want %v", err, ui.ErrTooManyStreams)
		}
		if _, err := s.Begin(t.Context(), spec("u2", "c", func() {})); err != nil {
			t.Errorf("Begin() for another owner error = %v", err)
		}
		if err := writers[0].Close(t.Context()); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if _, err := s.Begin(t.Context(), spec("u1", "c", func() {})); err != nil {
			t.Errorf("Begin() after a stream ended error = %v, want a free slot", err)
		}
	})
	t.Run("bytes", func(t *testing.T) {
		s := ui.NewMemoryStore()
		stopped := false
		sp := spec("u1", "c1", func() { stopped = true })
		sp.MaxBytes = 10
		w, _ := s.Begin(t.Context(), sp)
		if err := w.Append(t.Context(), []byte("123456")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		if err := w.Append(t.Context(), []byte("123456")); !errors.Is(err, ui.ErrStreamTooLarge) || !stopped {
			t.Errorf("Append() over the limit error = %v, stopped = %v, want ErrStreamTooLarge and a stop", err, stopped)
		}
		if err := w.Close(t.Context(), []byte("final frames are always kept")); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
}

func TestMemoryStore_ttlExpiresIdleStreams(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := ui.NewMemoryStore(ui.WithTTL(time.Minute))
		stopped := make(chan struct{})
		w, err := s.Begin(t.Context(), spec("u1", "c1", func() { close(stopped) }))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		r, _ := s.Attach(t.Context(), "c1", "u1")
		time.Sleep(50 * time.Second)
		if err := w.Append(t.Context(), []byte("keepalive")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		time.Sleep(50 * time.Second)
		select {
		case <-stopped:
			t.Fatal("stream expired although a frame arrived within the TTL")
		default:
		}
		time.Sleep(11 * time.Second)
		<-stopped
		if diff := cmp.Diff([]string{"keepalive"}, drain(t, r)); diff != "" {
			t.Errorf("frames mismatch (-want +got):\n%s", diff)
		}
		if _, err := s.Attach(t.Context(), "c1", "u1"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Attach() after expiry error = %v, want %v", err, ui.ErrNoStream)
		}
		if err := w.Append(t.Context(), []byte("late")); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Append() after expiry error = %v, want %v", err, ui.ErrNoStream)
		}
	})
}

func resumeOptions(store ui.StreamStore, finish *ui.Finish) ui.Options {
	return ui.Options{
		NewID:    sequentialIDs(),
		Resume:   &ui.Resume{Store: store, ChatID: "chat-1", Owner: "u1"},
		OnFinish: func(_ context.Context, f ui.Finish) error { *finish = f; return nil },
	}
}

func TestRespond_resumeGenerationOutlivesTheRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store := ui.NewMemoryStore()
		m := scripted(aitest.Turn(aitest.Text("part one "), aitest.Delay(time.Minute), aitest.Text("part two")))
		var finish ui.Finish
		var factoryCtx context.Context
		factory := func(ctx context.Context) *ai.TextStream {
			factoryCtx = ctx
			return ai.StreamText(ctx, m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}})
		}
		reqCtx, cancel := context.WithCancel(t.Context())
		req := httptest.NewRequestWithContext(reqCtx, http.MethodPost, "/chat", nil)
		rec := httptest.NewRecorder()
		var wg sync.WaitGroup
		wg.Go(func() { ui.Respond(rec, req, factory, resumeOptions(store, &finish)) })
		time.Sleep(time.Second)
		synctest.Wait()
		cancel()
		wg.Wait()
		if factoryCtx.Err() != nil {
			t.Fatal("generation context was canceled with the request")
		}
		reader, err := store.Attach(t.Context(), "chat-1", "u1")
		if err != nil {
			t.Fatalf("Attach() error = %v", err)
		}
		frames := strings.Join(drain(t, reader), "")
		got := typesOf(chunksOf(t, frames))
		want := []string{"start", "start-step", "text-start", "text-delta", "text-end", "text-start", "text-delta", "text-end", "finish-step", "finish"}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("resumed chunk types mismatch (-want +got):\n%s", diff)
		}
		if finish.Aborted || len(finish.New) != 1 || textOf(finish.UI) != "part one part two" {
			t.Errorf("Finish = %+v, want the whole response persisted after the client left", finish)
		}
	})
}

func TestRespond_withoutResumeRequestCancelStopsGeneration(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := scripted(aitest.Turn(aitest.Text("a"), aitest.Delay(time.Minute), aitest.Text("b")))
		var finish ui.Finish
		reqCtx, cancel := context.WithCancel(t.Context())
		req := httptest.NewRequestWithContext(reqCtx, http.MethodPost, "/chat", nil)
		time.AfterFunc(time.Second, cancel)
		ui.Respond(httptest.NewRecorder(), req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), ui.Options{OnFinish: func(_ context.Context, f ui.Finish) error { finish = f; return nil }})
		if !finish.Aborted {
			t.Errorf("Finish = %+v, want an aborted response", finish)
		}
	})
}

func TestRespond_resumeMaxDuration(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store := ui.NewMemoryStore()
		m := scripted(aitest.Turn(aitest.Text("a"), aitest.Delay(time.Hour), aitest.Text("b")))
		var finish ui.Finish
		opts := resumeOptions(store, &finish)
		opts.Resume.MaxDuration = time.Minute
		start := time.Now()
		req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
		ui.Respond(httptest.NewRecorder(), req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), opts)
		if !finish.Aborted || time.Since(start) != time.Minute {
			t.Errorf("Finish = %+v after %s, want an abort at the one minute limit", finish, time.Since(start))
		}
	})
}

func TestRespond_stopThroughTheStore(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store := ui.NewMemoryStore()
		m := scripted(aitest.Turn(aitest.Text("a"), aitest.Delay(time.Hour), aitest.Text("b")))
		var finish ui.Finish
		req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
		rec := httptest.NewRecorder()
		var wg sync.WaitGroup
		wg.Go(func() {
			ui.Respond(rec, req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), resumeOptions(store, &finish))
		})
		time.Sleep(time.Second)
		synctest.Wait()
		if err := ui.Stop(t.Context(), store, "chat-1", "mallory"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Stop(other owner) error = %v, want %v", err, ui.ErrNoStream)
		}
		synctest.Wait()
		if finish.UI.ID != "" || strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatal("another owner's Stop ended the generation")
		}
		if err := ui.Stop(t.Context(), store, "chat-1", "u1"); err != nil {
			t.Fatalf("Stop() error = %v", err)
		}
		wg.Wait()
		if !finish.Aborted {
			t.Errorf("Finish = %+v, want an aborted response", finish)
		}
		if !strings.HasSuffix(rec.Body.String(), "data: [DONE]\n\n") {
			t.Errorf("stopped stream did not end with [DONE]:\n%s", rec.Body.String())
		}
	})
}

func TestRespond_resumeLimitsAnswerWithStatus(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store := ui.NewMemoryStore()
		slow := func() *aitest.Model { return scripted(aitest.Turn(aitest.Delay(time.Hour), aitest.Text("x"))) }
		var wg sync.WaitGroup
		start := func(chat, owner string, rec *httptest.ResponseRecorder) {
			opts := ui.Options{Resume: &ui.Resume{Store: store, ChatID: chat, Owner: owner, MaxConcurrent: 1}}
			req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
			wg.Go(func() {
				ui.Respond(rec, req, streamOf(slow(), ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), opts)
			})
		}
		first := httptest.NewRecorder()
		start("a", "u1", first)
		time.Sleep(time.Second)
		synctest.Wait()
		sameChat, otherChat := httptest.NewRecorder(), httptest.NewRecorder()
		start("a", "u1", sameChat)
		start("b", "u1", otherChat)
		synctest.Wait()
		if sameChat.Code != http.StatusConflict || otherChat.Code != http.StatusTooManyRequests {
			t.Errorf("statuses = %d, %d, want 409 for the busy chat and 429 over the owner limit", sameChat.Code, otherChat.Code)
		}
		if err := ui.Stop(t.Context(), store, "a", "u1"); err != nil {
			t.Fatalf("Stop() error = %v", err)
		}
		wg.Wait()
	})
}

func TestRespond_resumeMaxBytes(t *testing.T) {
	store := ui.NewMemoryStore()
	m := scripted(aitest.Turn(aitest.TextChunks(strings.Repeat("x", 100), strings.Repeat("y", 100), strings.Repeat("z", 100))))
	var finish ui.Finish
	opts := resumeOptions(store, &finish)
	opts.Resume.MaxBytes = 300
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	rec := httptest.NewRecorder()
	ui.Respond(rec, req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), opts)
	body := rec.Body.String()
	if !finish.Aborted || !strings.Contains(body, `"type":"error"`) || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Errorf("aborted = %v, body = %s, want an error chunk and a closed stream", finish.Aborted, body)
	}
	if strings.Contains(body, strings.Repeat("z", 100)) {
		t.Error("stream continued past the byte limit")
	}
}

func TestRespond_resumeNeedsOwnerAndChat(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
	ui.Respond(rec, req, streamOf(scripted(), ai.TextRequest{}), ui.Options{Resume: &ui.Resume{Store: ui.NewMemoryStore()}})
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", rec.Code)
	}
}

func resumeServer(t *testing.T, store ui.StreamStore, ownerOf func(*http.Request) (string, error)) *http.ServeMux {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("GET /chat/{id}/stream", ui.ResumeHandler(store, ownerOf))
	return mux
}

func TestResumeHandler(t *testing.T) {
	store := ui.NewMemoryStore()
	w, err := store.Begin(t.Context(), spec("alice", "c1", func() {}))
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	if err := w.Append(t.Context(), []byte("data: {\"type\":\"start\"}\n\n")); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if err := w.Close(t.Context(), []byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	live, err := store.Begin(t.Context(), spec("alice", "live", func() {}))
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	if err := live.Append(t.Context(), []byte("data: {\"type\":\"start\"}\n\n")); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	t.Cleanup(func() {
		if err := live.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	owner := func(r *http.Request) (string, error) {
		if u := r.Header.Get("X-User"); u != "" {
			return u, nil
		}
		return "", errors.New("no session")
	}
	mux := resumeServer(t, store, owner)
	get := func(path, user string) *httptest.ResponseRecorder {
		req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, path, nil)
		if user != "" {
			req.Header.Set("X-User", user)
		}
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		return rec
	}
	tests := []struct {
		name       string
		path, user string
		wantStatus int
	}{
		{name: "finished stream is gone", path: "/chat/c1/stream", user: "alice", wantStatus: http.StatusNoContent},
		{name: "unknown chat", path: "/chat/zzz/stream", user: "alice", wantStatus: http.StatusNoContent},
		{name: "owner mismatch", path: "/chat/live/stream", user: "mallory", wantStatus: http.StatusNoContent},
		{name: "no session", path: "/chat/live/stream", wantStatus: http.StatusUnauthorized},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if rec := get(tt.path, tt.user); rec.Code != tt.wantStatus {
				t.Errorf("GET %s status = %d, want %d", tt.path, rec.Code, tt.wantStatus)
			}
		})
	}
}

func TestResumeHandler_replaysTheActiveStream(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store := ui.NewMemoryStore()
		m := scripted(aitest.Turn(aitest.Text("hello"), aitest.Delay(time.Minute), aitest.Text(" world")))
		var finish ui.Finish
		mux := resumeServer(t, store, func(*http.Request) (string, error) { return "u1", nil })
		var wg sync.WaitGroup
		wg.Go(func() {
			req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
			ui.Respond(httptest.NewRecorder(), req, streamOf(m, ai.TextRequest{Messages: []ai.Message{ai.User("q")}}), resumeOptions(store, &finish))
		})
		time.Sleep(time.Second)
		synctest.Wait()
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/chat/chat-1/stream", nil))
		wg.Wait()
		if rec.Header().Get("X-Vercel-Ai-Ui-Message-Stream") != "v1" || rec.Header().Get("Content-Type") != "text/event-stream" {
			t.Errorf("headers = %v", rec.Header())
		}
		if got := typesOf(chunksOf(t, rec.Body.String())); got[0] != "start" || got[len(got)-1] != "finish" {
			t.Errorf("replayed chunk types = %v, want the full stream from start to finish", got)
		}
		if !strings.Contains(rec.Body.String(), `"delta":" world"`) {
			t.Error("replay is missing the chunks written after the reconnect")
		}
	})
}
