package uiredis_test

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/go-cmp/cmp"
	"github.com/redis/go-redis/v9"

	"github.com/minodevss/ai-go/ui"
	"github.com/minodevss/ai-go/uiredis"
)

const waitLimit = 10 * time.Second

type backend struct {
	name string
	addr string
}

func backends(t *testing.T) []backend {
	t.Helper()
	mini := miniredis.RunT(t)
	out := []backend{{name: "miniredis", addr: mini.Addr()}}
	if addr := os.Getenv("UIREDIS_TEST_ADDR"); addr != "" {
		return append(out, backend{name: "redis " + addr, addr: addr})
	}
	if addr, ok := startRedisServer(t); ok {
		out = append(out, backend{name: "redis-server", addr: addr})
	}
	return out
}

func startRedisServer(t *testing.T) (string, bool) {
	t.Helper()
	bin, err := exec.LookPath("redis-server")
	if err != nil {
		return "", false
	}
	l, err := (&net.ListenConfig{}).Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen() error = %v", err)
	}
	tcp, ok := l.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("listener address %v is not TCP", l.Addr())
	}
	port := strconv.Itoa(tcp.Port)
	if err := l.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := os.MkdirAll("../run", 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	dir, err := os.MkdirTemp("../run", "uiredis-") //nolint:usetesting // scratch files belong in the project's ignored run/ directory
	if err != nil {
		t.Fatalf("MkdirTemp() error = %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	cmd := exec.CommandContext(t.Context(), bin, "--port", port, "--bind", "127.0.0.1", "--dir", filepath.Clean(dir), "--save", "", "--appendonly", "no")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start redis-server: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	addr := "127.0.0.1:" + port
	client := redis.NewClient(&redis.Options{Addr: addr})
	defer client.Close()
	deadline := time.Now().Add(waitLimit)
	for client.Ping(t.Context()).Err() != nil {
		if time.Now().After(deadline) {
			t.Fatal("redis-server did not become ready")
		}
		time.Sleep(20 * time.Millisecond)
	}
	return addr, true
}

func eachStore(t *testing.T, fn func(t *testing.T, newStore func(opts ...uiredis.Option) *uiredis.Store)) {
	t.Helper()
	for _, b := range backends(t) {
		t.Run(b.name, func(t *testing.T) {
			prefix := "t" + strconv.FormatInt(time.Now().UnixNano(), 36)
			fn(t, func(opts ...uiredis.Option) *uiredis.Store {
				client := redis.NewClient(&redis.Options{Addr: b.addr})
				t.Cleanup(func() { _ = client.Close() })
				return uiredis.New(client, append([]uiredis.Option{uiredis.WithPrefix(prefix)}, opts...)...)
			})
		})
	}
}

func drain(t *testing.T, r ui.StreamReader) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), waitLimit)
	defer cancel()
	var out []string
	for {
		f, err := r.Next(ctx)
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

func TestStore_tailsFramesFromTheStart(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
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
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("second Close() error = %v", err)
		}
	})
}

func TestStore_framesAreBinarySafeAndOrdered(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		w, err := s.Begin(t.Context(), spec("u1", "c1", func() {}))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		r, err := s.Attach(t.Context(), "c1", "u1")
		if err != nil {
			t.Fatalf("Attach() error = %v", err)
		}
		var want []string
		for i := range 200 {
			frame := "data: {\"i\":" + strconv.Itoa(i) + "}\n\n\x00\xff"
			want = append(want, frame)
			if err := w.Append(t.Context(), []byte(frame)); err != nil {
				t.Fatalf("Append() error = %v", err)
			}
		}
		if err := w.Close(t.Context()); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if diff := cmp.Diff(want, drain(t, r)); diff != "" {
			t.Errorf("frames mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestStore_readerBlocksUntilData(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		w, _ := s.Begin(t.Context(), spec("u1", "c1", func() {}))
		r, _ := s.Attach(t.Context(), "c1", "u1")
		got := make(chan string, 1)
		go func() {
			ctx, cancel := context.WithTimeout(t.Context(), waitLimit)
			defer cancel()
			frame, _ := r.Next(ctx)
			got <- string(frame)
		}()
		select {
		case frame := <-got:
			t.Fatalf("Next() returned %q before any frame", frame)
		case <-time.After(150 * time.Millisecond):
		}
		if err := w.Append(t.Context(), []byte("data")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		if frame := <-got; frame != "data" {
			t.Errorf("Next() = %q, want data", frame)
		}
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestStore_readerHonorsContext(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		w, _ := s.Begin(t.Context(), spec("u1", "c1", func() {}))
		r, _ := s.Attach(t.Context(), "c1", "u1")
		ctx, cancel := context.WithTimeout(t.Context(), 700*time.Millisecond)
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

func TestStore_ownersAreIsolated(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		stopped := make(chan struct{}, 1)
		w, err := s.Begin(t.Context(), spec("alice", "c1", func() { stopped <- struct{}{} }))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if _, err := s.Attach(t.Context(), "c1", "mallory"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Attach(other owner) error = %v, want %v", err, ui.ErrNoStream)
		}
		if err := ui.Stop(t.Context(), s, "c1", "mallory"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Stop(other owner) error = %v, want %v", err, ui.ErrNoStream)
		}
		select {
		case <-stopped:
			t.Fatal("another owner stopped the stream")
		case <-time.After(150 * time.Millisecond):
		}
		other, err := s.Begin(t.Context(), spec("mallory", "c1", func() {}))
		if err != nil {
			t.Errorf("Begin() for another owner with the same chat id error = %v, want an independent stream", err)
		} else if err := other.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
		if err := ui.Stop(t.Context(), s, "c1", "alice"); err != nil {
			t.Errorf("Stop(owner) error = %v", err)
		}
		waitFor(t, stopped, "owner Stop")
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(waitLimit):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func TestStore_ownerAndChatIDsCannotCollide(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		a, err := s.Begin(t.Context(), spec("a:b", "c", func() {}))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if _, err := s.Attach(t.Context(), "b:c", "a"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Attach() with shifted separator error = %v, want %v", err, ui.ErrNoStream)
		}
		if err := a.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestStore_onePerChat(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		if _, err := s.Begin(t.Context(), spec("u1", "c1", func() {})); err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if _, err := s.Begin(t.Context(), spec("u1", "c1", func() {})); !errors.Is(err, ui.ErrBusy) {
			t.Errorf("second Begin() error = %v, want %v", err, ui.ErrBusy)
		}
	})
}

func TestStore_streamsPerOwner(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
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
}

func TestStore_maxBytes(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		stopped := make(chan struct{}, 1)
		sp := spec("u1", "c1", func() { stopped <- struct{}{} })
		sp.MaxBytes = 10
		w, err := s.Begin(t.Context(), sp)
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if err := w.Append(t.Context(), []byte("123456")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		if err := w.Append(t.Context(), []byte("123456")); !errors.Is(err, ui.ErrStreamTooLarge) {
			t.Errorf("Append() over the limit error = %v, want %v", err, ui.ErrStreamTooLarge)
		}
		waitFor(t, stopped, "the size stop")
		r, err := s.Attach(t.Context(), "c1", "u1")
		if err != nil {
			t.Fatalf("Attach() error = %v", err)
		}
		if err := w.Close(t.Context(), []byte("final frames are always kept")); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if diff := cmp.Diff([]string{"123456", "final frames are always kept"}, drain(t, r)); diff != "" {
			t.Errorf("frames mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestStore_ttlExpiresIdleStreams(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore(uiredis.WithTTL(600 * time.Millisecond))
		stopped := make(chan struct{}, 1)
		w, err := s.Begin(t.Context(), spec("u1", "c1", func() { stopped <- struct{}{} }))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		r, _ := s.Attach(t.Context(), "c1", "u1")
		for range 3 {
			time.Sleep(300 * time.Millisecond)
			if err := w.Append(t.Context(), []byte("keepalive")); err != nil {
				t.Fatalf("Append() error = %v", err)
			}
		}
		select {
		case <-stopped:
			t.Fatal("stream expired although frames arrived within the TTL")
		default:
		}
		waitFor(t, stopped, "TTL expiry")
		if diff := cmp.Diff([]string{"keepalive", "keepalive", "keepalive"}, drain(t, r)); diff != "" {
			t.Errorf("frames mismatch (-want +got):\n%s", diff)
		}
		if _, err := s.Attach(t.Context(), "c1", "u1"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Attach() after expiry error = %v, want %v", err, ui.ErrNoStream)
		}
		if err := w.Append(t.Context(), []byte("late")); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Append() after expiry error = %v, want %v", err, ui.ErrNoStream)
		}
		if _, err := s.Begin(t.Context(), spec("u1", "c1", func() {})); err != nil {
			t.Errorf("Begin() after expiry error = %v, want the chat free again", err)
		}
	})
}

func TestStore_stopCrossesInstances(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		owner, other := newStore(), newStore()
		stopped := make(chan struct{}, 4)
		w, err := owner.Begin(t.Context(), spec("u1", "c1", func() { stopped <- struct{}{} }))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if err := w.Append(t.Context(), []byte("one")); err != nil {
			t.Fatalf("Append() error = %v", err)
		}
		r, err := other.Attach(t.Context(), "c1", "u1")
		if err != nil {
			t.Fatalf("Attach() on another instance error = %v", err)
		}
		if err := ui.Stop(t.Context(), other, "c1", "u1"); err != nil {
			t.Fatalf("Stop() on another instance error = %v", err)
		}
		waitFor(t, stopped, "the cross-instance stop")
		if err := w.Close(t.Context(), []byte("aborted")); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if diff := cmp.Diff([]string{"one", "aborted"}, drain(t, r)); diff != "" {
			t.Errorf("frames mismatch (-want +got):\n%s", diff)
		}
		if err := ui.Stop(t.Context(), other, "c1", "u1"); !errors.Is(err, ui.ErrNoStream) {
			t.Errorf("Stop() after the stream ended error = %v, want %v", err, ui.ErrNoStream)
		}
	})
}

func TestStore_stopFlagReachesWriterWithoutPubSub(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		s := newStore()
		stopped := make(chan struct{}, 4)
		w, err := s.Begin(t.Context(), spec("u1", "c1", func() { stopped <- struct{}{} }))
		if err != nil {
			t.Fatalf("Begin() error = %v", err)
		}
		if err := ui.Stop(t.Context(), s, "c1", "u1"); err != nil {
			t.Fatalf("Stop() error = %v", err)
		}
		waitFor(t, stopped, "the stop request")
		if err := w.Append(t.Context(), []byte("x")); err != nil {
			t.Errorf("Append() after Stop error = %v, want the stream usable until it is closed", err)
		}
		waitFor(t, stopped, "the stop flag seen by Append")
		if err := w.Close(t.Context()); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestStore_resumeServesEveryInstance(t *testing.T) {
	eachStore(t, func(t *testing.T, newStore func(...uiredis.Option) *uiredis.Store) {
		t.Helper()
		writerSide, readerSide := newStore(), newStore()
		release := make(chan struct{})
		started := make(chan struct{})
		var wg sync.WaitGroup
		wg.Go(func() {
			req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", nil)
			rec := httptest.NewRecorder()
			ui.Stream(rec, req, func(_ context.Context, uw *ui.Writer) error {
				if err := uw.Data("note", "before"); err != nil {
					return err
				}
				close(started)
				<-release
				return uw.Data("note", "after")
			}, ui.Options{Resume: &ui.Resume{Store: writerSide, ChatID: "chat", Owner: "u1"}})
		})
		waitFor(t, started, "the stream to start")
		mux := http.NewServeMux()
		mux.Handle("GET /chat/{id}/stream", ui.ResumeHandler(readerSide, func(*http.Request) (string, error) { return "u1", nil }))
		other := http.NewServeMux()
		other.Handle("GET /chat/{id}/stream", ui.ResumeHandler(readerSide, func(*http.Request) (string, error) { return "mallory", nil }))
		rec := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			defer close(done)
			mux.ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/chat/chat/stream", nil))
		}()
		stranger := httptest.NewRecorder()
		other.ServeHTTP(stranger, httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/chat/chat/stream", nil))
		if stranger.Code != http.StatusNoContent {
			t.Errorf("another owner got status %d, want 204", stranger.Code)
		}
		close(release)
		wg.Wait()
		waitFor(t, done, "the resumed response")
		body := rec.Body.String()
		for _, want := range []string{`"before"`, `"after"`, "[DONE]"} {
			if !strings.Contains(body, want) {
				t.Errorf("resumed body is missing %s:\n%s", want, body)
			}
		}
		if strings.Index(body, `"before"`) > strings.Index(body, `"after"`) {
			t.Errorf("resumed frames are out of order:\n%s", body)
		}
	})
}
