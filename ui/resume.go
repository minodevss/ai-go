package ui

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	defaultMaxDuration   = 10 * time.Minute
	defaultMaxConcurrent = 4
	defaultMaxBytes      = 8 << 20
	defaultStoreTTL      = 15 * time.Minute
)

// Resume makes generation outlive the request: frames go to Store and every reader, the original request included, tails them from there.
type Resume struct {
	Store         StreamStore
	ChatID        string
	Owner         string
	MaxDuration   time.Duration
	MaxConcurrent int
	MaxBytes      int64
}

// StreamStore keeps the frames of active streams per owner and chat. Implementations must never return another owner's stream.
type StreamStore interface {
	Begin(ctx context.Context, spec StreamSpec) (StreamWriter, error)
	Attach(ctx context.Context, chatID, owner string) (StreamReader, error)
	Stop(ctx context.Context, chatID, owner string) error
}

// StreamSpec.Stop must be called when the store wants the generation to end, on Stop, on expiry or on overflow.
type StreamSpec struct {
	ChatID        string
	Owner         string
	MaxConcurrent int
	MaxBytes      int64
	Stop          context.CancelFunc
}

type StreamWriter interface {
	// Append returns ErrStreamTooLarge when the stream is over its limit.
	Append(ctx context.Context, frame []byte) error
	// Close appends the final frames even over the limit, then ends the stream.
	Close(ctx context.Context, final ...[]byte) error
}

type StreamReader interface {
	// Next blocks for the next frame and returns io.EOF when the stream ended.
	Next(ctx context.Context) ([]byte, error)
	Close() error
}

func (r *Resume) validate() error {
	if r.Store == nil || r.Owner == "" || r.ChatID == "" {
		return errors.New("ui: resume needs a store, an owner and a chat id")
	}
	return nil
}

func (r *Resume) limits() (time.Duration, int, int64) {
	d, c, b := r.MaxDuration, r.MaxConcurrent, r.MaxBytes
	if d <= 0 {
		d = defaultMaxDuration
	}
	if c <= 0 {
		c = defaultMaxConcurrent
	}
	if b <= 0 {
		b = defaultMaxBytes
	}
	return d, c, b
}

type storeSink struct {
	ctx context.Context
	w   StreamWriter
}

func (s *storeSink) write(frame []byte) error { return s.w.Append(s.ctx, frame) }

func (s *storeSink) end(final [][]byte) error { return s.w.Close(s.ctx, final...) }

func streamResumable(w http.ResponseWriter, r *http.Request, fn func(context.Context, *Writer) error, o options) {
	res := o.Resume
	maxDuration, maxConcurrent, maxBytes := res.limits()
	gctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), maxDuration)
	handle, err := res.Store.Begin(r.Context(), StreamSpec{ChatID: res.ChatID, Owner: res.Owner, MaxConcurrent: maxConcurrent, MaxBytes: maxBytes, Stop: cancel})
	if err != nil {
		cancel()
		switch {
		case errors.Is(err, ErrBusy):
			http.Error(w, "a response is already being generated for this chat", http.StatusConflict)
		case errors.Is(err, ErrTooManyStreams):
			http.Error(w, "too many active responses", http.StatusTooManyRequests)
		default:
			failEarly(w, o, err)
		}
		return
	}
	reader, err := res.Store.Attach(r.Context(), res.ChatID, res.Owner)
	if err != nil {
		cancel()
		if cerr := handle.Close(context.WithoutCancel(r.Context())); cerr != nil {
			o.OnError(cerr)
		}
		failEarly(w, o, err)
		return
	}
	sctx := context.WithoutCancel(gctx)
	go func() {
		defer cancel()
		produce(gctx, cancel, &storeSink{ctx: sctx, w: handle}, fn, o)
	}()
	defer reader.Close()
	setStreamHeaders(w)
	w.WriteHeader(http.StatusOK)
	copyFrames(r.Context(), w, reader)
}

func copyFrames(ctx context.Context, w http.ResponseWriter, reader StreamReader) {
	out := &httpSink{w: w, rc: http.NewResponseController(w)}
	if out.flush() != nil {
		return
	}
	for {
		frame, err := reader.Next(ctx)
		if err != nil || out.write(frame) != nil {
			return
		}
	}
}

// ResumeHandler serves GET requests for the active stream of the chat named by the request's "id" path value, as useChat's reconnect expects: 204 when there is none or it belongs to someone else.
func ResumeHandler(store StreamStore, ownerOf func(*http.Request) (string, error)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		owner, err := ownerOf(r)
		if err != nil {
			http.Error(w, http.StatusText(http.StatusUnauthorized), http.StatusUnauthorized)
			return
		}
		reader, err := store.Attach(r.Context(), r.PathValue("id"), owner)
		if err != nil {
			if errors.Is(err, ErrNoStream) {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			return
		}
		defer reader.Close()
		setStreamHeaders(w)
		w.WriteHeader(http.StatusOK)
		copyFrames(r.Context(), w, reader)
	})
}

// Stop ends the active generation of a chat through the store. It returns ErrNoStream when there is none or it belongs to another owner.
func Stop(ctx context.Context, store StreamStore, chatID, owner string) error {
	return store.Stop(ctx, chatID, owner)
}

type streamKey struct{ owner, chat string }

// MemoryStore is an in-process StreamStore; a stream that gets no frame for the TTL is stopped and dropped.
type MemoryStore struct {
	mu      sync.Mutex
	ttl     time.Duration
	streams map[streamKey]*memStream
}

type MemoryOption func(*MemoryStore)

func WithTTL(d time.Duration) MemoryOption {
	return func(s *MemoryStore) {
		if d > 0 {
			s.ttl = d
		}
	}
}

func NewMemoryStore(opts ...MemoryOption) *MemoryStore {
	s := &MemoryStore{ttl: defaultStoreTTL, streams: map[streamKey]*memStream{}}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

type memStream struct {
	store *MemoryStore
	key   streamKey
	spec  StreamSpec

	mu     sync.Mutex
	frames [][]byte
	size   int64
	done   bool
	wake   chan struct{}
	timer  *time.Timer
}

func (s *MemoryStore) Begin(_ context.Context, spec StreamSpec) (StreamWriter, error) {
	key := streamKey{owner: spec.Owner, chat: spec.ChatID}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, busy := s.streams[key]; busy {
		return nil, ErrBusy
	}
	active := 0
	for k := range s.streams {
		if k.owner == spec.Owner {
			active++
		}
	}
	if spec.MaxConcurrent > 0 && active >= spec.MaxConcurrent {
		return nil, ErrTooManyStreams
	}
	ms := &memStream{store: s, key: key, spec: spec, wake: make(chan struct{})}
	ms.timer = time.AfterFunc(s.ttl, ms.expire)
	s.streams[key] = ms
	return ms, nil
}

func (s *MemoryStore) Attach(_ context.Context, chatID, owner string) (StreamReader, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ms, ok := s.streams[streamKey{owner: owner, chat: chatID}]
	if !ok {
		return nil, ErrNoStream
	}
	return &memReader{stream: ms}, nil
}

func (s *MemoryStore) Stop(_ context.Context, chatID, owner string) error {
	s.mu.Lock()
	ms, ok := s.streams[streamKey{owner: owner, chat: chatID}]
	s.mu.Unlock()
	if !ok {
		return ErrNoStream
	}
	ms.spec.Stop()
	return nil
}

func (m *memStream) expire() {
	m.spec.Stop()
	m.finish()
}

func (m *memStream) finish() {
	m.mu.Lock()
	if m.done {
		m.mu.Unlock()
		return
	}
	m.done = true
	m.timer.Stop()
	m.broadcast()
	m.mu.Unlock()
	m.store.mu.Lock()
	delete(m.store.streams, m.key)
	m.store.mu.Unlock()
}

func (m *memStream) broadcast() {
	close(m.wake)
	m.wake = make(chan struct{})
}

func (m *memStream) Append(_ context.Context, frame []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.done {
		return ErrNoStream
	}
	if m.spec.MaxBytes > 0 && m.size+int64(len(frame)) > m.spec.MaxBytes {
		m.spec.Stop()
		return ErrStreamTooLarge
	}
	m.add(frame)
	return nil
}

func (m *memStream) add(frame []byte) {
	m.frames = append(m.frames, append([]byte(nil), frame...))
	m.size += int64(len(frame))
	m.timer.Reset(m.store.ttl)
	m.broadcast()
}

func (m *memStream) Close(_ context.Context, final ...[]byte) error {
	m.mu.Lock()
	if !m.done {
		for _, f := range final {
			m.add(f)
		}
	}
	m.mu.Unlock()
	m.finish()
	return nil
}

type memReader struct {
	stream *memStream
	next   int
}

func (r *memReader) Next(ctx context.Context) ([]byte, error) {
	for {
		r.stream.mu.Lock()
		if r.next < len(r.stream.frames) {
			frame := r.stream.frames[r.next]
			r.next++
			r.stream.mu.Unlock()
			return frame, nil
		}
		if r.stream.done {
			r.stream.mu.Unlock()
			return nil, io.EOF
		}
		wake := r.stream.wake
		r.stream.mu.Unlock()
		select {
		case <-wake:
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		}
	}
}

func (r *memReader) Close() error { return nil }
