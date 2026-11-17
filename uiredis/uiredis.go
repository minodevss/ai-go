// Package uiredis is a Redis-backed ui.StreamStore: streams survive the process, any instance can resume or stop them, and a crashed instance's streams expire.
package uiredis

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/minodevss/ai-go/ui"
)

const (
	defaultTTL    = 15 * time.Minute
	defaultGrace  = time.Minute
	defaultPrefix = "ui"
	readBlock     = 500 * time.Millisecond
	readBatch     = 64
	cleanupBudget = 5 * time.Second
)

const (
	fieldFrame = "f"
	fieldEnd   = "e"
	fieldStart = "s"
)

const nowMillis = `local t = redis.call('TIME'); local now = t[1] * 1000 + math.floor(t[2] / 1000)`

var beginScript = redis.NewScript(nowMillis + `
if redis.call('EXISTS', KEYS[1]) == 1 then return 1 end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
local maxConcurrent = tonumber(ARGV[3])
if maxConcurrent > 0 and redis.call('ZCARD', KEYS[3]) >= maxConcurrent then return 2 end
redis.call('HSET', KEYS[1], 'gen', ARGV[1], 'size', 0, 'stop', 0, 'max', ARGV[4])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('XADD', KEYS[2], '*', '` + fieldStart + `', '1')
redis.call('PEXPIRE', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], now + ARGV[2], ARGV[5])
redis.call('PEXPIRE', KEYS[3], ARGV[2])
return 0`)

var appendScript = redis.NewScript(nowMillis + `
if redis.call('HGET', KEYS[1], 'gen') ~= ARGV[1] then return -1 end
local maxBytes = tonumber(redis.call('HGET', KEYS[1], 'max'))
local size = tonumber(redis.call('HGET', KEYS[1], 'size'))
if maxBytes > 0 and size + #ARGV[2] > maxBytes then return -2 end
redis.call('HINCRBY', KEYS[1], 'size', #ARGV[2])
redis.call('XADD', KEYS[2], '*', '` + fieldFrame + `', ARGV[2])
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
redis.call('ZADD', KEYS[3], now + ARGV[3], ARGV[4])
redis.call('PEXPIRE', KEYS[3], ARGV[3])
if redis.call('HGET', KEYS[1], 'stop') == '1' then return 1 end
return 0`)

var closeScript = redis.NewScript(`
if redis.call('HGET', KEYS[1], 'gen') ~= ARGV[1] then return 0 end
for i = 4, #ARGV do
  redis.call('XADD', KEYS[2], '*', '` + fieldFrame + `', ARGV[i])
end
redis.call('XADD', KEYS[2], '*', '` + fieldEnd + `', '1')
redis.call('PEXPIRE', KEYS[2], ARGV[2])
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[3], ARGV[3])
return 1`)

var stopScript = redis.NewScript(`
local gen = redis.call('HGET', KEYS[1], 'gen')
if not gen then return false end
redis.call('HSET', KEYS[1], 'stop', 1)
return gen`)

type options struct {
	ttl    time.Duration
	grace  time.Duration
	prefix string
}

type Option func(*options)

// WithTTL ends a stream that gets no frame for d; the default is 15 minutes. Keys outlive it by the grace period, so only a crashed instance leaves them to Redis expiry.
func WithTTL(d time.Duration) Option {
	return func(o *options) {
		if d > 0 {
			o.ttl = d
		}
	}
}

// WithGrace keeps the frames of an ended stream for d so slow readers can finish; the default is one minute.
func WithGrace(d time.Duration) Option {
	return func(o *options) {
		if d > 0 {
			o.grace = d
		}
	}
}

// WithPrefix namespaces every key and channel; the default is "ui".
func WithPrefix(prefix string) Option {
	return func(o *options) {
		if prefix != "" {
			o.prefix = prefix
		}
	}
}

// Store implements ui.StreamStore on Redis 7 or later, including cluster setups, because all keys of one owner share a hash tag.
type Store struct {
	client redis.UniversalClient
	opts   options
}

var _ ui.StreamStore = (*Store)(nil)

func New(client redis.UniversalClient, opts ...Option) *Store {
	o := options{ttl: defaultTTL, grace: defaultGrace, prefix: defaultPrefix}
	for _, opt := range opts {
		opt(&o)
	}
	return &Store{client: client, opts: o}
}

type keys struct {
	meta, active, stop string
	base               string
	chat               string
}

func (s *Store) keys(chatID, owner string) keys {
	enc := base64.RawURLEncoding
	tag := s.opts.prefix + ":{" + enc.EncodeToString([]byte(owner)) + "}"
	chat := enc.EncodeToString([]byte(chatID))
	return keys{
		meta:   tag + ":" + chat + ":meta",
		active: tag + ":active",
		stop:   tag + ":" + chat + ":stop",
		base:   tag + ":" + chat + ":",
		chat:   chat,
	}
}

func (k keys) stream(gen string) string { return k.base + gen }

func (s *Store) Begin(ctx context.Context, spec ui.StreamSpec) (ui.StreamWriter, error) {
	k := s.keys(spec.ChatID, spec.Owner)
	gen, err := newGeneration()
	if err != nil {
		return nil, err
	}
	ttl := strconv.FormatInt((s.opts.ttl + s.opts.grace).Milliseconds(), 10)
	code, err := beginScript.Run(ctx, s.client, []string{k.meta, k.stream(gen), k.active},
		gen, ttl, spec.MaxConcurrent, spec.MaxBytes, k.chat).Int()
	if err != nil {
		return nil, fmt.Errorf("uiredis: begin stream: %w", err)
	}
	switch code {
	case 1:
		return nil, ui.ErrBusy
	case 2:
		return nil, ui.ErrTooManyStreams
	default:
	}
	w := &writer{store: s, spec: spec, keys: k, gen: gen, ttl: ttl, base: context.WithoutCancel(ctx)}
	if err := w.listen(ctx); err != nil {
		w.end(w.base)
		return nil, err
	}
	w.timer = time.AfterFunc(s.opts.ttl, w.expire)
	return w, nil
}

func (s *Store) Attach(ctx context.Context, chatID, owner string) (ui.StreamReader, error) {
	k := s.keys(chatID, owner)
	gen, err := s.client.HGet(ctx, k.meta, "gen").Result()
	if errors.Is(err, redis.Nil) {
		return nil, ui.ErrNoStream
	}
	if err != nil {
		return nil, fmt.Errorf("uiredis: attach: %w", err)
	}
	return &reader{client: s.client, key: k.stream(gen), last: "0-0"}, nil
}

func (s *Store) Stop(ctx context.Context, chatID, owner string) error {
	k := s.keys(chatID, owner)
	gen, err := stopScript.Run(ctx, s.client, []string{k.meta}).Text()
	if errors.Is(err, redis.Nil) {
		return ui.ErrNoStream
	}
	if err != nil {
		return fmt.Errorf("uiredis: stop: %w", err)
	}
	if err := s.client.Publish(ctx, k.stop, gen).Err(); err != nil {
		return fmt.Errorf("uiredis: stop: %w", err)
	}
	return nil
}

func newGeneration() (string, error) {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("uiredis: generate stream id: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

type writer struct {
	store *Store
	spec  ui.StreamSpec
	keys  keys
	gen   string
	ttl   string

	base context.Context

	mu       sync.Mutex
	done     bool
	timer    *time.Timer
	unlisten context.CancelFunc
}

func (w *writer) listen(ctx context.Context) error {
	subCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	sub := w.store.client.Subscribe(subCtx, w.keys.stop)
	if _, err := sub.Receive(subCtx); err != nil {
		cancel()
		_ = sub.Close()
		return fmt.Errorf("uiredis: subscribe to stop requests: %w", err)
	}
	go func() {
		for msg := range sub.Channel() {
			if msg.Payload == w.gen {
				w.spec.Stop()
			}
		}
	}()
	w.unlisten = func() {
		cancel()
		_ = sub.Close()
	}
	return nil
}

func (w *writer) Append(ctx context.Context, frame []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.done {
		return ui.ErrNoStream
	}
	code, err := appendScript.Run(ctx, w.store.client, w.scriptKeys(), w.gen, frame, w.ttl, w.keys.chat).Int()
	if err != nil {
		return fmt.Errorf("uiredis: append frame: %w", err)
	}
	switch code {
	case -1:
		w.finishLocked()
		return ui.ErrNoStream
	case -2:
		w.spec.Stop()
		return ui.ErrStreamTooLarge
	case 1:
		w.spec.Stop()
	default:
	}
	w.timer.Reset(w.store.opts.ttl)
	return nil
}

func (w *writer) Close(ctx context.Context, final ...[]byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.done {
		return nil
	}
	err := w.closeScript(ctx, final)
	w.finishLocked()
	return err
}

func (w *writer) expire() {
	w.spec.Stop()
	w.end(w.base)
}

func (w *writer) end(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, cleanupBudget)
	defer cancel()
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.done {
		return
	}
	_ = w.closeScript(ctx, nil)
	w.finishLocked()
}

func (w *writer) closeScript(ctx context.Context, final [][]byte) error {
	args := make([]any, 0, 3+len(final))
	args = append(args, w.gen, w.store.opts.grace.Milliseconds(), w.keys.chat)
	for _, f := range final {
		args = append(args, f)
	}
	if err := closeScript.Run(ctx, w.store.client, w.scriptKeys(), args...).Err(); err != nil {
		return fmt.Errorf("uiredis: close stream: %w", err)
	}
	return nil
}

func (w *writer) finishLocked() {
	w.done = true
	if w.timer != nil {
		w.timer.Stop()
	}
	if w.unlisten != nil {
		w.unlisten()
	}
}

func (w *writer) scriptKeys() []string {
	return []string{w.keys.meta, w.keys.stream(w.gen), w.keys.active}
}

type reader struct {
	client redis.UniversalClient
	key    string
	last   string
	buf    []redis.XMessage
	ended  bool
}

func (r *reader) Next(ctx context.Context) ([]byte, error) {
	for {
		for len(r.buf) > 0 {
			msg := r.buf[0]
			r.buf = r.buf[1:]
			if _, end := msg.Values[fieldEnd]; end {
				r.ended = true
				return nil, io.EOF
			}
			if frame, ok := msg.Values[fieldFrame].(string); ok {
				return []byte(frame), nil
			}
		}
		if r.ended {
			return nil, io.EOF
		}
		if err := r.fill(ctx); err != nil {
			return nil, err
		}
	}
}

func (r *reader) fill(ctx context.Context) error {
	res, err := r.client.XRead(ctx, &redis.XReadArgs{Streams: []string{r.key, r.last}, Count: readBatch, Block: readBlock}).Result()
	switch {
	case errors.Is(err, redis.Nil):
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		exists, err := r.client.Exists(ctx, r.key).Result()
		if err != nil {
			return r.readError(ctx, err)
		}
		if exists == 0 {
			r.ended = true
		}
		return nil
	case err != nil:
		return r.readError(ctx, err)
	}
	for _, stream := range res {
		r.buf = append(r.buf, stream.Messages...)
		if n := len(stream.Messages); n > 0 {
			r.last = stream.Messages[n-1].ID
		}
	}
	return nil
}

func (r *reader) readError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	return fmt.Errorf("uiredis: read stream: %w", err)
}

func (r *reader) Close() error { return nil }
