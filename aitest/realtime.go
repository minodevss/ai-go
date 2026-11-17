package aitest

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/realtime"
)

// RealtimeInput is something the application sent to the model: Kind is "audio", "text", "interrupt" or "tool-result".
type RealtimeInput struct {
	Kind    string
	Audio   []byte
	Text    string
	CallID  string
	Output  ai.ToolOutput
	IsError bool
}

// RealtimeModel is a realtime.Model that the test drives: Emit plays an event as if the provider sent it, and Sent shows what the session sent back. It also implements realtime.SecretMinter.
type RealtimeModel struct {
	mu       sync.Mutex
	conn     *realtimeConn
	cfg      realtime.Config
	secrets  []realtime.MintOptions
	secretTO time.Duration
	failures failures
}

func NewRealtimeModel(opts ...RealtimeOption) *RealtimeModel {
	m := &RealtimeModel{secretTO: time.Minute}
	for _, opt := range opts {
		opt.applyRealtime(m)
	}
	return m
}

func (m *RealtimeModel) Provider() string { return "aitest" }

func (m *RealtimeModel) ModelID() string { return "scripted-realtime" }

func (m *RealtimeModel) Config() realtime.Config {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cfg
}

func (m *RealtimeModel) MintRequests() []realtime.MintOptions {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]realtime.MintOptions(nil), m.secrets...)
}

func (m *RealtimeModel) Connect(ctx context.Context, c *realtime.Config) (realtime.Conn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.failures.next(); err != nil {
		return nil, err
	}
	m.cfg = *c
	m.conn = &realtimeConn{events: make(chan realtimeItem), done: make(chan struct{}), ended: ctx.Done()}
	return m.conn, nil
}

func (m *RealtimeModel) MintClientSecret(_ context.Context, c *realtime.Config, o realtime.MintOptions) (*realtime.ClientSecret, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg = *c
	m.secrets = append(m.secrets, o)
	return &realtime.ClientSecret{Value: fmt.Sprintf("ek_%d", len(m.secrets)), ExpiresAt: time.Now().Add(o.TTL), Locked: true}, nil
}

// Emit delivers ev to the session and blocks until the session has taken it.
func (m *RealtimeModel) Emit(ctx context.Context, ev realtime.Event) error {
	c := m.current()
	if c == nil {
		return errNotConnected
	}
	return c.push(ctx, realtimeItem{event: ev})
}

// FailStream ends the session's event stream with err, as a lost connection would.
func (m *RealtimeModel) FailStream(ctx context.Context, err error) error {
	c := m.current()
	if c == nil {
		return errNotConnected
	}
	return c.push(ctx, realtimeItem{err: err})
}

// Sent returns everything the session sent to the model so far.
func (m *RealtimeModel) Sent() []RealtimeInput {
	c := m.current()
	if c == nil {
		return nil
	}
	return c.sentCopy()
}

var errNotConnected = errors.New("aitest: realtime model has no session yet")

func (m *RealtimeModel) current() *realtimeConn {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.conn
}

type realtimeItem struct {
	event realtime.Event
	err   error
}

type realtimeConn struct {
	events chan realtimeItem
	done   chan struct{}
	ended  <-chan struct{}
	once   sync.Once
	mu     sync.Mutex
	sent   []RealtimeInput
}

func (c *realtimeConn) push(ctx context.Context, it realtimeItem) error {
	select {
	case c.events <- it:
		return nil
	case <-c.done:
		return errors.New("aitest: realtime connection closed")
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *realtimeConn) record(in RealtimeInput) error {
	select {
	case <-c.done:
		return errors.New("aitest: realtime connection closed")
	default:
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sent = append(c.sent, in)
	return nil
}

func (c *realtimeConn) sentCopy() []RealtimeInput {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]RealtimeInput(nil), c.sent...)
}

func (c *realtimeConn) SendAudio(_ context.Context, pcm []byte) error {
	return c.record(RealtimeInput{Kind: "audio", Audio: append([]byte(nil), pcm...)})
}

func (c *realtimeConn) SendText(_ context.Context, text string) error {
	return c.record(RealtimeInput{Kind: "text", Text: text})
}

func (c *realtimeConn) Interrupt(context.Context) error {
	return c.record(RealtimeInput{Kind: "interrupt"})
}

func (c *realtimeConn) SendToolResult(_ context.Context, callID string, out ai.ToolOutput, isError bool) error {
	return c.record(RealtimeInput{Kind: "tool-result", CallID: callID, Output: out, IsError: isError})
}

func (c *realtimeConn) Events() iter.Seq2[realtime.Event, error] {
	return func(yield func(realtime.Event, error) bool) {
		for {
			select {
			case it := <-c.events:
				if it.err != nil {
					yield(nil, it.err)
					return
				}
				if !yield(it.event, nil) {
					return
				}
			case <-c.done:
				return
			case <-c.ended:
				return
			}
		}
	}
}

func (c *realtimeConn) Close() error {
	c.once.Do(func() { close(c.done) })
	return nil
}

var (
	_ realtime.Model        = (*RealtimeModel)(nil)
	_ realtime.SecretMinter = (*RealtimeModel)(nil)
)
