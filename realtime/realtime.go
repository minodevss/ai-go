// Package realtime runs low-latency voice and text sessions with models that stream audio both ways, and executes their tool calls on the server.
package realtime

import (
	"context"
	"errors"
	"iter"
	"time"

	"github.com/minodevss/ai-go"
)

var (
	ErrClosed                  = errors.New("realtime: session closed")
	ErrInputTooLarge           = errors.New("realtime: input exceeds the configured limit")
	ErrUnsupported             = errors.New("realtime: not supported by this model")
	ErrApprovalUnenforceable   = errors.New("realtime: a client secret cannot enforce tool approval")
	ErrEventsConsumed          = errors.New("realtime: events already consumed")
	ErrUnknownCall             = errors.New("realtime: unknown tool call")
	ErrInterrupted             = errors.New("realtime: tool call interrupted")
	errTooManyPendingApprovals = errors.New("realtime: too many pending approvals")
)

const (
	defaultEventBuffer         = 64
	defaultMaxInputBytes       = 1 << 20
	defaultApprovalTimeout     = 5 * time.Minute
	defaultMaxPendingApprovals = 16
	defaultMaxInFlightTools    = 16
	defaultSecretTTL           = time.Minute
	maxSecretTTL               = time.Hour
)

// Model is what a provider implements. Provider conns must return from Events when Close is called, and report a lost connection as the error of Events.
type Model interface {
	Provider() string
	ModelID() string
	Connect(ctx context.Context, c *Config) (Conn, error)
}

// Conn is one provider connection. The Session owns it: it consumes Events once and handles tool calls, so Conn implementations only translate wire messages.
type Conn interface {
	SendAudio(ctx context.Context, pcm []byte) error
	SendText(ctx context.Context, text string) error
	Interrupt(ctx context.Context) error
	SendToolResult(ctx context.Context, callID string, out ai.ToolOutput, isError bool) error
	Events() iter.Seq2[Event, error]
	Close() error
}

// SecretMinter is the capability of models whose provider issues short-lived credentials for clients that connect directly.
type SecretMinter interface {
	MintClientSecret(ctx context.Context, c *Config, o MintOptions) (*ClientSecret, error)
}

type AudioFormat struct {
	Encoding   string
	SampleRate int
}

type TurnMode string

const (
	TurnServerVAD   TurnMode = "server-vad"
	TurnSemanticVAD TurnMode = "semantic-vad"
	TurnManual      TurnMode = "manual"
)

// TurnDetection's zero value keeps the provider default.
type TurnDetection struct {
	Mode            TurnMode
	Threshold       *float64
	SilenceDuration time.Duration
	Eagerness       string
}

// Config's zero limits select the defaults: EventBuffer 64, MaxInputBytes 1 MiB, ApprovalTimeout 5 minutes, MaxPendingApprovals 16, MaxInFlightTools 16. Tools run on the server, honor Tool.Approval, and are declared to the model; tools without Execute are only declared, and the application answers them with Session.SendToolResult.
type Config struct {
	Instructions  string
	Tools         []ai.Tool
	Voice         string
	InputAudio    AudioFormat
	OutputAudio   AudioFormat
	TurnDetection TurnDetection
	Resume        string
	Provider      ai.ProviderData
	Telemetry     ai.Telemetry

	EventBuffer         int
	MaxInputBytes       int
	ApprovalTimeout     time.Duration
	MaxPendingApprovals int
	MaxInFlightTools    int
}

func (c Config) clone() Config {
	c.Tools = append([]ai.Tool(nil), c.Tools...)
	c.Provider = cloneProviderData(c.Provider)
	return c
}

func cloneProviderData(d ai.ProviderData) ai.ProviderData {
	if len(d) == 0 {
		return nil
	}
	out := make(ai.ProviderData, len(d))
	for k, v := range d {
		out[k] = append([]byte(nil), v...)
	}
	return out
}
