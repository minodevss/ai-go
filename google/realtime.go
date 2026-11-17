package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
	"github.com/minodevss/ai-go/realtime"
)

const (
	liveReadLimit     = 16 << 20
	liveSocketPath    = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	secretMessageTail = 30 * time.Minute
	maxSecretTTL      = 20*time.Hour - secretMessageTail
)

// RealtimeModel talks to the Live API on AI Studio over a WebSocket. Vertex AI sessions are not supported.
type RealtimeModel struct {
	backend *backend
	id      string
}

type liveConn struct {
	ws        *websocket.Conn
	state     liveState
	inputRate int
	closing   atomic.Bool
	once      sync.Once
	closed    error
	events    iter.Seq2[realtime.Event, error]
}

type liveClientMessage struct {
	RealtimeInput *liveRealtimeInput `json:"realtimeInput,omitzero"`
	ToolResponse  *liveToolResponse  `json:"toolResponse,omitzero"`
}

type liveRealtimeInput struct {
	Audio *liveBlob `json:"audio,omitzero"`
	Text  string    `json:"text,omitzero"`
}

type liveBlob struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type liveToolResponse struct {
	FunctionResponses []liveFunctionResponse `json:"functionResponses"`
}

type liveFunctionResponse struct {
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type secretRequest struct {
	Uses                 int               `json:"uses"`
	ExpireTime           string            `json:"expireTime"`
	NewSessionExpireTime string            `json:"newSessionExpireTime"`
	LiveConnect          secretConstraints `json:"liveConnectConstraints"`
}

type secretConstraints struct {
	Model  string            `json:"model"`
	Config liveConnectConfig `json:"config"`
}

type secretResponse struct {
	Name string `json:"name"`
}

func (p *Provider) RealtimeModel(id string) *RealtimeModel {
	return &RealtimeModel{backend: p.backend, id: id}
}

func (m *RealtimeModel) Provider() string { return m.backend.name }

func (m *RealtimeModel) ModelID() string { return m.id }

func (m *RealtimeModel) Connect(ctx context.Context, c *realtime.Config) (realtime.Conn, error) {
	if err := m.requireStudio(); err != nil {
		return nil, err
	}
	setup, err := liveSettings(c, m.id)
	if err != nil {
		return nil, err
	}
	conn, err := m.dial(ctx)
	if err != nil {
		return nil, err
	}
	conn.inputRate = c.InputAudio.SampleRate
	if conn.inputRate == 0 {
		conn.inputRate = defaultInputRate
	}
	if err := conn.start(ctx, setup); err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	conn.events = conn.readEvents(ctx)
	return conn, nil
}

func (m *RealtimeModel) requireStudio() error {
	if m.backend.vertex {
		return fmt.Errorf("%w: google realtime sessions are not supported on Vertex AI", realtime.ErrUnsupported)
	}
	return nil
}

func (m *RealtimeModel) dial(ctx context.Context) (*liveConn, error) {
	client := m.backend.client
	if err := client.Wait(ctx); err != nil {
		return nil, err
	}
	base, err := url.Parse(client.URL(""))
	if err != nil {
		return nil, fmt.Errorf("google: build live url: %w", err)
	}
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	case "http":
		base.Scheme = "ws"
	}
	base.Path = liveSocketPath
	ws, resp, err := websocket.Dial(ctx, base.String(), &websocket.DialOptions{HTTPClient: client.HTTPClient(), HTTPHeader: client.Headers()})
	if err != nil {
		if resp != nil {
			return nil, liveHandshakeError(client, resp)
		}
		return nil, fmt.Errorf("google: connect live session: %w", err)
	}
	ws.SetReadLimit(liveReadLimit)
	return &liveConn{ws: ws}, nil
}

func liveHandshakeError(client *transport.Client, resp *http.Response) error {
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if err != nil {
		body = nil
	}
	return fail(client.APIError(resp, body))
}

func (c *liveConn) start(ctx context.Context, setup liveSetup) error {
	if err := c.write(ctx, liveSetupMessage{Setup: setup}); err != nil {
		return err
	}
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		return fmt.Errorf("google: read live setup: %w", err)
	}
	var m liveServerMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return fmt.Errorf("google: decode live setup: %w", err)
	}
	if m.SetupComplete == nil {
		return errSetupFailed
	}
	return nil
}

func (c *liveConn) write(ctx context.Context, msg any) error {
	data, err := transport.EncodeJSON(msg)
	if err != nil {
		return fmt.Errorf("google: encode live message: %w", err)
	}
	if err := c.ws.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("google: send live message: %w", err)
	}
	return nil
}

func (c *liveConn) SendAudio(ctx context.Context, pcm []byte) error {
	blob := liveBlob{Data: base64.StdEncoding.EncodeToString(pcm), MimeType: "audio/pcm;rate=" + strconv.Itoa(c.inputRate)}
	return c.write(ctx, liveClientMessage{RealtimeInput: &liveRealtimeInput{Audio: &blob}})
}

func (c *liveConn) SendText(ctx context.Context, text string) error {
	return c.write(ctx, liveClientMessage{RealtimeInput: &liveRealtimeInput{Text: text}})
}

func (c *liveConn) Interrupt(context.Context) error {
	return fmt.Errorf("%w: the Live API interrupts a response when new user input arrives", realtime.ErrUnsupported)
}

func (c *liveConn) SendToolResult(ctx context.Context, callID string, out ai.ToolOutput, isError bool) error {
	resp, err := liveToolResult(out, isError)
	if err != nil {
		return err
	}
	fr := liveFunctionResponse{ID: callID, Name: c.state.nameOf(callID), Response: resp}
	return c.write(ctx, liveClientMessage{ToolResponse: &liveToolResponse{FunctionResponses: []liveFunctionResponse{fr}}})
}

func liveToolResult(out ai.ToolOutput, isError bool) (json.RawMessage, error) {
	key := "output"
	if isError {
		key = "error"
	}
	var value any = out.Text
	switch {
	case len(out.JSON) > 0:
		var obj map[string]json.RawMessage
		if !isError && json.Unmarshal(out.JSON, &obj) == nil && obj != nil {
			return out.JSON, nil
		}
		value = out.JSON
	case len(out.Parts) > 0:
		value = partsText(out.Parts)
	}
	raw, err := transport.EncodeJSON(map[string]any{key: value})
	if err != nil {
		return nil, fmt.Errorf("google: encode tool result: %w", err)
	}
	return raw, nil
}

func partsText(parts ai.Parts) string {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(ai.Text); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func (c *liveConn) Events() iter.Seq2[realtime.Event, error] { return c.events }

func (c *liveConn) readEvents(ctx context.Context) iter.Seq2[realtime.Event, error] {
	return func(yield func(realtime.Event, error) bool) {
		for {
			_, data, err := c.ws.Read(ctx)
			if err != nil {
				if c.closing.Load() || ctx.Err() != nil || websocket.CloseStatus(err) == websocket.StatusNormalClosure {
					return
				}
				yield(nil, fmt.Errorf("google: read live message: %w", err))
				return
			}
			events, err := decodeLiveMessage(data, &c.state)
			if err != nil {
				yield(nil, err)
				return
			}
			for _, ev := range events {
				if !yield(ev, nil) {
					return
				}
			}
		}
	}
}

func (c *liveConn) Close() error {
	c.once.Do(func() {
		c.closing.Store(true)
		err := c.ws.Close(websocket.StatusNormalClosure, "")
		if err != nil && websocket.CloseStatus(err) != websocket.StatusNormalClosure && !errors.Is(err, net.ErrClosed) {
			c.closed = err
		}
	})
	return c.closed
}

// MintClientSecret creates an ephemeral token that starts one session within o.TTL, locked to this model and the settings of c.
func (m *RealtimeModel) MintClientSecret(ctx context.Context, c *realtime.Config, o realtime.MintOptions) (*realtime.ClientSecret, error) {
	if err := m.requireStudio(); err != nil {
		return nil, err
	}
	if o.TTL > maxSecretTTL {
		return nil, fmt.Errorf("google: client secret TTL %s exceeds the %s limit", o.TTL, maxSecretTTL)
	}
	cfg, err := liveConstraints(c)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	startBy := now.Add(o.TTL)
	req := secretRequest{
		Uses:                 1,
		ExpireTime:           startBy.Add(secretMessageTail).UTC().Format(time.RFC3339),
		NewSessionExpireTime: startBy.UTC().Format(time.RFC3339),
		LiveConnect:          secretConstraints{Model: "models/" + strings.TrimPrefix(m.id, "models/"), Config: cfg},
	}
	var body secretResponse
	if _, err := m.backend.client.JSON(ctx, transport.Request{Path: "/auth_tokens", Body: req}, &body); err != nil {
		return nil, fail(err)
	}
	return &realtime.ClientSecret{Value: body.Name, ExpiresAt: startBy, Locked: true}, nil
}

var (
	_ realtime.Model        = (*RealtimeModel)(nil)
	_ realtime.SecretMinter = (*RealtimeModel)(nil)
)
