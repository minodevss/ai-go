package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime"
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
	"github.com/minodevss/ai-go/internal/emit"
	"github.com/minodevss/ai-go/internal/transport"
	"github.com/minodevss/ai-go/realtime"
)

const (
	realtimeReadLimit = 16 << 20
	minSecretTTL      = 10 * time.Second
	maxSecretTTL      = 2 * time.Hour
	liveAudioChunk    = 9600
)

// RealtimeModel talks to the Realtime API over a WebSocket and implements realtime.SecretMinter.
type RealtimeModel struct {
	client *transport.Client
	id     string
}

type realtimeConn struct {
	ws         *websocket.Conn
	state      wireState
	closing    atomic.Bool
	commitSent atomic.Bool
	once       sync.Once
	closed     error
	events     iter.Seq2[realtime.Event, error]
}

type secretRequest struct {
	ExpiresAfter secretExpiry `json:"expires_after"`
	Session      sessionWire  `json:"session"`
}

type secretExpiry struct {
	Anchor  string `json:"anchor"`
	Seconds int    `json:"seconds"`
}

type secretResponse struct {
	Value     string `json:"value"`
	ExpiresAt int64  `json:"expires_at"`
}

func (p *Provider) RealtimeModel(id string) *RealtimeModel {
	return &RealtimeModel{client: p.client, id: id}
}

func (m *RealtimeModel) Provider() string { return providerName }

func (m *RealtimeModel) ModelID() string { return m.id }

func (m *RealtimeModel) Connect(ctx context.Context, c *realtime.Config) (realtime.Conn, error) {
	if c.Resume != "" {
		return nil, fmt.Errorf("openai: realtime sessions cannot be resumed: %w", realtime.ErrUnsupported)
	}
	opts, err := decodeOptions[RealtimeOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	session, err := buildSession(c, "")
	if err != nil {
		return nil, err
	}
	conn, err := dialRealtime(ctx, m.client, "/realtime?model="+url.QueryEscape(m.id), realtimeHeaders(opts))
	if err != nil {
		return nil, err
	}
	if err := conn.send(ctx, "session.update", map[string]any{"session": session}); err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	conn.events = conn.readEvents(ctx)
	return conn, nil
}

func dialRealtime(ctx context.Context, client *transport.Client, path string, extra http.Header) (*realtimeConn, error) {
	if err := client.Wait(ctx); err != nil {
		return nil, err
	}
	target, err := url.Parse(client.URL(path))
	if err != nil {
		return nil, fmt.Errorf("openai: build realtime url: %w", err)
	}
	switch target.Scheme {
	case "https":
		target.Scheme = "wss"
	case "http":
		target.Scheme = "ws"
	}
	header := client.Headers()
	for k, v := range extra {
		header[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
	}
	ws, resp, err := websocket.Dial(ctx, target.String(), &websocket.DialOptions{HTTPClient: client.HTTPClient(), HTTPHeader: header})
	if err != nil {
		if resp != nil {
			return nil, client.HandshakeError(resp)
		}
		return nil, fmt.Errorf("openai: connect realtime: %w", err)
	}
	ws.SetReadLimit(realtimeReadLimit)
	return &realtimeConn{ws: ws}, nil
}

func (c *realtimeConn) send(ctx context.Context, typ string, fields map[string]any) error {
	data, err := clientEvent(typ, fields)
	if err != nil {
		return fmt.Errorf("openai: encode realtime event: %w", err)
	}
	if err := c.ws.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("openai: send realtime event: %w", err)
	}
	return nil
}

func (c *realtimeConn) SendAudio(ctx context.Context, pcm []byte) error {
	return c.send(ctx, "input_audio_buffer.append", map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)})
}

func (c *realtimeConn) SendText(ctx context.Context, text string) error {
	item := map[string]any{"type": "message", "role": "user", "content": []map[string]any{{"type": "input_text", "text": text}}}
	if err := c.send(ctx, "conversation.item.create", map[string]any{"item": item}); err != nil {
		return err
	}
	return c.send(ctx, "response.create", map[string]any{})
}

func (c *realtimeConn) Interrupt(ctx context.Context) error {
	return c.send(ctx, "response.cancel", map[string]any{})
}

func (c *realtimeConn) SendToolResult(ctx context.Context, callID string, out ai.ToolOutput, _ bool) error {
	item := map[string]any{"type": "function_call_output", "call_id": callID, "output": outputText(out)}
	if err := c.send(ctx, "conversation.item.create", map[string]any{"item": item}); err != nil {
		return err
	}
	if !c.state.answered(callID) {
		return nil
	}
	return c.send(ctx, "response.create", map[string]any{})
}

func outputText(out ai.ToolOutput) string {
	switch {
	case out.Text != "":
		return out.Text
	case len(out.JSON) > 0:
		return string(out.JSON)
	default:
		var b strings.Builder
		for _, p := range out.Parts {
			if t, ok := p.(ai.Text); ok {
				b.WriteString(t.Text)
			}
		}
		return b.String()
	}
}

func (c *realtimeConn) Events() iter.Seq2[realtime.Event, error] { return c.events }

func (c *realtimeConn) readEvents(ctx context.Context) iter.Seq2[realtime.Event, error] {
	return func(yield func(realtime.Event, error) bool) {
		for {
			_, data, err := c.ws.Read(ctx)
			if err != nil {
				if c.closing.Load() || ctx.Err() != nil || websocket.CloseStatus(err) == websocket.StatusNormalClosure {
					return
				}
				yield(nil, fmt.Errorf("openai: read realtime event: %w", err))
				return
			}
			events, err := decodeServerEvent(data, &c.state)
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

func (c *realtimeConn) Close() error {
	c.once.Do(func() {
		c.closing.Store(true)
		err := c.ws.Close(websocket.StatusNormalClosure, "")
		if err != nil && websocket.CloseStatus(err) != websocket.StatusNormalClosure && !errors.Is(err, net.ErrClosed) {
			c.closed = err
		}
	})
	return c.closed
}

// MintClientSecret creates an ephemeral key that lives 10 seconds to 2 hours. The client can override its session settings, so the secret is not Locked.
func (m *RealtimeModel) MintClientSecret(ctx context.Context, c *realtime.Config, o realtime.MintOptions) (*realtime.ClientSecret, error) {
	if o.TTL < minSecretTTL || o.TTL > maxSecretTTL {
		return nil, fmt.Errorf("openai: client secret TTL %s is outside %s to %s", o.TTL, minSecretTTL, maxSecretTTL)
	}
	session, err := buildSession(c, m.id)
	if err != nil {
		return nil, err
	}
	seconds := int((o.TTL + time.Second - 1) / time.Second)
	var body secretResponse
	req := transport.Request{Path: "/realtime/client_secrets", Body: secretRequest{ExpiresAfter: secretExpiry{Anchor: "created_at", Seconds: seconds}, Session: session}}
	if _, err := m.client.JSON(ctx, req, &body); err != nil {
		return nil, err
	}
	return &realtime.ClientSecret{
		Value:     body.Value,
		ExpiresAt: time.Unix(body.ExpiresAt, 0),
		Warnings:  []ai.Warning{{Message: "openai: the client can override the model, instructions, tools and voice of this secret with session.update; OpenAI does not lock them"}},
	}, nil
}

func (m *TranscriptionModel) streamLive(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	opts, err := decodeOptions[TranscriptionOptions](c.Provider)
	if err != nil {
		return nil, err
	}
	if err := checkLiveMediaType(c.MediaType); err != nil {
		return nil, err
	}
	conn, err := dialRealtime(ctx, m.client, "/realtime?intent=transcription", nil)
	if err != nil {
		return nil, err
	}
	input := map[string]any{
		"format":        audioFormatWire{Type: "audio/pcm", Rate: defaultRealtimeRate},
		"transcription": transcriptionWire{Model: m.id, Language: c.Language, Prompt: opts.Prompt},
	}
	if liveModelDetectsTurns(m.id) {
		input["turn_detection"] = map[string]any{"type": "server_vad"}
	} else {
		input["turn_detection"] = nil
	}
	session := map[string]any{"type": "transcription", "audio": map[string]any{"input": input}}
	if err := conn.send(ctx, "session.update", map[string]any{"session": session}); err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	return &ai.TranscribeStreamResponse{Segments: conn.liveSegments(ctx, c.Audio)}, nil
}

func checkLiveMediaType(mediaType string) error {
	_, params, err := mime.ParseMediaType(mediaType)
	if err != nil {
		return fmt.Errorf("openai: parse audio media type %q: %w", mediaType, err)
	}
	rate, ok := params["rate"]
	if !ok {
		return nil
	}
	hz, err := strconv.Atoi(rate)
	if err != nil {
		return fmt.Errorf("openai: audio media type %q has an invalid rate", mediaType)
	}
	return checkRealtimeRate(hz)
}

func liveModelDetectsTurns(model string) bool {
	return !strings.Contains(model, "live-transcribe") && !strings.Contains(model, "realtime-whisper")
}

func (c *realtimeConn) liveSegments(ctx context.Context, audio io.Reader) iter.Seq2[ai.Segment, error] {
	return func(yield func(ai.Segment, error) bool) {
		push := emit.New(yield)
		ctx, cancel := context.WithCancel(ctx)
		sent := make(chan error, 1)
		var wg sync.WaitGroup
		defer func() {
			cancel()
			c.closing.Store(true)
			if closer, ok := audio.(io.Closer); ok {
				if err := closer.Close(); err != nil {
					push(ai.Segment{}, fmt.Errorf("openai: close audio: %w", err))
				}
				wg.Wait()
			}
			if err := c.ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				push(ai.Segment{}, fmt.Errorf("openai: close realtime transcription: %w", err))
			}
		}()
		wg.Go(func() {
			if err := c.pumpAudio(ctx, audio); err != nil {
				sent <- err
				cancel()
			}
		})
		pending := map[string]struct{}{}
		resolved := false
		for {
			_, data, err := c.ws.Read(ctx)
			if err != nil {
				select {
				case perr := <-sent:
					push(ai.Segment{}, perr)
				default:
					if websocket.CloseStatus(err) != websocket.StatusNormalClosure {
						push(ai.Segment{}, fmt.Errorf("openai: read realtime transcription: %w", err))
					}
				}
				return
			}
			ev, err := decodeLiveEvent(data)
			if err != nil {
				push(ai.Segment{}, err)
				return
			}
			if ev.seg != nil && !push(*ev.seg, nil) {
				return
			}
			switch ev.kind {
			case liveNone:
			case liveCommitted:
				pending[ev.itemID] = struct{}{}
			case liveCompleted:
				delete(pending, ev.itemID)
			case liveEmptyCommit:
				if !c.commitSent.Load() {
					push(ai.Segment{}, errors.New("openai: realtime transcription commit without audio"))
					return
				}
				resolved = true
			}
			if resolved && len(pending) == 0 {
				return
			}
		}
	}
}

func (c *realtimeConn) pumpAudio(ctx context.Context, audio io.Reader) error {
	buf := make([]byte, liveAudioChunk)
	for {
		n, err := audio.Read(buf)
		if n > 0 {
			if serr := c.SendAudio(ctx, buf[:n]); serr != nil {
				return serr
			}
		}
		if errors.Is(err, io.EOF) {
			c.commitSent.Store(true)
			for range 2 {
				if err := c.send(ctx, "input_audio_buffer.commit", map[string]any{}); err != nil {
					return err
				}
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("openai: read audio: %w", err)
		}
	}
}

type liveKind int

const (
	liveNone liveKind = iota
	liveCommitted
	liveCompleted
	liveEmptyCommit
)

const errCommitEmpty = "commit_empty"

type liveEvent struct {
	seg    *ai.Segment
	kind   liveKind
	itemID string
}

func decodeLiveEvent(data []byte) (liveEvent, error) {
	var e serverEvent
	if err := json.Unmarshal(data, &e); err != nil {
		return liveEvent{}, fmt.Errorf("openai: decode realtime event: %w", err)
	}
	switch e.Type {
	case "input_audio_buffer.committed":
		return liveEvent{kind: liveCommitted, itemID: e.ItemID}, nil
	case "conversation.item.input_audio_transcription.delta":
		return liveEvent{seg: &ai.Segment{Text: e.Delta, Partial: true}}, nil
	case "conversation.item.input_audio_transcription.completed":
		return liveEvent{seg: &ai.Segment{Text: e.Transcript}, kind: liveCompleted, itemID: e.ItemID}, nil
	case "conversation.item.input_audio_transcription.failed":
		return liveEvent{}, errors.New("openai: realtime transcription failed")
	case "error":
		if e.Error != nil && strings.Contains(e.Error.Code, errCommitEmpty) {
			return liveEvent{kind: liveEmptyCommit}, nil
		}
		return liveEvent{}, realtimeError(e)
	default:
		return liveEvent{}, nil
	}
}

var (
	_ realtime.Model        = (*RealtimeModel)(nil)
	_ realtime.SecretMinter = (*RealtimeModel)(nil)
)
