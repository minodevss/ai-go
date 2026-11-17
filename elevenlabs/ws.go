package elevenlabs

import (
	"context"
	"fmt"
	"net/url"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go/internal/transport"
)

const wsReadLimit = 16 << 20

func dialWS(ctx context.Context, client *transport.Client, path string) (*websocket.Conn, error) {
	if err := client.Wait(ctx); err != nil {
		return nil, err
	}
	target, err := url.Parse(client.URL(path))
	if err != nil {
		return nil, fmt.Errorf("elevenlabs: build websocket url: %w", err)
	}
	switch target.Scheme {
	case "https":
		target.Scheme = "wss"
	case "http":
		target.Scheme = "ws"
	}
	ws, resp, err := websocket.Dial(ctx, target.String(), &websocket.DialOptions{HTTPClient: client.HTTPClient(), HTTPHeader: client.Headers()})
	if err != nil {
		if resp != nil {
			return nil, client.HandshakeError(resp)
		}
		return nil, fmt.Errorf("elevenlabs: connect websocket: %w", err)
	}
	ws.SetReadLimit(wsReadLimit)
	return ws, nil
}
