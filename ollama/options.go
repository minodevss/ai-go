package ollama

import (
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/minodevss/ai-go"
)

// Options are Ollama request settings passed with ai.Providers. Unset fields keep the model or server default.
type Options struct {
	Think         string         `json:"think,omitzero"`
	KeepAlive     string         `json:"keepAlive,omitzero"`
	NumCtx        int            `json:"numCtx,omitzero"`
	MinP          *float64       `json:"minP,omitzero"`
	RepeatPenalty *float64       `json:"repeatPenalty,omitzero"`
	RepeatLastN   *int           `json:"repeatLastN,omitzero"`
	Parameters    map[string]any `json:"parameters,omitzero"`
}

func (Options) ProviderName() string { return providerName }

// EmbeddingOptions are Ollama embedding settings passed with ai.Providers. Unset fields keep the model or server default.
type EmbeddingOptions struct {
	KeepAlive  string         `json:"keepAlive,omitzero"`
	NumCtx     int            `json:"numCtx,omitzero"`
	Truncate   *bool          `json:"truncate,omitzero"`
	Parameters map[string]any `json:"parameters,omitzero"`
}

func (EmbeddingOptions) ProviderName() string { return providerName }

// ResponseData is stored in Response.Provider and in the assistant Message.Provider under "ollama". Durations are the server's nanosecond timings.
type ResponseData struct {
	Model              string        `json:"model,omitzero"`
	CreatedAt          string        `json:"createdAt,omitzero"`
	TotalDuration      time.Duration `json:"totalDuration,omitzero"`
	LoadDuration       time.Duration `json:"loadDuration,omitzero"`
	PromptEvalDuration time.Duration `json:"promptEvalDuration,omitzero"`
	EvalDuration       time.Duration `json:"evalDuration,omitzero"`
}

type partData struct {
	Call json.RawMessage `json:"call,omitzero"`
}

func optionsOf[T any](d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[providerName]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("ollama: decode provider options: %w", err)
	}
	return o, nil
}

func joinHostPort(host, port string) string { return net.JoinHostPort(host, port) }
