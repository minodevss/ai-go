package ollama

import (
	"encoding/json"
	"errors"
	"slices"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

type chatChunk struct {
	Model                 string          `json:"model"`
	CreatedAt             string          `json:"created_at"`
	Message               responseMessage `json:"message"`
	Done                  bool            `json:"done"`
	DoneReason            string          `json:"done_reason"`
	TotalDuration         int64           `json:"total_duration"`
	LoadDuration          int64           `json:"load_duration"`
	PromptEvalCount       int             `json:"prompt_eval_count"`
	PromptEvalCachedCount int             `json:"prompt_eval_cached_count"`
	PromptEvalDuration    int64           `json:"prompt_eval_duration"`
	EvalCount             int             `json:"eval_count"`
	EvalDuration          int64           `json:"eval_duration"`
	Error                 string          `json:"error"`
}

type responseMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Thinking  string            `json:"thinking"`
	ToolCalls []json.RawMessage `json:"tool_calls"`
}

type usageRaw struct {
	PromptEvalCount       int   `json:"prompt_eval_count,omitzero"`
	PromptEvalCachedCount int   `json:"prompt_eval_cached_count,omitzero"`
	EvalCount             int   `json:"eval_count,omitzero"`
	TotalDuration         int64 `json:"total_duration,omitzero"`
	LoadDuration          int64 `json:"load_duration,omitzero"`
	PromptEvalDuration    int64 `json:"prompt_eval_duration,omitzero"`
	EvalDuration          int64 `json:"eval_duration,omitzero"`
}

func (c chatChunk) metadata() ai.ProviderData {
	return pdata.Encode(providerName, ResponseData{
		Model:              c.Model,
		CreatedAt:          c.CreatedAt,
		TotalDuration:      durationOf(c.TotalDuration),
		LoadDuration:       durationOf(c.LoadDuration),
		PromptEvalDuration: durationOf(c.PromptEvalDuration),
		EvalDuration:       durationOf(c.EvalDuration),
	})
}

func usageOf(c chatChunk) ai.Usage {
	return ai.Usage{
		Input:     c.PromptEvalCount,
		Output:    c.EvalCount,
		CacheRead: min(c.PromptEvalCachedCount, c.PromptEvalCount),
		Raw: pdata.JSON(providerName, usageRaw{
			PromptEvalCount:       c.PromptEvalCount,
			PromptEvalCachedCount: c.PromptEvalCachedCount,
			EvalCount:             c.EvalCount,
			TotalDuration:         c.TotalDuration,
			LoadDuration:          c.LoadDuration,
			PromptEvalDuration:    c.PromptEvalDuration,
			EvalDuration:          c.EvalDuration,
		}),
	}
}

func finishOf(reason string, hasCalls bool) (ai.FinishReason, string) {
	if hasCalls {
		return ai.FinishToolCalls, reason
	}
	switch reason {
	case "", "stop":
		return ai.FinishStop, reason
	case "length":
		return ai.FinishLength, reason
	default:
		return ai.FinishOther, reason
	}
}

func hasToolCall(parts ai.Parts) bool {
	return slices.ContainsFunc(parts, func(p ai.Part) bool {
		_, ok := p.(ai.ToolCall)
		return ok
	})
}

func convertMessage(m responseMessage) ai.Parts {
	var parts ai.Parts
	if m.Thinking != "" {
		parts = append(parts, ai.Reasoning{Text: m.Thinking, Provider: pdata.Encode(providerName, partData{})})
	}
	if m.Content != "" {
		parts = append(parts, ai.Text{Text: m.Content})
	}
	for i, raw := range m.ToolCalls {
		parts = append(parts, toolCallOf(raw, i))
	}
	return parts
}

func toolCallOf(raw json.RawMessage, index int) ai.ToolCall {
	var c wireCall
	if err := json.Unmarshal(raw, &c); err != nil {
		return ai.ToolCall{ID: callID("", nil, index), Provider: pdata.Encode(providerName, partData{Call: raw})}
	}
	args := c.Function.Arguments
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	return ai.ToolCall{
		ID:       callID(c.Function.Name, args, index),
		Name:     c.Function.Name,
		Input:    args,
		Provider: pdata.Encode(providerName, partData{Call: raw}),
	}
}

func streamError(message string) *ai.APIError {
	err := transport.ErrorFromEvent(providerName, "stream_error", message)
	return err
}

var errStreamEnded = errors.New("ollama: stream ended before the response finished")
