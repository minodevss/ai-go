package ollama

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxLineBytes    = 32 << 20
	reasoningBlock  = "reasoning"
	textBlock       = "text"
	initialLineSize = 64 << 10
)

type chatStream struct {
	yield     func(ai.StreamPart, error) bool
	warnings  []ai.Warning
	reasoning bool
	text      bool
	calls     []json.RawMessage
	stopped   bool
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	p, err := m.prepare(c, true)
	if err != nil {
		return nil, err
	}
	resp, err := m.client.Do(ctx, transport.Request{Path: chatPath, Body: p.req, Header: p.header, Stream: true})
	if err != nil {
		return nil, err
	}
	return &ai.StreamResponse{Parts: chatParts(resp.Body, p.warnings), Header: resp.Header}, nil
}

func chatParts(body io.ReadCloser, warnings []ai.Warning) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &chatStream{yield: yield, warnings: warnings}
		scan := bufio.NewScanner(body)
		scan.Buffer(make([]byte, 0, initialLineSize), maxLineBytes)
		for scan.Scan() {
			line := scan.Bytes()
			if len(line) == 0 {
				continue
			}
			var chunk chatChunk
			if err := json.Unmarshal(line, &chunk); err != nil {
				yield(nil, fmt.Errorf("ollama: decode stream chunk: %w", err))
				return
			}
			if chunk.Error != "" {
				yield(nil, streamError(chunk.Error))
				return
			}
			s.chunk(chunk)
			if s.stopped {
				return
			}
			if chunk.Done {
				s.finish(chunk)
				return
			}
		}
		if err := scan.Err(); err != nil {
			yield(nil, fmt.Errorf("ollama: read stream: %w", err))
			return
		}
		yield(nil, errStreamEnded)
	}
}

func (s *chatStream) emit(p ai.StreamPart) {
	if s.stopped {
		return
	}
	if !s.yield(p, nil) {
		s.stopped = true
	}
}

func (s *chatStream) chunk(c chatChunk) {
	if c.Message.Thinking != "" {
		if !s.reasoning {
			s.reasoning = true
			s.emit(ai.ReasoningStart{ID: reasoningBlock})
		}
		s.emit(ai.ReasoningDelta{ID: reasoningBlock, Text: c.Message.Thinking})
	}
	if c.Message.Content != "" {
		s.closeReasoning()
		if !s.text {
			s.text = true
			s.emit(ai.TextStart{ID: textBlock})
		}
		s.emit(ai.TextDelta{ID: textBlock, Text: c.Message.Content})
	}
	s.calls = append(s.calls, c.Message.ToolCalls...)
}

func (s *chatStream) closeReasoning() {
	if s.reasoning {
		s.reasoning = false
		s.emit(ai.ReasoningEnd{ID: reasoningBlock, Provider: pdata.Encode(providerName, partData{})})
	}
}

func (s *chatStream) finish(c chatChunk) {
	s.closeReasoning()
	if s.text {
		s.text = false
		s.emit(ai.TextEnd{ID: textBlock})
	}
	for i, raw := range s.calls {
		call := toolCallOf(raw, i)
		s.emit(ai.ToolInputStart{ID: call.ID, Name: call.Name})
		s.emit(ai.ToolInputDelta{ID: call.ID, Text: string(call.Input)})
		s.emit(ai.ToolInputEnd{ID: call.ID})
		s.emit(call)
	}
	finish, raw := finishOf(c.DoneReason, len(s.calls) > 0)
	s.emit(ai.StepEnd{Usage: usageOf(c), FinishReason: finish, FinishRaw: raw, Warnings: s.warnings, Provider: c.metadata()})
}
