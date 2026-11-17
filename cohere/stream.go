package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
)

const planID = "plan"

type streamEvent struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Index int             `json:"index"`
	Delta json.RawMessage `json:"delta"`
}

type streamDelta struct {
	Message struct {
		Content   json.RawMessage `json:"content"`
		ToolPlan  string          `json:"tool_plan"`
		ToolCalls json.RawMessage `json:"tool_calls"`
		Citations json.RawMessage `json:"citations"`
	} `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Error        string          `json:"error"`
	Usage        json.RawMessage `json:"usage"`
}

type contentDelta struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

func (d streamDelta) content() contentDelta {
	var c contentDelta
	if len(d.Message.Content) > 0 && d.Message.Content[0] == '{' {
		if err := json.Unmarshal(d.Message.Content, &c); err != nil {
			return contentDelta{}
		}
	}
	return c
}

type streamBlock struct {
	thinking bool
}

type streamCall struct {
	id   string
	name string
	args strings.Builder
}

type chatStream struct {
	yield    func(ai.StreamPart, error) bool
	warnings []ai.Warning
	docIDs   []string
	data     ResponseData
	blocks   map[int]*streamBlock
	calls    map[int]*streamCall
	cites    map[int][]ai.Citation
	planOpen bool
	stopped  bool
	finished bool
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
	return &ai.StreamResponse{Parts: chatParts(resp.Body, p), Header: resp.Header}, nil
}

func chatParts(body io.ReadCloser, p *prepared) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &chatStream{yield: yield, warnings: p.warnings, docIDs: p.docIDs, blocks: map[int]*streamBlock{}, calls: map[int]*streamCall{}, cites: map[int][]ai.Citation{}}
		r := sse.NewReader(body)
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				yield(nil, errors.New("cohere: stream ended before the message finished"))
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("cohere: read stream: %w", err))
				return
			}
			if err := s.handle(ev.Data); err != nil {
				yield(nil, err)
				return
			}
			if s.finished || s.stopped {
				return
			}
		}
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

func (s *chatStream) handle(data string) error {
	var ev streamEvent
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("cohere: decode stream event: %w", err)
	}
	var d streamDelta
	if len(ev.Delta) > 0 {
		if err := json.Unmarshal(ev.Delta, &d); err != nil {
			return fmt.Errorf("cohere: decode stream delta: %w", err)
		}
	}
	switch ev.Type {
	case "message-start":
		s.data.ID = ev.ID
	case "content-start":
		s.closePlan()
		s.contentStart(ev.Index, d)
	case "content-delta":
		s.onContentDelta(ev.Index, d)
	case "content-end":
		s.contentEnd(ev.Index)
	case "tool-plan-delta":
		s.planDelta(d.Message.ToolPlan)
	case "tool-call-start":
		return s.callStart(ev.Index, d)
	case "tool-call-delta":
		return s.callDelta(ev.Index, d)
	case "tool-call-end":
		s.callEnd(ev.Index)
	case "citation-start":
		return s.citation(d)
	case "message-end":
		return s.finish(ev, d)
	default:
	}
	return nil
}

func (s *chatStream) contentStart(idx int, d streamDelta) {
	st := &streamBlock{thinking: d.content().Type == "thinking"}
	s.blocks[idx] = st
	id := blockID(idx)
	if st.thinking {
		s.emit(ai.ReasoningStart{ID: id})
		s.reasoningDelta(id, d.content().Thinking)
		return
	}
	s.emit(ai.TextStart{ID: id})
	s.textDelta(id, d.content().Text)
}

func (s *chatStream) onContentDelta(idx int, d streamDelta) {
	st, ok := s.blocks[idx]
	if !ok {
		return
	}
	if st.thinking {
		s.reasoningDelta(blockID(idx), d.content().Thinking)
		return
	}
	s.textDelta(blockID(idx), d.content().Text)
}

func (s *chatStream) textDelta(id, text string) {
	if text != "" {
		s.emit(ai.TextDelta{ID: id, Text: text})
	}
}

func (s *chatStream) reasoningDelta(id, text string) {
	if text != "" {
		s.emit(ai.ReasoningDelta{ID: id, Text: text})
	}
}

func (s *chatStream) contentEnd(idx int) {
	st, ok := s.blocks[idx]
	if !ok {
		return
	}
	delete(s.blocks, idx)
	id := blockID(idx)
	if st.thinking {
		s.emit(ai.ReasoningEnd{ID: id, Provider: pdata.Encode(providerName, partData{Thinking: true})})
		return
	}
	s.emit(ai.TextEnd{ID: id})
	for _, c := range s.cites[idx] {
		s.emit(c)
	}
	delete(s.cites, idx)
}

func (s *chatStream) planDelta(text string) {
	if !s.planOpen {
		s.planOpen = true
		s.emit(ai.ReasoningStart{ID: planID})
	}
	s.reasoningDelta(planID, text)
}

func (s *chatStream) closePlan() {
	if !s.planOpen {
		return
	}
	s.planOpen = false
	s.emit(ai.ReasoningEnd{ID: planID, Provider: pdata.Encode(providerName, partData{ToolPlan: true})})
}

type streamToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (s *chatStream) callStart(idx int, d streamDelta) error {
	s.closePlan()
	var tc streamToolCall
	if err := json.Unmarshal(d.Message.ToolCalls, &tc); err != nil {
		return fmt.Errorf("cohere: decode tool call: %w", err)
	}
	call := &streamCall{id: tc.ID, name: tc.Function.Name}
	call.args.WriteString(tc.Function.Arguments)
	s.calls[idx] = call
	s.emit(ai.ToolInputStart{ID: call.id, Name: call.name})
	if tc.Function.Arguments != "" {
		s.emit(ai.ToolInputDelta{ID: call.id, Text: tc.Function.Arguments})
	}
	return nil
}

func (s *chatStream) callDelta(idx int, d streamDelta) error {
	call, ok := s.calls[idx]
	if !ok {
		return nil
	}
	var tc streamToolCall
	if err := json.Unmarshal(d.Message.ToolCalls, &tc); err != nil {
		return fmt.Errorf("cohere: decode tool call delta: %w", err)
	}
	if tc.Function.Arguments == "" {
		return nil
	}
	call.args.WriteString(tc.Function.Arguments)
	s.emit(ai.ToolInputDelta{ID: call.id, Text: tc.Function.Arguments})
	return nil
}

func (s *chatStream) callEnd(idx int) {
	call, ok := s.calls[idx]
	if !ok {
		return
	}
	delete(s.calls, idx)
	s.emit(ai.ToolInputEnd{ID: call.id})
	s.emit(ai.ToolCall{ID: call.id, Name: call.name, Input: openaiwire.ToolInput(call.args.String())})
}

func (s *chatStream) citation(d streamDelta) error {
	c, err := parseCitation(d.Message.Citations)
	if err != nil {
		return err
	}
	if c.Type != "" && c.Type != "TEXT_CONTENT" {
		s.data.Citations = append(s.data.Citations, d.Message.Citations)
		return nil
	}
	cites := citationsOf(c, s.docIDs)
	if _, open := s.blocks[c.ContentIndex]; !open {
		for _, cite := range cites {
			s.emit(cite)
		}
		return nil
	}
	s.cites[c.ContentIndex] = append(s.cites[c.ContentIndex], cites...)
	return nil
}

func (s *chatStream) finish(ev streamEvent, d streamDelta) error {
	if d.Error != "" {
		return &ai.APIError{Provider: providerName, Code: "stream_error", Message: d.Error}
	}
	s.closePlan()
	if ev.ID != "" {
		s.data.ID = ev.ID
	}
	s.finished = true
	s.emit(ai.StepEnd{Usage: usageOf(d.Usage), FinishReason: finishOf(d.FinishReason), FinishRaw: d.FinishReason, Warnings: s.warnings, Provider: responseProvider(s.data)})
	return nil
}
