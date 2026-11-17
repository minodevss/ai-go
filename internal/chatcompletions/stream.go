package chatcompletions

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
)

type chunk struct {
	ID          string `json:"id"`
	ServiceTier string `json:"service_tier"`
	Choices     []struct {
		Delta        json.RawMessage `json:"delta"`
		FinishReason *string         `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
	Error *struct {
		Code    any    `json:"code"`
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type delta struct {
	Content     json.RawMessage   `json:"content"`
	Refusal     string            `json:"refusal"`
	ToolCalls   []toolCall        `json:"tool_calls"`
	Annotations []json.RawMessage `json:"annotations"`
}

type blockKind int

const (
	blockNone blockKind = iota
	blockReasoning
	blockText
	blockRefusal
)

type pendingCall struct {
	id      string
	name    string
	args    strings.Builder
	started bool
}

type stream struct {
	m           *Model
	yield       func(ai.StreamPart, error) bool
	warnings    []ai.Warning
	open        blockKind
	blockID     string
	blocks      int
	field       string
	thinking    strings.Builder
	signature   string
	verbatim    map[string][]json.RawMessage
	annotations []json.RawMessage
	extras      ResponseExtras
	calls       map[int]*pendingCall
	finish      string
	refusal     bool
	id          string
	tier        string
	usage       json.RawMessage
	stopped     bool
}

func (m *Model) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	req, warnings, err := m.request(c, true)
	if err != nil {
		return nil, err
	}
	resp, err := m.client.Do(ctx, transport.Request{Path: m.path(), Body: req, Header: c.Headers, Stream: true})
	if err != nil {
		return nil, err
	}
	return &ai.StreamResponse{Parts: m.streamParts(resp.Body, warnings), Header: resp.Header}, nil
}

func (m *Model) streamParts(body io.ReadCloser, warnings []ai.Warning) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &stream{m: m, yield: yield, warnings: warnings, calls: map[int]*pendingCall{}, verbatim: map[string][]json.RawMessage{}}
		r := sse.NewReader(body)
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				if s.finish != "" {
					s.end()
					return
				}
				yield(nil, errors.New(m.dialect.Provider+": stream ended before the response finished"))
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("%s: read stream: %w", m.dialect.Provider, err))
				return
			}
			if ev.Data == "[DONE]" {
				s.end()
				return
			}
			if err := s.handle(ev.Data); err != nil {
				yield(nil, err)
				return
			}
			if s.stopped {
				return
			}
		}
	}
}

func (s *stream) emit(p ai.StreamPart) {
	if s.stopped {
		return
	}
	if !s.yield(p, nil) {
		s.stopped = true
	}
}

func (s *stream) handle(data string) error {
	var ch chunk
	if err := json.Unmarshal([]byte(data), &ch); err != nil {
		return fmt.Errorf("%s: decode stream chunk: %w", s.m.dialect.Provider, err)
	}
	if ch.Error != nil {
		return s.eventError(ch.Error.Code, ch.Error.Type, ch.Error.Message)
	}
	s.id = cmp.Or(ch.ID, s.id)
	s.tier = cmp.Or(ch.ServiceTier, s.tier)
	if len(ch.Usage) > 0 && string(ch.Usage) != "null" {
		s.usage = ch.Usage
	}
	s.collectExtras([]byte(data))
	if len(ch.Choices) == 0 {
		return nil
	}
	choice := ch.Choices[0]
	if err := s.delta(choice.Delta); err != nil {
		return err
	}
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		s.finish = *choice.FinishReason
		s.closeBlock()
		s.flushCalls()
	}
	return nil
}

func (s *stream) collectExtras(data []byte) {
	got := s.m.extras(data)
	if len(got.Data) > 0 {
		if s.extras.Data == nil {
			s.extras.Data = map[string]json.RawMessage{}
		}
		maps.Copy(s.extras.Data, got.Data)
	}
	if len(got.Citations) > 0 {
		s.extras.Citations = got.Citations
	}
	for _, w := range got.Warnings {
		if !slices.Contains(s.extras.Warnings, w) {
			s.extras.Warnings = append(s.extras.Warnings, w)
		}
	}
}

func (s *stream) delta(raw json.RawMessage) error {
	var d delta
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("%s: decode stream delta: %w", s.m.dialect.Provider, err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		fields = nil
	}
	for k, v := range verbatimOf(s.m.dialect.Verbatim, raw) {
		s.verbatim[k] = append(s.verbatim[k], v)
	}
	body := parseContent(d.Content)
	text, field := s.m.reasoningText(fields)
	if body.thinking != "" {
		text, field = body.thinking, "content"
		s.signature = cmp.Or(body.signature, s.signature)
	}
	if text != "" {
		s.field = field
		s.write(blockReasoning, text)
	}
	if body.text != "" {
		s.write(blockText, body.text)
	}
	if d.Refusal != "" {
		s.refusal = true
		s.write(blockRefusal, d.Refusal)
	}
	s.annotations = append(s.annotations, d.Annotations...)
	for _, c := range d.ToolCalls {
		s.toolDelta(c)
	}
	return nil
}

func (s *stream) write(kind blockKind, text string) {
	if s.open != kind {
		s.closeBlock()
		s.blocks++
		s.open = kind
		switch kind {
		case blockReasoning:
			s.blockID = fmt.Sprintf("reasoning-%d", s.blocks)
			s.emit(ai.ReasoningStart{ID: s.blockID})
		default:
			s.blockID = fmt.Sprintf("text-%d", s.blocks)
			s.emit(ai.TextStart{ID: s.blockID})
		}
	}
	if kind == blockReasoning {
		s.thinking.WriteString(text)
		s.emit(ai.ReasoningDelta{ID: s.blockID, Text: text})
		return
	}
	s.emit(ai.TextDelta{ID: s.blockID, Text: text})
}

func (s *stream) closeBlock() {
	switch s.open {
	case blockReasoning:
		s.emit(ai.ReasoningEnd{ID: s.blockID, Provider: s.reasoningData()})
	case blockText:
		for _, c := range append(openaiwire.Citations(s.m.dialect.Provider, s.annotations), s.extras.Citations...) {
			s.emit(c)
		}
		s.annotations = nil
		s.emit(ai.TextEnd{ID: s.blockID})
	case blockRefusal:
		s.emit(ai.TextEnd{ID: s.blockID, Provider: pdata.Encode(s.m.dialect.Provider, partData{Refusal: true})})
	case blockNone:
	}
	s.open = blockNone
}

func (s *stream) reasoningData() ai.ProviderData {
	var verbatim map[string]json.RawMessage
	for k, items := range s.verbatim {
		if verbatim == nil {
			verbatim = map[string]json.RawMessage{}
		}
		verbatim[k] = joinVerbatim(items)
	}
	s.verbatim = map[string][]json.RawMessage{}
	p, _ := s.m.reasoningPart(s.thinking.String(), s.field, s.signature, verbatim)
	s.thinking.Reset()
	return p.Provider
}

func (s *stream) toolDelta(c toolCall) {
	call, ok := s.calls[c.Index]
	if !ok {
		call = &pendingCall{}
		s.calls[c.Index] = call
	}
	if c.ID != "" {
		call.id = c.ID
	}
	if c.Function.Name != "" {
		call.name = c.Function.Name
	}
	if !call.started && call.id != "" && call.name != "" {
		s.closeBlock()
		call.started = true
		s.emit(ai.ToolInputStart{ID: call.id, Name: call.name})
		if call.args.Len() > 0 {
			s.emit(ai.ToolInputDelta{ID: call.id, Text: call.args.String()})
		}
	}
	if c.Function.Arguments == "" {
		return
	}
	call.args.WriteString(c.Function.Arguments)
	if call.started {
		s.emit(ai.ToolInputDelta{ID: call.id, Text: c.Function.Arguments})
	}
}

func (s *stream) flushCalls() {
	indexes := make([]int, 0, len(s.calls))
	for i := range s.calls {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	for _, i := range indexes {
		call := s.calls[i]
		if !call.started {
			call.started = true
			s.emit(ai.ToolInputStart{ID: call.id, Name: call.name})
			s.emit(ai.ToolInputDelta{ID: call.id, Text: call.args.String()})
		}
		s.emit(ai.ToolInputEnd{ID: call.id})
		s.emit(ai.ToolCall{ID: call.id, Name: call.name, Input: openaiwire.ToolInput(call.args.String())})
	}
	s.calls = map[int]*pendingCall{}
}

func (s *stream) end() {
	if len(s.verbatim) > 0 && s.open == blockNone {
		s.blocks++
		s.blockID = fmt.Sprintf("reasoning-%d", s.blocks)
		s.emit(ai.ReasoningStart{ID: s.blockID})
		s.emit(ai.ReasoningEnd{ID: s.blockID, Provider: s.reasoningData()})
	}
	s.closeBlock()
	s.flushCalls()
	finish, raw := finishReason(s.finish, s.refusal)
	s.emit(ai.StepEnd{Usage: s.m.usage(s.usage), FinishReason: finish, FinishRaw: raw, Warnings: slices.Concat(s.warnings, s.extras.Warnings), Provider: s.m.responseProvider(s.id, s.tier, s.extras.Data)})
}

func (s *stream) eventError(code any, typ, message string) error {
	switch c := code.(type) {
	case float64:
		err := transport.ErrorFromEvent(s.m.dialect.Provider, strconv.Itoa(int(c)), message)
		err.StatusCode = int(c)
		return err
	case string:
		return transport.ErrorFromEvent(s.m.dialect.Provider, cmp.Or(c, typ), message)
	default:
		return transport.ErrorFromEvent(s.m.dialect.Provider, typ, message)
	}
}
