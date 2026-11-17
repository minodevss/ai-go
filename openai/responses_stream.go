package openai

import (
	"cmp"
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

type streamEvent struct {
	Type         string          `json:"type"`
	OutputIndex  int             `json:"output_index"`
	ItemID       string          `json:"item_id"`
	ContentIndex int             `json:"content_index"`
	SummaryIndex int             `json:"summary_index"`
	Delta        string          `json:"delta"`
	Item         json.RawMessage `json:"item"`
	Response     json.RawMessage `json:"response"`
	Code         string          `json:"code"`
	Message      string          `json:"message"`
	Error        *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type itemState struct {
	id         string
	callID     string
	started    bool
	acc        strings.Builder
	summaryIdx int
	text       map[int]*textState
}

type textState struct {
	started bool
	acc     strings.Builder
}

type responseStream struct {
	yield    func(ai.StreamPart, error) bool
	warnings []ai.Warning
	items    map[int]*itemState
	done     map[int]bool
	flags    converted
	stopped  bool
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	req, warnings, err := m.request(c, true)
	if err != nil {
		return nil, err
	}
	resp, err := m.client.Do(ctx, transport.Request{Path: "/responses", Body: req, Header: c.Headers, Stream: true})
	if err != nil {
		return nil, err
	}
	return &ai.StreamResponse{Parts: responseParts(resp.Body, warnings), Header: resp.Header}, nil
}

func responseParts(body io.ReadCloser, warnings []ai.Warning) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &responseStream{yield: yield, warnings: warnings, items: map[int]*itemState{}, done: map[int]bool{}}
		r := sse.NewReader(body)
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				yield(nil, errors.New("openai: stream ended before the response finished"))
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("openai: read stream: %w", err))
				return
			}
			if ev.Data == "[DONE]" {
				continue
			}
			finished, err := s.handle(ev.Data)
			if err != nil {
				yield(nil, err)
				return
			}
			if finished || s.stopped {
				return
			}
		}
	}
}

func (s *responseStream) emit(p ai.StreamPart) {
	if s.stopped {
		return
	}
	if !s.yield(p, nil) {
		s.stopped = true
	}
}

func (s *responseStream) handle(data string) (bool, error) {
	var ev streamEvent
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return false, fmt.Errorf("openai: decode stream event: %w", err)
	}
	switch ev.Type {
	case "response.output_item.added":
		return false, s.added(ev)
	case "response.content_part.added":
		st := s.item(ev.OutputIndex, ev.ItemID)
		s.startText(st, ev.ContentIndex)
	case "response.output_text.delta", "response.refusal.delta":
		st := s.item(ev.OutputIndex, ev.ItemID)
		ts := s.startText(st, ev.ContentIndex)
		ts.acc.WriteString(ev.Delta)
		s.emit(ai.TextDelta{ID: textID(st.id, ev.ContentIndex), Text: ev.Delta})
	case "response.function_call_arguments.delta":
		st := s.item(ev.OutputIndex, ev.ItemID)
		st.acc.WriteString(ev.Delta)
		if st.started {
			s.emit(ai.ToolInputDelta{ID: st.callID, Text: ev.Delta})
		}
	case "response.reasoning_summary_text.delta":
		st := s.item(ev.OutputIndex, ev.ItemID)
		s.startReasoning(st)
		for ; st.summaryIdx < ev.SummaryIndex; st.summaryIdx++ {
			s.reasoningDelta(st, "\n\n")
		}
		s.reasoningDelta(st, ev.Delta)
	case "response.reasoning_text.delta":
		st := s.item(ev.OutputIndex, ev.ItemID)
		s.startReasoning(st)
		s.reasoningDelta(st, ev.Delta)
	case "response.output_item.done":
		return false, s.finishItem(ev.OutputIndex, ev.Item)
	case "response.completed", "response.incomplete":
		return true, s.complete(ev.Response)
	case "response.failed":
		return true, s.failed(ev.Response)
	case "error":
		return true, eventError(ev)
	default:
	}
	return false, nil
}

func eventError(ev streamEvent) error {
	code, message := ev.Code, ev.Message
	if ev.Error != nil {
		code, message = ev.Error.Code, ev.Error.Message
	}
	return transport.ErrorFromEvent(providerName, code, message)
}

func (s *responseStream) item(idx int, id string) *itemState {
	st, ok := s.items[idx]
	if !ok {
		st = &itemState{id: id, text: map[int]*textState{}}
		s.items[idx] = st
	}
	if st.id == "" {
		st.id = id
	}
	return st
}

func (s *responseStream) added(ev streamEvent) error {
	var it outputItem
	if err := json.Unmarshal(ev.Item, &it); err != nil {
		return fmt.Errorf("openai: decode output item: %w", err)
	}
	st := s.item(ev.OutputIndex, it.ID)
	st.id = cmp.Or(it.ID, st.id)
	switch it.Type {
	case "reasoning":
		s.startReasoning(st)
	case "function_call":
		st.callID = it.CallID
		s.startToolInput(st, it.Name)
	default:
	}
	return nil
}

func (s *responseStream) startText(st *itemState, idx int) *textState {
	ts, ok := st.text[idx]
	if !ok {
		ts = &textState{}
		st.text[idx] = ts
	}
	if !ts.started {
		ts.started = true
		s.emit(ai.TextStart{ID: textID(st.id, idx)})
	}
	return ts
}

func (s *responseStream) startReasoning(st *itemState) {
	if !st.started {
		st.started = true
		s.emit(ai.ReasoningStart{ID: st.id})
	}
}

func (s *responseStream) reasoningDelta(st *itemState, text string) {
	st.acc.WriteString(text)
	s.emit(ai.ReasoningDelta{ID: st.id, Text: text})
}

func (s *responseStream) startToolInput(st *itemState, name string) {
	if !st.started {
		st.started = true
		s.emit(ai.ToolInputStart{ID: st.callID, Name: name})
	}
}

func (s *responseStream) finishItem(idx int, raw json.RawMessage) error {
	s.done[idx] = true
	var it outputItem
	if err := json.Unmarshal(raw, &it); err != nil {
		return fmt.Errorf("openai: decode output item: %w", err)
	}
	conv, err := convertItem(raw)
	if err != nil {
		return err
	}
	s.flags.clientCall = s.flags.clientCall || conv.clientCall
	s.flags.refusal = s.flags.refusal || conv.refusal
	st := s.item(idx, it.ID)
	switch it.Type {
	case "reasoning":
		s.startReasoning(st)
		s.emitRemainder(st.acc.String(), reasoningText(it), func(d string) { s.reasoningDelta(st, d) })
		s.emit(ai.ReasoningEnd{ID: it.ID, Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})})
	case "message":
		s.finishMessage(st, conv)
	case "function_call":
		st.callID = it.CallID
		s.startToolInput(st, it.Name)
		s.emitRemainder(st.acc.String(), it.Arguments, func(d string) { s.emit(ai.ToolInputDelta{ID: it.CallID, Text: d}) })
		s.emit(ai.ToolInputEnd{ID: it.CallID})
		s.emitParts(conv.parts)
	default:
		s.emitParts(conv.parts)
	}
	return nil
}

func (s *responseStream) finishMessage(st *itemState, conv converted) {
	for i, p := range conv.parts {
		t, ok := p.(ai.Text)
		if !ok {
			continue
		}
		ts := s.startText(st, i)
		id := textID(st.id, i)
		s.emitRemainder(ts.acc.String(), t.Text, func(d string) { s.emit(ai.TextDelta{ID: id, Text: d}) })
		s.emit(ai.TextEnd{ID: id, Provider: t.Provider})
		for _, c := range t.Citations {
			s.emit(c)
		}
	}
}

func (s *responseStream) emitParts(parts ai.Parts) {
	for _, p := range parts {
		if sp, ok := p.(ai.StreamPart); ok {
			s.emit(sp)
		}
	}
}

func (s *responseStream) emitRemainder(sent, final string, emit func(string)) {
	if rest, ok := strings.CutPrefix(final, sent); ok && rest != "" {
		emit(rest)
	}
}

func (s *responseStream) complete(raw json.RawMessage) error {
	var body responseBody
	if err := json.Unmarshal(raw, &body); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}
	for i, item := range body.Output {
		if !s.done[i] {
			if err := s.finishItem(i, item); err != nil {
				return err
			}
		}
	}
	incomplete := ""
	if body.IncompleteDetails != nil {
		incomplete = body.IncompleteDetails.Reason
	}
	finish, rawFinish := finishOf(body.Status, incomplete, s.flags)
	s.emit(ai.StepEnd{Usage: openaiwire.Usage(body.Usage), FinishReason: finish, FinishRaw: rawFinish, Warnings: s.warnings, Provider: body.metadata()})
	return nil
}

func (s *responseStream) failed(raw json.RawMessage) error {
	var body responseBody
	if err := json.Unmarshal(raw, &body); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}
	if err := body.failure(); err != nil {
		return err
	}
	return errors.New("openai: response failed")
}

func textID(itemID string, idx int) string { return fmt.Sprintf("%s:%d", itemID, idx) }
