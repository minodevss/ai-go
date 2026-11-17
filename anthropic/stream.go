package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/claudewire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
)

type streamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Message      json.RawMessage `json:"message"`
	ContentBlock json.RawMessage `json:"content_block"`
	Delta        json.RawMessage `json:"delta"`
	Usage        json.RawMessage `json:"usage"`
	Error        *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type deltaBody struct {
	Type        string          `json:"type"`
	Text        string          `json:"text"`
	Thinking    string          `json:"thinking"`
	Signature   string          `json:"signature"`
	PartialJSON string          `json:"partial_json"`
	Citation    json.RawMessage `json:"citation"`
	StopReason  string          `json:"stop_reason"`
	StopSeq     string          `json:"stop_sequence"`
	StopDetails json.RawMessage `json:"stop_details"`
	Container   *containerBody  `json:"container"`
}

type blockState struct {
	kind      string
	fields    map[string]json.RawMessage
	text      strings.Builder
	thinking  strings.Builder
	signature strings.Builder
	input     strings.Builder
	cites     []json.RawMessage
	forced    bool
}

type messageStream struct {
	yield    func(ai.StreamPart, error) bool
	warnings []ai.Warning
	forced   string
	blocks   map[int]*blockState
	names    map[string]string
	usage    map[string]json.RawMessage
	msg      messageBody
	stop     string
	hasTool  bool
	stopped  bool
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	p, err := m.prepare(c, true)
	if err != nil {
		return nil, err
	}
	resp, err := m.client.Do(ctx, transport.Request{Path: "/v1/messages", Body: p.req, Header: p.header, Stream: true})
	if err != nil {
		return nil, err
	}
	return &ai.StreamResponse{Parts: messageParts(resp.Body, p), Header: resp.Header}, nil
}

func messageParts(body io.ReadCloser, p *prepared) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &messageStream{yield: yield, warnings: p.warnings, forced: p.forced, blocks: map[int]*blockState{}, names: map[string]string{}, usage: map[string]json.RawMessage{}}
		r := sse.NewReader(body)
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				yield(nil, errors.New("anthropic: stream ended before the message finished"))
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("anthropic: read stream: %w", err))
				return
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

func (s *messageStream) emit(p ai.StreamPart) {
	if s.stopped {
		return
	}
	if !s.yield(p, nil) {
		s.stopped = true
	}
}

func (s *messageStream) handle(data string) (bool, error) {
	var ev streamEvent
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return false, fmt.Errorf("anthropic: decode stream event: %w", err)
	}
	switch ev.Type {
	case "message_start":
		return false, s.start(ev)
	case "content_block_start":
		return false, s.blockStart(ev)
	case "content_block_delta":
		return false, s.blockDelta(ev)
	case "content_block_stop":
		return false, s.blockStop(ev.Index)
	case "message_delta":
		return false, s.messageDelta(ev)
	case "message_stop":
		s.finish()
		return true, nil
	case "error":
		if ev.Error == nil {
			return true, eventError("", "")
		}
		return true, eventError(ev.Error.Type, ev.Error.Message)
	default:
		return false, nil
	}
}

func (s *messageStream) start(ev streamEvent) error {
	if err := json.Unmarshal(ev.Message, &s.msg); err != nil {
		return fmt.Errorf("anthropic: decode message: %w", err)
	}
	s.mergeUsage(s.msg.Usage)
	return nil
}

func (s *messageStream) mergeUsage(raw json.RawMessage) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return
	}
	for k, v := range fields {
		if string(v) != "null" {
			s.usage[k] = v
		}
	}
}

func (s *messageStream) blockStart(ev streamEvent) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(ev.ContentBlock, &fields); err != nil {
		return fmt.Errorf("anthropic: decode content block: %w", err)
	}
	var b contentBlock
	if err := json.Unmarshal(ev.ContentBlock, &b); err != nil {
		return fmt.Errorf("anthropic: decode content block: %w", err)
	}
	st := &blockState{kind: b.Type, fields: fields}
	s.blocks[ev.Index] = st
	id := strconv.Itoa(ev.Index)
	switch b.Type {
	case "text":
		s.openText(st, id, b.Text)
	case "thinking":
		s.emit(ai.ReasoningStart{ID: id})
		s.reasoningDelta(st, id, b.Thinking)
	case "tool_use", "server_tool_use", "mcp_tool_use":
		s.startTool(st, id, b)
	default:
	}
	return nil
}

func (s *messageStream) openText(st *blockState, id, initial string) {
	s.emit(ai.TextStart{ID: id})
	s.textDelta(st, id, initial)
}

func (s *messageStream) textDelta(st *blockState, id, text string) {
	if text == "" {
		return
	}
	st.text.WriteString(text)
	s.emit(ai.TextDelta{ID: id, Text: text})
}

func (s *messageStream) reasoningDelta(st *blockState, id, text string) {
	if text == "" {
		return
	}
	st.thinking.WriteString(text)
	s.emit(ai.ReasoningDelta{ID: id, Text: text})
}

func (s *messageStream) startTool(st *blockState, id string, b contentBlock) {
	if b.Type == "tool_use" && s.forced != "" && b.Name == s.forced {
		st.forced = true
		s.emit(ai.TextStart{ID: id})
		return
	}
	name := callName(b)
	s.names[b.ID] = name
	s.emit(ai.ToolInputStart{ID: b.ID, Name: name})
}

func (s *messageStream) blockDelta(ev streamEvent) error {
	st, ok := s.blocks[ev.Index]
	if !ok {
		return nil
	}
	var d deltaBody
	if err := json.Unmarshal(ev.Delta, &d); err != nil {
		return fmt.Errorf("anthropic: decode content block delta: %w", err)
	}
	id := strconv.Itoa(ev.Index)
	switch d.Type {
	case "text_delta":
		s.textDelta(st, id, d.Text)
	case "citations_delta":
		st.cites = append(st.cites, d.Citation)
	case "thinking_delta":
		s.reasoningDelta(st, id, d.Thinking)
	case "signature_delta":
		st.signature.WriteString(d.Signature)
	case "input_json_delta":
		st.input.WriteString(d.PartialJSON)
		if d.PartialJSON == "" {
			return nil
		}
		if st.forced {
			s.emit(ai.TextDelta{ID: id, Text: d.PartialJSON})
			return nil
		}
		s.emit(ai.ToolInputDelta{ID: s.toolID(st), Text: d.PartialJSON})
	default:
	}
	return nil
}

func (s *messageStream) toolID(st *blockState) string {
	var id string
	if err := json.Unmarshal(st.fields["id"], &id); err != nil {
		return ""
	}
	return id
}

func (s *messageStream) blockStop(idx int) error {
	st, ok := s.blocks[idx]
	if !ok {
		return nil
	}
	delete(s.blocks, idx)
	id := strconv.Itoa(idx)
	switch {
	case st.kind == "text":
		s.emit(ai.TextEnd{ID: id})
		for _, raw := range st.cites {
			if c, ok := citationOf(raw); ok {
				s.emit(c)
			}
		}
	case st.kind == "thinking":
		st.fields["thinking"] = pdata.JSON(providerName, st.thinking.String())
		if st.signature.Len() > 0 {
			st.fields["signature"] = pdata.JSON(providerName, startSignature(st)+st.signature.String())
		}
		s.emit(ai.ReasoningEnd{ID: id, Provider: blockData(pdata.JSON(providerName, st.fields))})
	case st.kind == "redacted_thinking":
		s.emit(ai.ReasoningStart{ID: id})
		s.emit(ai.ReasoningEnd{ID: id, Provider: blockData(pdata.JSON(providerName, st.fields))})
	case st.forced:
		s.emit(ai.TextEnd{ID: id, Provider: pdata.Encode(providerName, partData{ForcedTool: true})})
	case st.kind == "tool_use", st.kind == "server_tool_use", st.kind == "mcp_tool_use":
		return s.finishTool(st)
	default:
		return s.finishWhole(st)
	}
	return nil
}

func startSignature(st *blockState) string {
	var sig string
	if err := json.Unmarshal(st.fields["signature"], &sig); err != nil {
		return ""
	}
	return sig
}

func (s *messageStream) finishTool(st *blockState) error {
	if acc := strings.TrimSpace(st.input.String()); acc != "" {
		if !json.Valid([]byte(acc)) {
			return fmt.Errorf("anthropic: tool input of block %s is not valid JSON", s.toolID(st))
		}
		st.fields["input"] = json.RawMessage(acc)
	}
	raw := pdata.JSON(providerName, st.fields)
	var b contentBlock
	if err := json.Unmarshal(raw, &b); err != nil {
		return fmt.Errorf("anthropic: decode content block: %w", err)
	}
	call := ai.ToolCall{ID: b.ID, Name: callName(b), Input: claudewire.ToolInput(b.Input), ProviderExecuted: b.Type != "tool_use", Provider: blockData(raw)}
	s.hasTool = s.hasTool || b.Type == "tool_use"
	s.emit(ai.ToolInputEnd{ID: b.ID})
	s.emit(call)
	return nil
}

func (s *messageStream) finishWhole(st *blockState) error {
	raw := pdata.JSON(providerName, st.fields)
	var b contentBlock
	if err := json.Unmarshal(raw, &b); err != nil {
		return fmt.Errorf("anthropic: decode content block: %w", err)
	}
	conv := &converter{names: s.names}
	if p, ok := conv.convert(raw, b); ok {
		if sp, ok := p.(ai.StreamPart); ok {
			s.emit(sp)
		}
	}
	return nil
}

func (s *messageStream) messageDelta(ev streamEvent) error {
	var d deltaBody
	if err := json.Unmarshal(ev.Delta, &d); err != nil {
		return fmt.Errorf("anthropic: decode message delta: %w", err)
	}
	if d.StopReason != "" {
		s.stop = d.StopReason
	}
	if d.StopSeq != "" {
		s.msg.StopSequence = d.StopSeq
	}
	if len(d.StopDetails) > 0 {
		s.msg.StopDetails = d.StopDetails
	}
	if d.Container != nil && d.Container.ID != "" {
		s.msg.Container = d.Container
	}
	s.mergeUsage(ev.Usage)
	return nil
}

func (s *messageStream) finish() {
	usage := pdata.JSON(providerName, s.usage)
	s.msg.Usage = usage
	finish, raw := finishOf(s.stop)
	if finish == ai.FinishToolCalls && !s.hasTool {
		finish = ai.FinishStop
	}
	s.emit(ai.StepEnd{Usage: usageOf(usage), FinishReason: finish, FinishRaw: raw, Warnings: s.warnings, Provider: s.msg.metadata()})
}
