package bedrock

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
	"github.com/minodevss/ai-go/internal/eventstream"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	eventStreamType = "application/vnd.amazon.eventstream"
	headerMsgType   = ":message-type"
	headerEvent     = ":event-type"
	headerException = ":exception-type"
	headerErrorCode = ":error-code"
	headerErrorMsg  = ":error-message"
)

type blockKind int

const (
	kindText blockKind = iota + 1
	kindReasoning
	kindTool
	kindToolResult
	kindOther
)

type startEvent struct {
	Index int `json:"contentBlockIndex"`
	Start struct {
		ToolUse *struct {
			ID   string `json:"toolUseId"`
			Name string `json:"name"`
			Type string `json:"type"`
		} `json:"toolUse"`
		ToolResult *struct {
			ID     string `json:"toolUseId"`
			Status string `json:"status"`
			Type   string `json:"type"`
		} `json:"toolResult"`
	} `json:"start"`
}

type deltaEvent struct {
	Index int `json:"contentBlockIndex"`
	Delta struct {
		Text      *string `json:"text"`
		Reasoning *struct {
			Text      *string `json:"text"`
			Signature *string `json:"signature"`
			Redacted  *string `json:"redactedContent"`
		} `json:"reasoningContent"`
		ToolUse *struct {
			Input string `json:"input"`
		} `json:"toolUse"`
		Citation   json.RawMessage   `json:"citation"`
		ToolResult []json.RawMessage `json:"toolResult"`
	} `json:"delta"`
}

type stopEvent struct {
	Index int `json:"contentBlockIndex"`
}

type messageStopEvent struct {
	StopReason                    string          `json:"stopReason"`
	AdditionalModelResponseFields json.RawMessage `json:"additionalModelResponseFields"`
}

type metadataEvent struct {
	Usage             json.RawMessage `json:"usage"`
	Metrics           json.RawMessage `json:"metrics"`
	Trace             json.RawMessage `json:"trace"`
	PerformanceConfig json.RawMessage `json:"performanceConfig"`
	ServiceTier       json.RawMessage `json:"serviceTier"`
}

type blockState struct {
	kind      blockKind
	id        string
	name      string
	toolType  string
	status    string
	text      strings.Builder
	signature strings.Builder
	redacted  string
	input     strings.Builder
	cites     []json.RawMessage
	results   []any
	forced    bool
	opened    bool
}

type converseStream struct {
	yield    func(ai.StreamPart, error) bool
	model    string
	warnings []ai.Warning
	forced   string
	blocks   map[int]*blockState
	names    map[string]string
	stop     messageStopEvent
	stopped  bool
	done     bool
	hasTool  bool
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	p, err := m.prepare(c, true)
	if err != nil {
		return nil, err
	}
	p.header.Set("Accept", eventStreamType)
	resp, err := m.client.Do(ctx, transport.Request{Path: modelPath(m.id, "converse-stream"), Body: p.req, Header: p.header, Stream: true})
	if err != nil {
		return nil, err
	}
	return &ai.StreamResponse{Parts: converseParts(resp.Body, m.id, p), Header: resp.Header}, nil
}

func converseParts(body io.ReadCloser, model string, p *prepared) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &converseStream{yield: yield, model: model, warnings: p.warnings, forced: p.forced, blocks: map[int]*blockState{}, names: map[string]string{}}
		d := eventstream.NewDecoder(body)
		for {
			msg, err := d.Next()
			if errors.Is(err, io.EOF) {
				s.eof()
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("bedrock: read stream: %w", err))
				return
			}
			if err := s.handle(msg); err != nil {
				yield(nil, err)
				return
			}
			if s.done {
				return
			}
		}
	}
}

func (s *converseStream) emit(p ai.StreamPart) {
	if s.done {
		return
	}
	if !s.yield(p, nil) {
		s.done = true
	}
}

func (s *converseStream) eof() {
	if s.done {
		return
	}
	if !s.stopped {
		s.yield(nil, errors.New("bedrock: stream ended before the message finished"))
		return
	}
	s.finish(metadataEvent{})
}

func (s *converseStream) handle(msg eventstream.Message) error {
	kind, _ := msg.String(headerMsgType)
	switch kind {
	case "exception":
		typ, _ := msg.String(headerException)
		return streamError(typ, msg.Payload)
	case "error":
		code, _ := msg.String(headerErrorCode)
		message, _ := msg.String(headerErrorMsg)
		message = transport.Truncate(message)
		return &ai.APIError{Provider: providerName, Code: code, Message: message, ContextTooLong: contextTooLong(0, code, message)}
	default:
	}
	event, _ := msg.String(headerEvent)
	switch event {
	case "contentBlockStart":
		return s.blockStart(msg.Payload)
	case "contentBlockDelta":
		return s.blockDelta(msg.Payload)
	case "contentBlockStop":
		return s.blockStop(msg.Payload)
	case "messageStop":
		if err := json.Unmarshal(msg.Payload, &s.stop); err != nil {
			return fmt.Errorf("bedrock: decode message stop: %w", err)
		}
		s.stopped = true
	case "metadata":
		var md metadataEvent
		if err := json.Unmarshal(msg.Payload, &md); err != nil {
			return fmt.Errorf("bedrock: decode metadata: %w", err)
		}
		s.finish(md)
	default:
	}
	return nil
}

func (s *converseStream) block(index int, kind blockKind) *blockState {
	if b, ok := s.blocks[index]; ok {
		return b
	}
	b := &blockState{kind: kind, id: strconv.Itoa(index)}
	s.blocks[index] = b
	return b
}

func (s *converseStream) blockStart(payload []byte) error {
	var ev startEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return fmt.Errorf("bedrock: decode content block start: %w", err)
	}
	switch {
	case ev.Start.ToolUse != nil:
		u := ev.Start.ToolUse
		b := s.block(ev.Index, kindTool)
		b.id, b.name, b.toolType = u.ID, u.Name, u.Type
		if b.toolType == "" && s.forced != "" && b.name == s.forced {
			b.forced = true
			s.emit(ai.TextStart{ID: strconv.Itoa(ev.Index)})
			return nil
		}
		if b.toolType != "" {
			s.names[b.id] = b.name
		}
		s.emit(ai.ToolInputStart{ID: b.id, Name: b.name})
	case ev.Start.ToolResult != nil:
		r := ev.Start.ToolResult
		b := s.block(ev.Index, kindToolResult)
		b.id, b.status, b.toolType = r.ID, r.Status, r.Type
	default:
		s.block(ev.Index, kindOther)
	}
	return nil
}

func (s *converseStream) blockDelta(payload []byte) error {
	var ev deltaEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return fmt.Errorf("bedrock: decode content block delta: %w", err)
	}
	d := ev.Delta
	id := strconv.Itoa(ev.Index)
	switch {
	case d.Text != nil:
		s.textDelta(s.block(ev.Index, kindText), id, *d.Text)
	case d.Citation != nil:
		b := s.block(ev.Index, kindText)
		s.openText(b, id)
		b.cites = append(b.cites, d.Citation)
	case d.Reasoning != nil:
		b := s.block(ev.Index, kindReasoning)
		s.reasoningDelta(b, id, *d.Reasoning)
	case d.ToolUse != nil:
		b := s.block(ev.Index, kindTool)
		b.input.WriteString(d.ToolUse.Input)
		switch {
		case d.ToolUse.Input == "":
		case b.forced:
			s.emit(ai.TextDelta{ID: id, Text: d.ToolUse.Input})
		default:
			s.emit(ai.ToolInputDelta{ID: b.id, Text: d.ToolUse.Input})
		}
	case d.ToolResult != nil:
		b := s.block(ev.Index, kindToolResult)
		for _, raw := range d.ToolResult {
			b.addResult(raw)
		}
	default:
	}
	return nil
}

func (s *converseStream) openText(b *blockState, id string) {
	if !b.opened {
		b.opened = true
		s.emit(ai.TextStart{ID: id})
	}
}

func (s *converseStream) textDelta(b *blockState, id, text string) {
	s.openText(b, id)
	if text == "" {
		return
	}
	b.text.WriteString(text)
	s.emit(ai.TextDelta{ID: id, Text: text})
}

func (s *converseStream) reasoningDelta(b *blockState, id string, d struct {
	Text      *string `json:"text"`
	Signature *string `json:"signature"`
	Redacted  *string `json:"redactedContent"`
},
) {
	if !b.opened {
		b.opened = true
		s.emit(ai.ReasoningStart{ID: id})
	}
	if d.Signature != nil {
		b.signature.WriteString(*d.Signature)
	}
	if d.Redacted != nil {
		b.redacted += *d.Redacted
	}
	if d.Text != nil && *d.Text != "" {
		b.text.WriteString(*d.Text)
		s.emit(ai.ReasoningDelta{ID: id, Text: *d.Text})
	}
}

func (b *blockState) addResult(raw json.RawMessage) {
	var probe struct {
		Text *string         `json:"text"`
		JSON json.RawMessage `json:"json"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		b.results = append(b.results, raw)
		return
	}
	if probe.Text != nil {
		if n := len(b.results); n > 0 {
			if last, ok := b.results[n-1].(*textBlock); ok {
				last.Text += *probe.Text
				return
			}
		}
		b.results = append(b.results, &textBlock{Text: *probe.Text})
		return
	}
	b.results = append(b.results, raw)
}

func (s *converseStream) blockStop(payload []byte) error {
	var ev stopEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return fmt.Errorf("bedrock: decode content block stop: %w", err)
	}
	b, ok := s.blocks[ev.Index]
	if !ok {
		return nil
	}
	delete(s.blocks, ev.Index)
	id := strconv.Itoa(ev.Index)
	switch {
	case b.kind == kindText:
		s.endText(b, id)
	case b.kind == kindReasoning:
		s.emit(ai.ReasoningEnd{ID: id, Provider: blockData(reasoningBlock(b), s.model)})
	case b.kind == kindTool && b.forced:
		s.emit(ai.TextEnd{ID: id, Provider: pdata.Encode(providerName, partData{ForcedTool: true})})
	case b.kind == kindTool:
		return s.endTool(b)
	case b.kind == kindToolResult:
		s.endToolResult(b)
	default:
	}
	return nil
}

type citationsContentBlock struct {
	CitationsContent struct {
		Content []textBlock       `json:"content"`
		Cites   []json.RawMessage `json:"citations"`
	} `json:"citationsContent"`
}

func (s *converseStream) endText(b *blockState, id string) {
	if len(b.cites) == 0 {
		s.emit(ai.TextEnd{ID: id})
		return
	}
	var blk citationsContentBlock
	blk.CitationsContent.Content = []textBlock{{Text: b.text.String()}}
	blk.CitationsContent.Cites = b.cites
	s.emit(ai.TextEnd{ID: id, Provider: blockData(pdata.JSON(providerName, blk), "")})
	for _, raw := range b.cites {
		var cb citationBody
		if err := json.Unmarshal(raw, &cb); err == nil {
			s.emit(citationOf(raw, cb))
		}
	}
}

type reasoningBlockWire struct {
	ReasoningContent struct {
		Text     *reasoningTextWire `json:"reasoningText,omitzero"`
		Redacted string             `json:"redactedContent,omitzero"`
	} `json:"reasoningContent"`
}

type reasoningTextWire struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitzero"`
}

func reasoningBlock(b *blockState) json.RawMessage {
	var blk reasoningBlockWire
	if b.redacted != "" {
		blk.ReasoningContent.Redacted = b.redacted
	} else {
		blk.ReasoningContent.Text = &reasoningTextWire{Text: b.text.String(), Signature: b.signature.String()}
	}
	return pdata.JSON(providerName, blk)
}

type streamToolUse struct {
	ToolUse struct {
		ID    string          `json:"toolUseId"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
		Type  string          `json:"type,omitzero"`
	} `json:"toolUse"`
}

func (s *converseStream) endTool(b *blockState) error {
	input := strings.TrimSpace(b.input.String())
	if input == "" {
		input = "{}"
	}
	if !json.Valid([]byte(input)) {
		return fmt.Errorf("bedrock: tool input of block %s is not valid JSON", b.id)
	}
	var blk streamToolUse
	blk.ToolUse.ID, blk.ToolUse.Name, blk.ToolUse.Input, blk.ToolUse.Type = b.id, b.name, json.RawMessage(input), b.toolType
	executed := b.toolType != ""
	s.hasTool = s.hasTool || !executed
	s.emit(ai.ToolInputEnd{ID: b.id})
	s.emit(ai.ToolCall{ID: b.id, Name: b.name, Input: json.RawMessage(input), ProviderExecuted: executed, Provider: blockData(pdata.JSON(providerName, blk), s.model)})
	return nil
}

type streamToolResult struct {
	ToolResult struct {
		ID      string `json:"toolUseId"`
		Content []any  `json:"content"`
		Status  string `json:"status,omitzero"`
		Type    string `json:"type,omitzero"`
	} `json:"toolResult"`
}

func (s *converseStream) endToolResult(b *blockState) {
	if b.results == nil {
		b.results = []any{}
	}
	var blk streamToolResult
	blk.ToolResult.ID, blk.ToolResult.Content, blk.ToolResult.Status, blk.ToolResult.Type = b.id, b.results, b.status, b.toolType
	raw := pdata.JSON(providerName, blk)
	name := s.names[b.id]
	if name == "" {
		name = b.toolType
	}
	s.emit(ai.ToolResult{
		CallID:           b.id,
		Name:             name,
		Output:           ai.ToolOutput{JSON: pdata.JSON(providerName, b.results)},
		IsError:          b.status == "error",
		ProviderExecuted: true,
		Provider:         blockData(raw, s.model),
	})
}

func (s *converseStream) finish(md metadataEvent) {
	if s.done {
		return
	}
	finish, raw := finishOf(s.stop.StopReason, s.hasTool)
	s.emit(ai.StepEnd{
		Usage:        usageOf(md.Usage),
		FinishReason: finish,
		FinishRaw:    raw,
		Warnings:     s.warnings,
		Provider:     responseData(s.stop.AdditionalModelResponseFields, md.Metrics, md.Trace, md.PerformanceConfig, md.ServiceTier),
	})
	s.done = true
}
