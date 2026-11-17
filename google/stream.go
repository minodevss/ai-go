package google

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/sse"
	"github.com/minodevss/ai-go/internal/transport"
)

type textOut struct {
	Text             string `json:"text"`
	Thought          bool   `json:"thought,omitzero"`
	ThoughtSignature string `json:"thoughtSignature,omitzero"`
}

type block struct {
	id      string
	thought bool
	text    strings.Builder
}

type heldText struct {
	id       string
	provider ai.ProviderData
}

type streamParser struct {
	b            *backend
	yield        func(ai.StreamPart, error) bool
	warnings     []ai.Warning
	hold         bool
	conv         *converter
	meta         metadata
	usage        json.RawMessage
	finishReason string
	clientCall   bool
	finished     bool
	open         *block
	seq          int
	texts        []string
	held         []heldText
	stopped      bool
}

func (m *LanguageModel) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	req, warnings, err := m.request(c)
	if err != nil {
		return nil, err
	}
	resp, err := m.backend.client.Do(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":streamGenerateContent?alt=sse", Body: req, Header: c.Headers, Stream: true})
	if err != nil {
		return nil, fail(err)
	}
	return &ai.StreamResponse{Parts: m.backend.streamParts(resp.Body, warnings, hasGroundingTool(c.Tools)), Header: resp.Header}, nil
}

func (b *backend) streamParts(body io.ReadCloser, warnings []ai.Warning, hold bool) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		defer body.Close()
		s := &streamParser{b: b, yield: yield, warnings: warnings, hold: hold, conv: &converter{b: b}}
		r := sse.NewReader(body)
		for {
			ev, err := r.Next()
			if errors.Is(err, io.EOF) {
				if !s.finished {
					yield(nil, errors.New("google: stream ended before the response finished"))
					return
				}
				s.finish()
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("google: read stream: %w", err))
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

func (s *streamParser) emit(p ai.StreamPart) {
	if s.stopped {
		return
	}
	if !s.yield(p, nil) {
		s.stopped = true
	}
}

func (s *streamParser) handle(data string) error {
	var body responseBody
	if err := json.Unmarshal([]byte(data), &body); err != nil {
		return fmt.Errorf("google: decode stream event: %w", err)
	}
	if body.Error != nil {
		return body.Error.apiError(s.b.name, []byte(data))
	}
	cand := body.first()
	s.meta.take(body, cand)
	s.conv.responseID = s.meta.responseID
	if len(body.UsageMetadata) > 0 && string(body.UsageMetadata) != "null" {
		s.usage = body.UsageMetadata
	}
	for _, raw := range cand.Content.Parts {
		if err := s.part(raw); err != nil {
			return err
		}
	}
	if cand.FinishReason != "" {
		s.finishReason = cand.FinishReason
		s.finished = true
	}
	if blockReason(body.PromptFeedback) != "" {
		s.finished = true
	}
	return nil
}

func (s *streamParser) part(raw json.RawMessage) error {
	var p partProbe
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("google: decode part: %w", err)
	}
	if p.Text != nil && p.FunctionCall == nil && p.ExecutableCode == nil && p.CodeResult == nil && p.InlineData == nil && p.FileData == nil {
		s.textPart(p)
		return nil
	}
	s.closeBlock()
	converted, err := s.conv.part(raw)
	if err != nil {
		return err
	}
	if call, ok := converted.(ai.ToolCall); ok && p.FunctionCall != nil {
		s.clientCall = true
		s.emit(ai.ToolInputStart{ID: call.ID, Name: call.Name})
		s.emit(ai.ToolInputDelta{ID: call.ID, Text: string(call.Input)})
		s.emit(ai.ToolInputEnd{ID: call.ID})
	}
	if sp, ok := converted.(ai.StreamPart); ok {
		s.emit(sp)
	}
	return nil
}

func (s *streamParser) textPart(p partProbe) {
	text := *p.Text
	if s.open != nil && (s.open.thought != p.Thought || p.ThoughtSignature != "" && s.open.text.Len() > 0) {
		s.closeBlock()
	}
	if text == "" && p.ThoughtSignature == "" {
		return
	}
	if s.open == nil {
		s.openBlock(p.Thought)
	}
	if text != "" {
		s.open.text.WriteString(text)
		if s.open.thought {
			s.emit(ai.ReasoningDelta{ID: s.open.id, Text: text})
		} else {
			s.emit(ai.TextDelta{ID: s.open.id, Text: text})
		}
	}
	if p.ThoughtSignature != "" {
		s.closeBlockSigned(p.ThoughtSignature)
	}
}

func (s *streamParser) openBlock(thought bool) {
	s.seq++
	s.open = &block{thought: thought}
	if thought {
		s.open.id = fmt.Sprintf("reasoning-%d", s.seq)
		s.emit(ai.ReasoningStart{ID: s.open.id})
		return
	}
	s.open.id = fmt.Sprintf("text-%d", s.seq)
	s.emit(ai.TextStart{ID: s.open.id})
}

func (s *streamParser) closeBlock() { s.closeBlockSigned("") }

func (s *streamParser) closeBlockSigned(signature string) {
	blk := s.open
	if blk == nil {
		return
	}
	s.open = nil
	raw := pdata.JSON(providerName, textOut{Text: blk.text.String(), Thought: blk.thought, ThoughtSignature: signature})
	provider := s.b.encode(partData{Part: raw})
	if blk.thought {
		s.emit(ai.ReasoningEnd{ID: blk.id, Provider: provider})
		return
	}
	s.texts = append(s.texts, blk.text.String())
	if s.hold {
		s.held = append(s.held, heldText{id: blk.id, provider: provider})
		return
	}
	s.emit(ai.TextEnd{ID: blk.id, Provider: provider})
}

func (s *streamParser) finish() {
	s.closeBlock()
	cites := s.b.citations(s.texts, s.meta.fields["groundingMetadata"])
	if s.hold {
		for i, h := range s.held {
			s.emit(ai.TextEnd{ID: h.id, Provider: h.provider})
			for _, c := range cites[i] {
				s.emit(c)
			}
		}
	} else {
		for _, list := range cites {
			for _, c := range list {
				s.emit(c)
			}
		}
	}
	finish, raw := finishOf(s.finishReason, blockReason(s.meta.fields["promptFeedback"]), s.clientCall)
	s.emit(ai.StepEnd{Usage: usageOf(s.usage), FinishReason: finish, FinishRaw: raw, Warnings: s.warnings, Provider: s.meta.provider(s.b)})
}
