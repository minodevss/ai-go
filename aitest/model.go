// Package aitest provides scripted models for testing code built on ai without a network.
package aitest

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"github.com/minodevss/ai-go"
)

type Item interface {
	apply(*turn)
}

type TurnScript struct {
	items []Item
}

type event struct {
	kind   eventKind
	text   []string
	name   string
	input  string
	err    error
	delay  time.Duration
	part   ai.Part
	stream ai.StreamPart
}

type eventKind int

const (
	eventText eventKind = iota + 1
	eventReasoning
	eventToolCall
	eventErr
	eventDelay
	eventPart
	eventStreamPart
)

type turn struct {
	events []event
	finish ai.FinishReason
	usage  ai.Usage
}

type itemFunc func(*turn)

func (f itemFunc) apply(t *turn) { f(t) }

func Text(s string) Item { return TextChunks(s) }

func TextChunks(chunks ...string) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventText, text: chunks}) })
}

func Reasoning(s string) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventReasoning, text: []string{s}}) })
}

func ToolCall(name, jsonInput string) Item {
	return itemFunc(func(t *turn) {
		t.events = append(t.events, event{kind: eventToolCall, name: name, input: jsonInput})
	})
}

func Err(err error) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventErr, err: err}) })
}

func Delay(d time.Duration) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventDelay, delay: d}) })
}

func Part(p ai.Part) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventPart, part: p}) })
}

func StreamPart(p ai.StreamPart) Item {
	return itemFunc(func(t *turn) { t.events = append(t.events, event{kind: eventStreamPart, stream: p}) })
}

func Finish(r ai.FinishReason) Item {
	return itemFunc(func(t *turn) { t.finish = r })
}

func Usage(input, output int) Item {
	return itemFunc(func(t *turn) { t.usage = ai.Usage{Input: input, Output: output} })
}

func Turn(items ...Item) TurnScript { return TurnScript{items: items} }

func (t TurnScript) applyModel(m *Model) { m.turns = append(m.turns, t) }

// Model plays its turns in order, one per Generate or Stream call, and records every call.
type Model struct {
	mu       sync.Mutex
	turns    []TurnScript
	next     int
	ids      map[string]int
	calls    []ai.Call
	failures failures
}

func NewModel(opts ...ModelOption) *Model {
	m := &Model{}
	for _, opt := range opts {
		opt.applyModel(m)
	}
	return m
}

func (m *Model) Provider() string { return "aitest" }

func (m *Model) ModelID() string { return "scripted" }

func (m *Model) Calls() []ai.Call {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.Call(nil), m.calls...)
}

func (m *Model) begin(c *ai.Call) (turn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, *c.Clone())
	if err := m.failures.next(); err != nil {
		return turn{}, err
	}
	if m.next >= len(m.turns) {
		return turn{}, fmt.Errorf("aitest: no scripted turn for call %d", m.next+1)
	}
	t := turn{usage: ai.Usage{Input: 10, Output: 5}}
	for _, item := range m.turns[m.next].items {
		item.apply(&t)
	}
	m.next++
	return t, nil
}

func (m *Model) nextID(prefix string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ids == nil {
		m.ids = map[string]int{}
	}
	m.ids[prefix]++
	return fmt.Sprintf("%s_%d", prefix, m.ids[prefix])
}

func (t turn) finishReason() ai.FinishReason {
	if t.finish != "" {
		return t.finish
	}
	for _, e := range t.events {
		if e.kind == eventToolCall {
			return ai.FinishToolCalls
		}
	}
	return ai.FinishStop
}

func (m *Model) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	t, err := m.begin(c)
	if err != nil {
		return nil, err
	}
	var content ai.Parts
	for _, e := range t.events {
		switch e.kind {
		case eventDelay:
			if err := wait(ctx, e.delay); err != nil {
				return nil, err
			}
		case eventErr:
			return nil, e.err
		case eventText:
			content = append(content, ai.Text{Text: join(e.text)})
		case eventReasoning:
			content = append(content, ai.Reasoning{Text: join(e.text)})
		case eventToolCall:
			content = append(content, ai.ToolCall{ID: m.nextID("call"), Name: e.name, Input: json.RawMessage(e.input)})
		case eventPart:
			content = append(content, e.part)
		case eventStreamPart:
		}
	}
	reason := t.finishReason()
	return &ai.Response{Content: content, Usage: t.usage, FinishReason: reason, FinishRaw: string(reason)}, nil
}

func (m *Model) Stream(ctx context.Context, c *ai.Call) (*ai.StreamResponse, error) {
	t, err := m.begin(c)
	if err != nil {
		return nil, err
	}
	i := 0
	for ; i < len(t.events); i++ {
		e := t.events[i]
		if e.kind == eventDelay {
			if err := wait(ctx, e.delay); err != nil {
				return nil, err
			}
			continue
		}
		if e.kind == eventErr {
			return nil, e.err
		}
		break
	}
	rest := t.events[i:]
	return &ai.StreamResponse{Parts: m.stream(ctx, t, rest)}, nil
}

func (m *Model) stream(ctx context.Context, t turn, events []event) iter.Seq2[ai.StreamPart, error] {
	return func(yield func(ai.StreamPart, error) bool) {
		for _, e := range events {
			switch e.kind {
			case eventDelay:
				if err := wait(ctx, e.delay); err != nil {
					yield(nil, err)
					return
				}
			case eventErr:
				yield(nil, e.err)
				return
			default:
				for _, p := range m.parts(e) {
					if !yield(p, nil) {
						return
					}
				}
			}
		}
		reason := t.finishReason()
		yield(ai.StepEnd{Usage: t.usage, FinishReason: reason, FinishRaw: string(reason)}, nil)
	}
}

func (m *Model) parts(e event) []ai.StreamPart {
	switch e.kind {
	case eventText:
		id := m.nextID("text")
		out := []ai.StreamPart{ai.TextStart{ID: id}}
		for _, c := range e.text {
			out = append(out, ai.TextDelta{ID: id, Text: c})
		}
		return append(out, ai.TextEnd{ID: id})
	case eventReasoning:
		id := m.nextID("reasoning")
		return []ai.StreamPart{ai.ReasoningStart{ID: id}, ai.ReasoningDelta{ID: id, Text: join(e.text)}, ai.ReasoningEnd{ID: id}}
	case eventToolCall:
		id := m.nextID("call")
		return []ai.StreamPart{
			ai.ToolInputStart{ID: id, Name: e.name},
			ai.ToolInputDelta{ID: id, Text: e.input},
			ai.ToolInputEnd{ID: id},
			ai.ToolCall{ID: id, Name: e.name, Input: json.RawMessage(e.input)},
		}
	case eventPart:
		return partToStream(m, e.part)
	case eventStreamPart:
		return []ai.StreamPart{e.stream}
	case eventDelay, eventErr:
	}
	return nil
}

func partToStream(m *Model, p ai.Part) []ai.StreamPart {
	switch v := p.(type) {
	case ai.Text:
		id := m.nextID("text")
		return []ai.StreamPart{ai.TextStart{ID: id}, ai.TextDelta{ID: id, Text: v.Text}, ai.TextEnd{ID: id, Provider: v.Provider}}
	case ai.Reasoning:
		id := m.nextID("reasoning")
		return []ai.StreamPart{ai.ReasoningStart{ID: id}, ai.ReasoningDelta{ID: id, Text: v.Text}, ai.ReasoningEnd{ID: id, Provider: v.Provider}}
	case ai.ToolCall:
		return []ai.StreamPart{v}
	case ai.ToolResult:
		return []ai.StreamPart{v}
	case ai.File:
		return []ai.StreamPart{v}
	case ai.Custom:
		return []ai.StreamPart{v}
	default:
		return nil
	}
}

func join(chunks []string) string { return strings.Join(chunks, "") }

func wait(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

var _ ai.LanguageModel = (*Model)(nil)
