package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"

	"github.com/minodevss/ai-go/internal/partialjson"
	"github.com/minodevss/ai-go/internal/schema"
)

// ObjectStream is single use like TextStream. Partials and Elements are best effort; Result validates the final output.
type ObjectStream[T any] struct {
	text    *TextStream
	wrapped bool
	settled bool
	value   T
	err     error
}

func StreamObject[T any](ctx context.Context, m LanguageModel, req TextRequest) *ObjectStream[T] {
	req.Output = ObjectOutput[T]()
	s := &ObjectStream[T]{text: StreamText(ctx, m, req)}
	if raw, err := schema.For[T](); err == nil {
		s.wrapped = !rootIsObject(raw)
	}
	return s
}

func (s *ObjectStream[T]) Parts() iter.Seq[StreamPart] { return s.text.Parts() }

func (s *ObjectStream[T]) Partials() iter.Seq[T] {
	return func(yield func(T) bool) {
		s.consume(yield)
	}
}

func (s *ObjectStream[T]) Err() error {
	if s.text.state == streamDone {
		s.settle()
		if s.err != nil {
			return s.err
		}
	}
	return s.text.Err()
}

func (s *ObjectStream[T]) Result() (T, *TextResult, error) {
	res, err := s.text.Result()
	if errors.Is(err, ErrStreamActive) {
		return *new(T), nil, err
	}
	s.settle()
	return s.value, res, s.err
}

func Elements[E any](s *ObjectStream[[]E]) iter.Seq[E] {
	return func(yield func(E) bool) {
		emitted := 0
		stopped := false
		s.consume(func(partial []E) bool {
			for emitted < len(partial)-1 {
				if !yield(partial[emitted]) {
					stopped = true
					return false
				}
				emitted++
			}
			return true
		})
		if stopped || s.err != nil {
			return
		}
		for ; emitted < len(s.value); emitted++ {
			if !yield(s.value[emitted]) {
				return
			}
		}
	}
}

func (s *ObjectStream[T]) consume(onPartial func(T) bool) {
	snap := &snapshotter{parser: partialjson.New(0)}
	emit := func() bool {
		snapshot, ok := snap.take()
		if !ok {
			return true
		}
		value, decoded := s.decodePartial(snapshot)
		return !decoded || onPartial(value)
	}
	for p := range s.text.Parts() {
		switch v := p.(type) {
		case StepStart:
			snap.reset()
		case TextDelta:
			if snap.parser.Write(v.Text) == nil && snap.due() && !emit() {
				return
			}
		default:
		}
	}
	if snap.pending() && !emit() {
		return
	}
	s.settle()
}

const snapshotGrowthDivisor = 8

type snapshotter struct {
	parser *partialjson.Parser
	taken  int
	last   []byte
}

func (s *snapshotter) reset() { *s = snapshotter{parser: partialjson.New(0)} }

func (s *snapshotter) due() bool {
	return s.parser.Len()-s.taken >= s.taken/snapshotGrowthDivisor
}

func (s *snapshotter) pending() bool { return s.parser.Len() > s.taken }

func (s *snapshotter) take() ([]byte, bool) {
	s.taken = s.parser.Len()
	snapshot := s.parser.Partial()
	if snapshot == nil || bytes.Equal(snapshot, s.last) {
		return nil, false
	}
	s.last = snapshot
	return snapshot, true
}

func (s *ObjectStream[T]) decodePartial(snapshot []byte) (T, bool) {
	var v T
	doc := snapshot
	if s.wrapped {
		var wrapper struct {
			Result json.RawMessage `json:"result"`
		}
		if json.Unmarshal(snapshot, &wrapper) != nil || len(wrapper.Result) == 0 {
			return v, false
		}
		doc = wrapper.Result
	}
	if json.Unmarshal(doc, &v) != nil {
		return *new(T), false
	}
	return v, true
}

func (s *ObjectStream[T]) settle() {
	if s.settled {
		return
	}
	res, err := s.text.Result()
	if errors.Is(err, ErrStreamActive) {
		return
	}
	s.settled = true
	s.err = err
	if err == nil {
		s.value, s.err = DecodeObject[T](res)
	}
}
