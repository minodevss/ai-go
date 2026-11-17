package ai

import (
	"context"
	"errors"
	"iter"
)

type streamState int

const (
	streamIdle streamState = iota
	streamActive
	streamDone
)

// TextStream is lazy and single use: the first call to Parts, Text or Result starts it. It is not safe for concurrent use.
type TextStream struct {
	run    func(emit func(StreamPart) bool) (*TextResult, error)
	state  streamState
	res    *TextResult
	err    error
	misuse error
}

// GenerateText returns the completed steps together with any error, so callers can append res.Messages and continue.
func GenerateText(ctx context.Context, m LanguageModel, req TextRequest) (*TextResult, error) {
	r := newRunner(m, req.Clone(), false, func(StreamPart) bool { return true })
	return r.run(ctx)
}

func StreamText(ctx context.Context, m LanguageModel, req TextRequest) *TextStream {
	req = req.Clone()
	return &TextStream{
		run: func(emit func(StreamPart) bool) (*TextResult, error) {
			return newRunner(m, req, true, emit).run(ctx)
		},
	}
}

// Parts stops the stream when the loop breaks; Result then returns context.Canceled with the completed steps.
func (s *TextStream) Parts() iter.Seq[StreamPart] {
	return func(yield func(StreamPart) bool) {
		if !s.begin() {
			return
		}
		s.finish(s.run(yield))
	}
}

func (s *TextStream) Text() iter.Seq[string] {
	return func(yield func(string) bool) {
		if !s.begin() {
			return
		}
		s.finish(s.run(func(p StreamPart) bool {
			if d, ok := p.(TextDelta); ok {
				return yield(d.Text)
			}
			return true
		}))
	}
}

// Err returns the terminal error of a finished stream, or ErrStreamConsumed and ErrStreamActive after a second consumer.
func (s *TextStream) Err() error {
	if s.misuse != nil {
		return s.misuse
	}
	return s.err
}

// Result drains the stream if nobody has consumed it and returns the completed steps together with the error.
func (s *TextStream) Result() (*TextResult, error) {
	switch s.state {
	case streamActive:
		return nil, ErrStreamActive
	case streamIdle:
		s.state = streamActive
		s.finish(s.run(func(StreamPart) bool { return true }))
	case streamDone:
	}
	return s.res, s.err
}

func (s *TextStream) begin() bool {
	switch s.state {
	case streamIdle:
		s.state = streamActive
		return true
	case streamActive:
		s.misuse = ErrStreamActive
	case streamDone:
		s.misuse = ErrStreamConsumed
	}
	return false
}

func (s *TextStream) finish(res *TextResult, err error) {
	s.state = streamDone
	s.res = res
	if errors.Is(err, errConsumerStopped) {
		err = context.Canceled
	}
	s.err = err
}
