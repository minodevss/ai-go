package ai

import (
	"context"
	"errors"
	"io"
	"iter"
	"net/http"
)

type flow[T any] struct {
	ctx       context.Context
	open      func(ctx context.Context) (iter.Seq2[T, error], error)
	telemetry Telemetry
	provider  string
	modelID   string
	state     streamState
	err       error
	misuse    error
}

func (f *flow[T]) seq() iter.Seq[T] {
	return func(yield func(T) bool) {
		if !f.begin() {
			return
		}
		err := f.run(yield)
		f.state = streamDone
		f.err = err
	}
}

func (f *flow[T]) run(yield func(T) bool) error {
	ctx, finish := startMedia(f.ctx, f.telemetry, f.provider, f.modelID)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	seq, err := f.open(ctx)
	if err == nil && seq == nil {
		err = errors.New("ai: model returned no stream")
	}
	if err == nil {
		for v, serr := range seq {
			if serr != nil {
				err = serr
				break
			}
			if !yield(v) {
				break
			}
		}
	}
	finish(Step{FinishReason: FinishStop}, err)
	return err
}

func (f *flow[T]) begin() bool {
	switch f.state {
	case streamIdle:
		f.state = streamActive
		return true
	case streamActive:
		f.misuse = ErrStreamActive
	case streamDone:
		f.misuse = ErrStreamConsumed
	}
	return false
}

func (f *flow[T]) fail() error {
	if f.misuse != nil {
		return f.misuse
	}
	return f.err
}

type SpeechStreamRequest struct {
	Text         iter.Seq[string]
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	Headers      http.Header
	Provider     ProviderData
	Telemetry    Telemetry
}

// SpeechStream is lazy and single use: ranging Audio sends the request. Stopping the loop early is not an error. Streams are not retried, because the text iterator cannot be replayed.
type SpeechStream struct {
	f         flow[[]byte]
	mediaType string
}

func StreamSpeech(ctx context.Context, m SpeechStreamModel, req SpeechStreamRequest) *SpeechStream {
	s := &SpeechStream{}
	call := &SpeechStreamCall{
		Text:         req.Text,
		Voice:        req.Voice,
		Format:       req.Format,
		Instructions: req.Instructions,
		Speed:        clonePtr(req.Speed),
		Headers:      req.Headers.Clone(),
		Provider:     req.Provider.clone(),
	}
	s.f = flow[[]byte]{ctx: ctx, telemetry: req.Telemetry, open: func(ctx context.Context) (iter.Seq2[[]byte, error], error) {
		if m == nil {
			return nil, errors.New("ai: nil model")
		}
		if req.Text == nil {
			return nil, errors.New("ai: speech: no text")
		}
		resp, err := m.StreamSpeech(ctx, call)
		if err != nil {
			return nil, err
		}
		if resp == nil {
			return nil, errors.New("ai: model returned no stream")
		}
		s.mediaType = resp.MediaType
		return resp.Audio, nil
	}}
	if m != nil {
		s.f.provider, s.f.modelID = m.Provider(), m.ModelID()
	}
	return s
}

func (s *SpeechStream) Audio() iter.Seq[[]byte] { return s.f.seq() }

// MediaType is known once Audio has started yielding.
func (s *SpeechStream) MediaType() string { return s.mediaType }

// Err returns the terminal error of a finished stream, or ErrStreamConsumed and ErrStreamActive after a second consumer.
func (s *SpeechStream) Err() error { return s.f.fail() }

type TranscribeStreamRequest struct {
	Audio     io.Reader
	MediaType string
	Language  string
	Headers   http.Header
	Provider  ProviderData
	Telemetry Telemetry
}

// TranscribeStream is lazy and single use: ranging Segments sends the request. Partial segments are replaced by later ones.
type TranscribeStream struct {
	f flow[Segment]
}

func StreamTranscribe(ctx context.Context, m TranscriptionStreamModel, req TranscribeStreamRequest) *TranscribeStream {
	s := &TranscribeStream{}
	call := &TranscribeCall{
		Audio:     req.Audio,
		MediaType: req.MediaType,
		Language:  req.Language,
		Headers:   req.Headers.Clone(),
		Provider:  req.Provider.clone(),
	}
	s.f = flow[Segment]{ctx: ctx, telemetry: req.Telemetry, open: func(ctx context.Context) (iter.Seq2[Segment, error], error) {
		if m == nil {
			return nil, errors.New("ai: nil model")
		}
		if req.Audio == nil {
			return nil, errors.New("ai: transcribe: no audio")
		}
		resp, err := m.StreamTranscribe(ctx, call)
		if err != nil {
			return nil, err
		}
		if resp == nil {
			return nil, errors.New("ai: model returned no stream")
		}
		return resp.Segments, nil
	}}
	if m != nil {
		s.f.provider, s.f.modelID = m.Provider(), m.ModelID()
	}
	return s
}

func (s *TranscribeStream) Segments() iter.Seq[Segment] { return s.f.seq() }

func (s *TranscribeStream) Err() error { return s.f.fail() }
