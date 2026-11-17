package aitest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/minodevss/ai-go"
)

// ImageModel returns one PNG-signature file per requested image.
type ImageModel struct {
	mu       sync.Mutex
	calls    []ai.ImageCall
	failures failures
}

func NewImageModel(opts ...ImageOption) *ImageModel {
	m := &ImageModel{}
	for _, opt := range opts {
		opt.applyImage(m)
	}
	return m
}

func (m *ImageModel) Provider() string { return "aitest" }

func (m *ImageModel) ModelID() string { return "scripted-image" }

func (m *ImageModel) Calls() []ai.ImageCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.ImageCall(nil), m.calls...)
}

func (m *ImageModel) GenerateImage(ctx context.Context, c *ai.ImageCall) (*ai.ImageResponse, error) {
	m.mu.Lock()
	m.calls = append(m.calls, *c)
	failure := m.failures.next()
	m.mu.Unlock()
	if failure != nil {
		return nil, failure
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	images := make([]ai.File, max(c.N, 1))
	for i := range images {
		images[i] = ai.File{MediaType: "image/png", Data: []byte("\x89PNG\r\n\x1a\n" + c.Prompt)}
	}
	return &ai.ImageResponse{Images: images, Usage: ai.Usage{Input: len(c.Prompt), Output: len(images)}}, nil
}

type VideoStep struct {
	poll ai.VideoPoll
	err  error
}

func (s VideoStep) applyVideo(m *VideoModel) { m.steps = append(m.steps, s) }

func VideoProgress(percent int) VideoStep { return VideoStep{poll: ai.VideoPoll{Progress: percent}} }

func VideoDone(files ...ai.File) VideoStep {
	return VideoStep{poll: ai.VideoPoll{Done: true, Progress: 100, Videos: files}}
}

func VideoFailed(err error) VideoStep { return VideoStep{poll: ai.VideoPoll{Failure: err}} }

func VideoPollError(err error) VideoStep { return VideoStep{err: err} }

func VideoPollHint(step VideoStep, d time.Duration) VideoStep {
	step.poll.Interval = d
	return step
}

// VideoModel plays its steps in order, one per poll, and records the time of every poll.
type VideoModel struct {
	mu        sync.Mutex
	steps     []VideoStep
	started   []ai.VideoCall
	pollTimes []time.Time
	polls     int
	failures  failures
}

func NewVideoModel(opts ...VideoOption) *VideoModel {
	m := &VideoModel{}
	for _, opt := range opts {
		opt.applyVideo(m)
	}
	return m
}

func (m *VideoModel) Provider() string { return "aitest" }

func (m *VideoModel) ModelID() string { return "scripted-video" }

func (m *VideoModel) Started() []ai.VideoCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.VideoCall(nil), m.started...)
}

func (m *VideoModel) PollTimes() []time.Time {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]time.Time(nil), m.pollTimes...)
}

func (m *VideoModel) StartVideo(_ context.Context, c *ai.VideoCall) (*ai.VideoJob, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.started = append(m.started, *c)
	if err := m.failures.next(); err != nil {
		return nil, err
	}
	return &ai.VideoJob{ID: fmt.Sprintf("video_%d", len(m.started))}, nil
}

func (m *VideoModel) PollVideo(ctx context.Context, _ *ai.VideoJob) (*ai.VideoPoll, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pollTimes = append(m.pollTimes, time.Now())
	if m.polls >= len(m.steps) {
		return nil, fmt.Errorf("aitest: no scripted poll %d", m.polls+1)
	}
	step := m.steps[m.polls]
	m.polls++
	if step.err != nil {
		return nil, step.err
	}
	poll := step.poll
	return &poll, nil
}

// SpeechModel answers with the UTF-8 bytes of the text as "audio". Its stream yields one chunk per text piece it receives.
type SpeechModel struct {
	mu       sync.Mutex
	calls    []ai.SpeechCall
	streams  []string
	failures failures
}

func NewSpeechModel(opts ...SpeechOption) *SpeechModel {
	m := &SpeechModel{}
	for _, opt := range opts {
		opt.applySpeech(m)
	}
	return m
}

func (m *SpeechModel) Provider() string { return "aitest" }

func (m *SpeechModel) ModelID() string { return "scripted-speech" }

func (m *SpeechModel) Calls() []ai.SpeechCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ai.SpeechCall(nil), m.calls...)
}

func (m *SpeechModel) StreamedText() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.streams...)
}

func (m *SpeechModel) GenerateSpeech(ctx context.Context, c *ai.SpeechCall) (*ai.SpeechResponse, error) {
	m.mu.Lock()
	m.calls = append(m.calls, *c)
	failure := m.failures.next()
	m.mu.Unlock()
	if failure != nil {
		return nil, failure
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &ai.SpeechResponse{Audio: ai.File{MediaType: "audio/mpeg", Data: []byte(c.Text)}}, nil
}

func (m *SpeechModel) StreamSpeech(_ context.Context, c *ai.SpeechStreamCall) (*ai.SpeechStreamResponse, error) {
	audio := func(yield func([]byte, error) bool) {
		for piece := range c.Text {
			m.mu.Lock()
			m.streams = append(m.streams, piece)
			m.mu.Unlock()
			if !yield([]byte(piece), nil) {
				return
			}
		}
	}
	return &ai.SpeechStreamResponse{Audio: audio, MediaType: "audio/mpeg"}, nil
}

// TranscriptionModel returns the scripted text for the audio it reads, or the audio itself as text when no segments are scripted.
type TranscriptionModel struct {
	mu       sync.Mutex
	segments []ai.Segment
	audio    [][]byte
	failures failures
}

type transcriptionOptionFunc func(*TranscriptionModel)

func (f transcriptionOptionFunc) applyTranscription(m *TranscriptionModel) { f(m) }

func Segments(segs ...ai.Segment) TranscriptionOption {
	return transcriptionOptionFunc(func(m *TranscriptionModel) { m.segments = segs })
}

func NewTranscriptionModel(opts ...TranscriptionOption) *TranscriptionModel {
	m := &TranscriptionModel{}
	for _, opt := range opts {
		opt.applyTranscription(m)
	}
	return m
}

func (m *TranscriptionModel) Provider() string { return "aitest" }

func (m *TranscriptionModel) ModelID() string { return "scripted-transcription" }

func (m *TranscriptionModel) Audio() [][]byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([][]byte(nil), m.audio...)
}

func (m *TranscriptionModel) Transcribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeResponse, error) {
	data, err := m.read(ctx, c)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	failure := m.failures.next()
	m.mu.Unlock()
	if failure != nil {
		return nil, failure
	}
	segments := m.segments
	if len(segments) == 0 {
		segments = []ai.Segment{{Text: string(data)}}
	}
	var text []string
	for _, s := range segments {
		text = append(text, s.Text)
	}
	return &ai.TranscribeResponse{Text: strings.Join(text, " "), Segments: segments, Language: c.Language}, nil
}

func (m *TranscriptionModel) StreamTranscribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeStreamResponse, error) {
	data, err := m.read(ctx, c)
	if err != nil {
		return nil, err
	}
	segments := m.segments
	if len(segments) == 0 {
		segments = []ai.Segment{{Text: string(data)}}
	}
	seq := func(yield func(ai.Segment, error) bool) {
		for _, s := range segments {
			if err := ctx.Err(); err != nil {
				yield(ai.Segment{}, err)
				return
			}
			if !yield(s, nil) {
				return
			}
		}
	}
	return &ai.TranscribeStreamResponse{Segments: seq}, nil
}

func (m *TranscriptionModel) read(ctx context.Context, c *ai.TranscribeCall) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(c.Audio)
	if err != nil {
		return nil, fmt.Errorf("aitest: read audio: %w", err)
	}
	m.mu.Lock()
	m.audio = append(m.audio, bytes.Clone(data))
	m.mu.Unlock()
	return data, nil
}

var (
	_ ai.ImageModel               = (*ImageModel)(nil)
	_ ai.VideoModel               = (*VideoModel)(nil)
	_ ai.SpeechModel              = (*SpeechModel)(nil)
	_ ai.SpeechStreamModel        = (*SpeechModel)(nil)
	_ ai.TranscriptionModel       = (*TranscriptionModel)(nil)
	_ ai.TranscriptionStreamModel = (*TranscriptionModel)(nil)
)
