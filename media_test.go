package ai_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type recordingTelemetry struct {
	started  []ai.StepInfo
	finished []error
	steps    []ai.Step
}

func (r *recordingTelemetry) StartStep(ctx context.Context, info ai.StepInfo) context.Context {
	r.started = append(r.started, info)
	return ctx
}

func (r *recordingTelemetry) FinishStep(_ context.Context, _ ai.StepInfo, step ai.Step, err error) {
	r.steps = append(r.steps, step)
	r.finished = append(r.finished, err)
}

func (r *recordingTelemetry) StartTool(ctx context.Context, _ ai.ToolCall) context.Context {
	return ctx
}

func (r *recordingTelemetry) FinishTool(context.Context, ai.ToolCall, ai.ToolResult, time.Duration, error) {
}

func TestGenerateImage(t *testing.T) {
	m := aitest.NewImageModel()
	tel := &recordingTelemetry{}
	mask := ai.File{MediaType: "image/png", Data: []byte("m")}
	res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
		Prompt: "a fox", N: 2, Size: "1024x1024", Seed: new(7), Mask: &mask,
		Files: []ai.File{{Data: []byte("in")}}, Headers: map[string][]string{"X-A": {"b"}}, Telemetry: tel,
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(res.Images) != 2 || res.Usage.Input != len("a fox") {
		t.Errorf("GenerateImage() = %+v, want 2 images and usage", res)
	}
	calls := m.Calls()
	if len(calls) != 1 || calls[0].Prompt != "a fox" || calls[0].N != 2 || *calls[0].Seed != 7 || string(calls[0].Mask.Data) != "m" || calls[0].Headers.Get("X-A") != "b" {
		t.Errorf("call = %+v, want request fields forwarded", calls)
	}
	if len(tel.started) != 1 || tel.started[0].Provider != "aitest" || len(tel.finished) != 1 || tel.finished[0] != nil {
		t.Errorf("telemetry = %+v, want one finished step", tel)
	}
}

func TestGenerateImage_requestIsNotMutated(t *testing.T) {
	m := aitest.NewImageModel()
	files := []ai.File{{Data: []byte("a")}}
	seed := 1
	req := ai.ImageRequest{Prompt: "p", Files: files, Seed: &seed}
	if _, err := ai.GenerateImage(t.Context(), m, req); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	*m.Calls()[0].Seed = 99
	m.Calls()[0].Files[0].Data = nil
	if seed != 1 || files[0].Data == nil {
		t.Errorf("request mutated through call: seed = %d, files = %v", seed, files)
	}
}

func TestGenerateImage_validation(t *testing.T) {
	tests := []struct {
		name  string
		model ai.ImageModel
		req   ai.ImageRequest
		want  string
	}{
		{name: "empty prompt", model: aitest.NewImageModel(), req: ai.ImageRequest{}, want: "empty prompt"},
		{name: "negative n", model: aitest.NewImageModel(), req: ai.ImageRequest{Prompt: "p", N: -1}, want: "negative N"},
		{name: "nil model", req: ai.ImageRequest{Prompt: "p"}, want: "nil model"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateImage(t.Context(), tt.model, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestGenerateImage_retries(t *testing.T) {
	tests := []struct {
		name         string
		errs         []error
		maxRetries   *int
		wantAttempts int
		wantErr      bool
	}{
		{name: "retries 503 twice by default", errs: []error{apiErr(503), apiErr(503)}, wantAttempts: 3},
		{name: "gives up after default retries", errs: []error{apiErr(503), apiErr(503), apiErr(503)}, wantAttempts: 3, wantErr: true},
		{name: "does not retry 400", errs: []error{apiErr(400)}, wantAttempts: 1, wantErr: true},
		{name: "zero retries", errs: []error{apiErr(503)}, maxRetries: new(0), wantAttempts: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewImageModel(aitest.Fail(tt.errs...))
				_, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{Prompt: "p", MaxRetries: tt.maxRetries})
				if (err != nil) != tt.wantErr || len(m.Calls()) != tt.wantAttempts {
					t.Errorf("GenerateImage() error = %v after %d attempts, want error=%v after %d", err, len(m.Calls()), tt.wantErr, tt.wantAttempts)
				}
			})
		})
	}
}

func TestGenerateImage_telemetryRecordsFailure(t *testing.T) {
	tel := &recordingTelemetry{}
	_, err := ai.GenerateImage(t.Context(), aitest.NewImageModel(aitest.Fail(apiErr(400))), ai.ImageRequest{Prompt: "p", Telemetry: tel})
	if err == nil || len(tel.finished) != 1 || tel.finished[0] == nil {
		t.Errorf("GenerateImage() error = %v, telemetry = %+v, want a recorded failure", err, tel)
	}
}

func TestGenerateVideo_pollsWithBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		clip := ai.File{MediaType: "video/mp4", Data: []byte("mp4")}
		m := aitest.NewVideoModel(aitest.VideoProgress(10), aitest.VideoProgress(50), aitest.VideoProgress(80), aitest.VideoDone(clip))
		start := time.Now()
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "waves", PollInterval: 2 * time.Second, Duration: 8 * time.Second})
		if err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		if res.ID != "video_1" || len(res.Videos) != 1 || string(res.Videos[0].Data) != "mp4" {
			t.Errorf("GenerateVideo() = %+v, want video_1 with the clip", res)
		}
		var gaps []time.Duration
		times := m.PollTimes()
		for i := 1; i < len(times); i++ {
			gaps = append(gaps, times[i].Sub(times[i-1]))
		}
		want := []time.Duration{2 * time.Second, 3 * time.Second, 4500 * time.Millisecond}
		if diff := cmp.Diff(want, gaps); diff != "" {
			t.Errorf("poll gaps mismatch (-want +got):\n%s", diff)
		}
		if got := time.Since(start); got != 9500*time.Millisecond {
			t.Errorf("total wait = %v, want 9.5s", got)
		}
		if m.Started()[0].Duration != 8*time.Second {
			t.Errorf("started call = %+v, want the 8s duration", m.Started()[0])
		}
	})
}

func TestGenerateVideo_intervalIsCapped(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		steps := make([]aitest.VideoOption, 12)
		for i := range steps {
			steps[i] = aitest.VideoProgress(i)
		}
		steps = append(steps, aitest.VideoDone(ai.File{Data: []byte("v")}))
		m := aitest.NewVideoModel(steps...)
		if _, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "p"}); err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		times := m.PollTimes()
		for i := 1; i < len(times); i++ {
			if gap := times[i].Sub(times[i-1]); gap > 30*time.Second {
				t.Errorf("gap %d = %v, want at most 30s", i, gap)
			}
		}
		if last := times[len(times)-1].Sub(times[len(times)-2]); last != 30*time.Second {
			t.Errorf("last gap = %v, want the 30s cap", last)
		}
	})
}

func TestGenerateVideo_providerHintWins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewVideoModel(aitest.VideoPollHint(aitest.VideoProgress(1), 7*time.Second), aitest.VideoDone(ai.File{Data: []byte("v")}))
		if _, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "p", PollInterval: time.Second}); err != nil {
			t.Fatalf("GenerateVideo() error = %v", err)
		}
		times := m.PollTimes()
		if gap := times[1].Sub(times[0]); gap != 7*time.Second {
			t.Errorf("gap = %v, want the 7s hint", gap)
		}
	})
}

func TestGenerateVideo_contextBoundsPolling(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		steps := make([]aitest.VideoOption, 100)
		for i := range steps {
			steps[i] = aitest.VideoProgress(1)
		}
		ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
		defer cancel()
		_, err := ai.GenerateVideo(ctx, aitest.NewVideoModel(steps...), ai.VideoRequest{Prompt: "p"})
		jobErr, ok := errors.AsType[*ai.JobError](err)
		if !ok || jobErr.ID != "video_1" || jobErr.Kind != "video" || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("GenerateVideo() error = %v, want a JobError for video_1 wrapping the deadline", err)
		}
	})
}

func TestGenerateVideo_failures(t *testing.T) {
	boom := errors.New("content policy")
	tests := []struct {
		name  string
		model *aitest.VideoModel
		want  error
	}{
		{name: "job failed", model: aitest.NewVideoModel(aitest.VideoProgress(5), aitest.VideoFailed(boom)), want: boom},
		{name: "poll fails permanently", model: aitest.NewVideoModel(aitest.VideoPollError(apiErr(400))), want: nil},
		{name: "done without video", model: aitest.NewVideoModel(aitest.VideoDone()), want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				_, err := ai.GenerateVideo(t.Context(), tt.model, ai.VideoRequest{Prompt: "p"})
				jobErr, ok := errors.AsType[*ai.JobError](err)
				if !ok || jobErr.ID != "video_1" {
					t.Fatalf("GenerateVideo() error = %v, want a JobError for video_1", err)
				}
				if tt.want != nil && !errors.Is(err, tt.want) {
					t.Errorf("GenerateVideo() error = %v, want it to wrap %v", err, tt.want)
				}
			})
		})
	}
}

func TestGenerateVideo_retriesTransientErrors(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewVideoModel(aitest.VideoPollError(apiErr(503)), aitest.VideoDone(ai.File{Data: []byte("v")}), aitest.Fail(apiErr(429)))
		res, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "p"})
		if err != nil || len(res.Videos) != 1 {
			t.Fatalf("GenerateVideo() = %+v, %v, want a video after transient errors", res, err)
		}
		if len(m.Started()) != 2 {
			t.Errorf("StartVideo calls = %d, want 2", len(m.Started()))
		}
	})
}

func TestGenerateVideo_startFailureHasNoJob(t *testing.T) {
	m := aitest.NewVideoModel(aitest.Fail(apiErr(400)))
	_, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{Prompt: "p"})
	if _, isJob := errors.AsType[*ai.JobError](err); err == nil || isJob {
		t.Errorf("GenerateVideo() error = %v, want the plain start error", err)
	}
	if _, err := ai.GenerateVideo(t.Context(), m, ai.VideoRequest{}); err == nil {
		t.Error("GenerateVideo() with empty prompt succeeded")
	}
}

func TestGenerateSpeech(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewSpeechModel(aitest.Fail(apiErr(503)))
		tel := &recordingTelemetry{}
		res, err := ai.GenerateSpeech(t.Context(), m, ai.SpeechRequest{Text: "hello", Voice: "marin", Format: "wav", Instructions: "calm", Speed: new(1.25), Telemetry: tel})
		if err != nil {
			t.Fatalf("GenerateSpeech() error = %v", err)
		}
		if string(res.Audio.Data) != "hello" || res.Audio.MediaType != "audio/mpeg" {
			t.Errorf("audio = %+v, want the echoed text", res.Audio)
		}
		calls := m.Calls()
		if len(calls) != 2 || calls[1].Voice != "marin" || calls[1].Format != "wav" || calls[1].Instructions != "calm" || *calls[1].Speed != 1.25 {
			t.Errorf("calls = %+v, want two with every field", calls)
		}
		if len(tel.finished) != 1 {
			t.Errorf("telemetry finished = %d, want 1 for the whole call", len(tel.finished))
		}
	})
	if _, err := ai.GenerateSpeech(t.Context(), aitest.NewSpeechModel(), ai.SpeechRequest{}); err == nil {
		t.Error("GenerateSpeech() with empty text succeeded")
	}
}

func TestStreamSpeech(t *testing.T) {
	m := aitest.NewSpeechModel()
	text := func(yield func(string) bool) {
		for _, s := range []string{"Hello ", "there."} {
			if !yield(s) {
				return
			}
		}
	}
	s := ai.StreamSpeech(t.Context(), m, ai.SpeechStreamRequest{Text: text, Voice: "alloy"})
	var got []string
	for chunk := range s.Audio() {
		got = append(got, string(chunk))
	}
	if err := s.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if diff := cmp.Diff([]string{"Hello ", "there."}, got); diff != "" {
		t.Errorf("audio mismatch (-want +got):\n%s", diff)
	}
	if s.MediaType() != "audio/mpeg" {
		t.Errorf("MediaType() = %q, want audio/mpeg", s.MediaType())
	}
}

func TestStreamSpeech_isLazyAndSingleUse(t *testing.T) {
	m := aitest.NewSpeechModel()
	pulled := false
	text := func(yield func(string) bool) {
		pulled = true
		yield("x")
	}
	s := ai.StreamSpeech(t.Context(), m, ai.SpeechStreamRequest{Text: text})
	if pulled {
		t.Fatal("StreamSpeech() read the text before Audio was ranged")
	}
	for range s.Audio() {
	}
	for range s.Audio() {
		t.Error("second Audio() loop yielded a chunk")
	}
	if !errors.Is(s.Err(), ai.ErrStreamConsumed) {
		t.Errorf("Err() = %v, want ErrStreamConsumed", s.Err())
	}
}

func TestStreamSpeech_earlyStopIsNotAnError(t *testing.T) {
	m := aitest.NewSpeechModel()
	text := func(yield func(string) bool) {
		for range 10 {
			if !yield("x") {
				return
			}
		}
	}
	s := ai.StreamSpeech(t.Context(), m, ai.SpeechStreamRequest{Text: text})
	for range s.Audio() {
		break
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil after an early stop", err)
	}
	if got := len(m.StreamedText()); got != 1 {
		t.Errorf("model consumed %d text pieces, want 1", got)
	}
}

func TestStreamSpeech_errors(t *testing.T) {
	text := func(yield func(string) bool) { yield("x") }
	tests := []struct {
		name string
		m    ai.SpeechStreamModel
		req  ai.SpeechStreamRequest
		want string
	}{
		{name: "no text", m: aitest.NewSpeechModel(), req: ai.SpeechStreamRequest{}, want: "no text"},
		{name: "nil model", req: ai.SpeechStreamRequest{Text: text}, want: "nil model"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := ai.StreamSpeech(t.Context(), tt.m, tt.req)
			for range s.Audio() {
			}
			if err := s.Err(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Err() = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestStreamSpeech_telemetryOnce(t *testing.T) {
	tel := &recordingTelemetry{}
	s := ai.StreamSpeech(t.Context(), aitest.NewSpeechModel(), ai.SpeechStreamRequest{Text: func(yield func(string) bool) { yield("x") }, Telemetry: tel})
	for range s.Audio() {
	}
	if len(tel.started) != 1 || len(tel.finished) != 1 {
		t.Errorf("telemetry = %+v, want one start and one finish", tel)
	}
}

func TestTranscribe(t *testing.T) {
	m := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "hi", End: time.Second}, ai.Segment{Text: "there", Start: time.Second, End: 2 * time.Second}))
	res, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{Audio: strings.NewReader("pcm"), MediaType: "audio/wav", Language: "en"})
	if err != nil {
		t.Fatalf("Transcribe() error = %v", err)
	}
	if res.Text != "hi there" || len(res.Segments) != 2 || res.Language != "en" {
		t.Errorf("Transcribe() = %+v, want joined text, 2 segments, en", res)
	}
	if _, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{}); err == nil {
		t.Error("Transcribe() without audio succeeded")
	}
}

func TestTranscribe_retriesOnlyRewindableAudio(t *testing.T) {
	tests := []struct {
		name         string
		audio        io.Reader
		wantAttempts int
		wantErr      bool
	}{
		{name: "seekable audio is rewound", audio: bytes.NewReader([]byte("abc")), wantAttempts: 2},
		{name: "plain reader is not retried", audio: io.MultiReader(strings.NewReader("abc")), wantAttempts: 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "ok"}), aitest.Fail(apiErr(503)))
				_, err := ai.Transcribe(t.Context(), m, ai.TranscribeRequest{Audio: tt.audio})
				if (err != nil) != tt.wantErr || len(m.Audio()) != tt.wantAttempts {
					t.Errorf("Transcribe() error = %v after %d reads, want error=%v after %d", err, len(m.Audio()), tt.wantErr, tt.wantAttempts)
				}
				for i, a := range m.Audio() {
					if string(a) != "abc" {
						t.Errorf("attempt %d read %q, want the full audio", i, a)
					}
				}
			})
		})
	}
}

func TestStreamTranscribe(t *testing.T) {
	m := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "he", Partial: true}, ai.Segment{Text: "hello"}))
	s := ai.StreamTranscribe(t.Context(), m, ai.TranscribeStreamRequest{Audio: strings.NewReader("x")})
	var got []ai.Segment
	for seg := range s.Segments() {
		got = append(got, seg)
	}
	if err := s.Err(); err != nil || len(got) != 2 || !got[0].Partial || got[1].Partial {
		t.Errorf("Segments() = %+v, Err() = %v, want a partial then a final segment", got, err)
	}
	for range s.Segments() {
		t.Error("second Segments() loop yielded")
	}
	if !errors.Is(s.Err(), ai.ErrStreamConsumed) {
		t.Errorf("Err() = %v, want ErrStreamConsumed", s.Err())
	}
}

func TestStreamTranscribe_errors(t *testing.T) {
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		m    ai.TranscriptionStreamModel
		req  ai.TranscribeStreamRequest
		want string
	}{
		{name: "no audio", ctx: t.Context(), m: aitest.NewTranscriptionModel(), want: "no audio"},
		{name: "nil model", ctx: t.Context(), req: ai.TranscribeStreamRequest{Audio: strings.NewReader("x")}, want: "nil model"},
		{name: "canceled", ctx: canceled, m: aitest.NewTranscriptionModel(), req: ai.TranscribeStreamRequest{Audio: strings.NewReader("x")}, want: "canceled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := ai.StreamTranscribe(tt.ctx, tt.m, tt.req)
			for range s.Segments() {
			}
			if err := s.Err(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Err() = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestImageCall_cloneCopiesMask(t *testing.T) {
	c := &ai.ImageCall{Mask: &ai.File{Filename: "m"}}
	got := c.Clone()
	got.Mask.Filename = "changed"
	if c.Mask.Filename != "m" {
		t.Error("Clone() shares the mask")
	}
	wrapped := ai.WrapImageModel(aitest.NewImageModel(), ai.ImageMiddleware{TransformRequest: func(_ context.Context, c *ai.ImageCall) error {
		c.Mask = nil
		return nil
	}})
	in := &ai.ImageCall{Prompt: "p", Mask: &ai.File{}}
	if _, err := wrapped.GenerateImage(t.Context(), in); err != nil || in.Mask == nil {
		t.Errorf("middleware changed the caller's call: err = %v, mask = %v", err, in.Mask)
	}
}
