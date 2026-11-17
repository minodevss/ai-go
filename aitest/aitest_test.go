package aitest_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/realtime"
)

func TestModel_generateAndStreamAgree(t *testing.T) {
	script := func() *aitest.Model {
		return aitest.NewModel(aitest.Turn(aitest.Reasoning("r"), aitest.TextChunks("a", "b"), aitest.ToolCall("t", `{"x":1}`), aitest.Usage(2, 3)))
	}
	gen, err := script().Generate(t.Context(), &ai.Call{})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	stream, err := script().Stream(t.Context(), &ai.Call{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var types []string
	var finish ai.StepEnd
	for p, perr := range stream.Parts {
		if perr != nil {
			t.Fatalf("stream error = %v", perr)
		}
		if end, ok := p.(ai.StepEnd); ok {
			finish = end
		}
		types = append(types, typeName(p))
	}
	if gen.FinishReason != ai.FinishToolCalls || finish.FinishReason != ai.FinishToolCalls {
		t.Errorf("finish reasons = %q / %q, want tool-calls", gen.FinishReason, finish.FinishReason)
	}
	if !cmp.Equal(gen.Usage, finish.Usage) || gen.Usage.Input != 2 || gen.Usage.Output != 3 {
		t.Errorf("usage = %+v / %+v, want 2/3", gen.Usage, finish.Usage)
	}
	if len(gen.Content) != 3 || types[len(types)-1] != "StepEnd" {
		t.Errorf("content = %v, stream = %v", gen.Content, types)
	}
}

func typeName(p ai.StreamPart) string {
	switch p.(type) {
	case ai.StepEnd:
		return "StepEnd"
	default:
		return "other"
	}
}

func TestModel_exhaustedScript(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("only")))
	if _, err := m.Generate(t.Context(), &ai.Call{}); err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	if _, err := m.Generate(t.Context(), &ai.Call{}); err == nil {
		t.Error("second Generate() error = nil, want an exhausted script error")
	}
	if _, err := m.Stream(t.Context(), &ai.Call{}); err == nil {
		t.Error("Stream() error = nil, want an exhausted script error")
	}
}

func TestModel_callsAreRecordedAsClones(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("a")), aitest.Turn(aitest.Text("b")))
	call := &ai.Call{Messages: []ai.Message{ai.User("first")}}
	if _, err := m.Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	call.Messages[0] = ai.User("mutated after the call")
	if _, err := m.Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	calls := m.Calls()
	if len(calls) != 2 || !cmp.Equal(calls[0].Messages[0], ai.User("first")) || !cmp.Equal(calls[1].Messages[0], ai.User("mutated after the call")) {
		t.Errorf("recorded calls = %+v", calls)
	}
}

func TestModel_errBeforeOutputFailsTheCall(t *testing.T) {
	boom := errors.New("boom")
	m := aitest.NewModel(aitest.Turn(aitest.Err(boom)), aitest.Turn(aitest.Text("x"), aitest.Err(boom)))

	if _, err := m.Stream(t.Context(), &ai.Call{}); !errors.Is(err, boom) {
		t.Errorf("Stream() error = %v, want %v", err, boom)
	}
	resp, err := m.Stream(t.Context(), &ai.Call{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var got error
	for _, perr := range resp.Parts {
		if perr != nil {
			got = perr
		}
	}
	if !errors.Is(got, boom) {
		t.Errorf("mid-stream error = %v, want %v", got, boom)
	}
}

func TestModel_delayHonorsContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		defer cancel()
		m := aitest.NewModel(aitest.Turn(aitest.Delay(time.Minute), aitest.Text("late")))
		start := time.Now()
		_, err := m.Generate(ctx, &ai.Call{})
		if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) != time.Second {
			t.Errorf("Generate() error = %v after %v, want a deadline error after 1s", err, time.Since(start))
		}
	})
}

func TestEmbeddingModel_defaultsAndFunc(t *testing.T) {
	def := aitest.NewEmbeddingModel()
	resp, err := def.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "abc"}}, {ai.Text{Text: "a"}, ai.Text{Text: "b"}}}})
	if err != nil || resp.Embeddings[0][0] != 3 || resp.Embeddings[1][0] != 2 {
		t.Errorf("Embed() = %+v, %v", resp, err)
	}
	custom := aitest.NewEmbeddingModel(aitest.EmbedFunc(func(string) []float32 { return []float32{9, 9} }), aitest.MaxBatch(5))
	resp, err = custom.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "x"}}}})
	if err != nil || len(resp.Embeddings[0]) != 2 || custom.MaxBatch() != 5 {
		t.Errorf("custom Embed() = %+v, %v, MaxBatch %d", resp, err, custom.MaxBatch())
	}
}

func TestFail(t *testing.T) {
	boom := errors.New("boom")
	ctx := t.Context()
	tests := []struct {
		name string
		call func(errs ...error) []error
	}{
		{name: "model", call: func(errs ...error) []error {
			m := aitest.NewModel(aitest.Turn(aitest.Text("ok")), aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.Generate(ctx, &ai.Call{}); return err })
		}},
		{name: "batch", call: func(errs ...error) []error {
			m := aitest.NewBatchModel(aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.StartBatch(ctx, nil); return err })
		}},
		{name: "embedding", call: func(errs ...error) []error {
			m := aitest.NewEmbeddingModel(aitest.MaxBatch(2), aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.Embed(ctx, &ai.EmbedCall{}); return err })
		}},
		{name: "rerank", call: func(errs ...error) []error {
			m := aitest.NewRerankModel(aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.Rerank(ctx, &ai.RerankCall{}); return err })
		}},
		{name: "image", call: func(errs ...error) []error {
			m := aitest.NewImageModel(aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.GenerateImage(ctx, &ai.ImageCall{}); return err })
		}},
		{name: "video", call: func(errs ...error) []error {
			m := aitest.NewVideoModel(aitest.VideoDone(), aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.StartVideo(ctx, &ai.VideoCall{}); return err })
		}},
		{name: "speech", call: func(errs ...error) []error {
			m := aitest.NewSpeechModel(aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.GenerateSpeech(ctx, &ai.SpeechCall{}); return err })
		}},
		{name: "transcription", call: func(errs ...error) []error {
			m := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "x"}), aitest.Fail(errs...))
			return runTwice(func() error {
				_, err := m.Transcribe(ctx, &ai.TranscribeCall{Audio: strings.NewReader("a")})
				return err
			})
		}},
		{name: "realtime", call: func(errs ...error) []error {
			m := aitest.NewRealtimeModel(aitest.Fail(errs...))
			return runTwice(func() error { _, err := m.Connect(ctx, &realtime.Config{}); return err })
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.call(boom)
			if !errors.Is(got[0], boom) || got[1] != nil {
				t.Errorf("calls with Fail(boom) = %v, want [boom <nil>]", got)
			}
			if got := tt.call(); got[0] != nil {
				t.Errorf("first call with Fail() = %v, want no error", got[0])
			}
		})
	}
}

func runTwice(f func() error) []error { return []error{f(), f()} }
