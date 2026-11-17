package ai_test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type scriptedResponses struct {
	responses []*ai.Response
	next      int
}

func (*scriptedResponses) Provider() string { return "test" }

func (*scriptedResponses) ModelID() string { return "responses" }

func (s *scriptedResponses) Generate(context.Context, *ai.Call) (*ai.Response, error) {
	r := s.responses[s.next]
	s.next++
	return r, nil
}

func (*scriptedResponses) Stream(context.Context, *ai.Call) (*ai.StreamResponse, error) {
	return nil, errors.New("unused")
}

func TestGenerateText_keepsProviderDataLossless(t *testing.T) {
	container := ai.ProviderData{}.WithRaw("anthropic", rawJSON(`{"container":"c9"}`))
	header := http.Header{"X-Request-Id": {"r1"}}
	m := &scriptedResponses{responses: []*ai.Response{
		{
			Content:      ai.Parts{ai.Reasoning{Provider: ai.ProviderData{}.WithRaw("anthropic", rawJSON(`{"signature":"s"}`))}, ai.ToolCall{ID: "c1", Name: "search", Input: rawJSON(`{"query":"q"}`)}},
			Usage:        ai.Usage{Input: 1, Output: 2, Raw: rawJSON(`{"a":1}`)},
			FinishReason: ai.FinishToolCalls,
			FinishRaw:    "tool_use",
			Warnings:     []ai.Warning{{Message: "w1"}},
			Provider:     container,
			Header:       header,
		},
		{
			Content:      ai.Parts{ai.Text{Text: "done"}},
			Usage:        ai.Usage{Input: 3, Output: 4, CacheRead: 5, Raw: rawJSON(`{"a":2}`)},
			FinishReason: ai.FinishStop,
			FinishRaw:    "end_turn",
			Warnings:     []ai.Warning{{Message: "w2"}},
		},
	}}
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})

	if diff := cmp.Diff(container, res.Messages[0].Provider); diff != "" {
		t.Errorf("assistant message Provider mismatch (-want +got):\n%s", diff)
	}
	if r, ok := res.Messages[0].Content[0].(ai.Reasoning); !ok || len(r.Provider) == 0 {
		t.Error("signature-carrying empty reasoning part was dropped")
	}
	if diff := cmp.Diff([]ai.Warning{{Message: "w1"}, {Message: "w2"}}, res.Warnings); diff != "" {
		t.Errorf("Warnings mismatch (-want +got):\n%s", diff)
	}
	if res.FinishRaw != "end_turn" || res.FinishReason != ai.FinishStop {
		t.Errorf("finish = %q (%q), want stop (end_turn)", res.FinishReason, res.FinishRaw)
	}
	if got, want := res.Usage.Input+res.Usage.Output+res.Usage.CacheRead, 15; got != want {
		t.Errorf("summed usage = %d, want %d", got, want)
	}
	if got := res.Steps[0].Header.Get("X-Request-Id"); got != "r1" {
		t.Errorf("step header lost: %q", got)
	}
}

func TestStreamText_emitsToolFailedWithRealError(t *testing.T) {
	secret := errors.New("db password is hunter2")
	tool := ai.NewTool("t", "d", func(context.Context, struct{}) (int, error) { return 0, secret })
	m := aitest.NewModel(toolTurn("t", `{}`), aitest.Turn(aitest.Text("ok")))

	var failed []ai.ToolFailed
	var modelSaw string
	s := ai.StreamText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{tool}})
	for p := range s.Parts() {
		switch p := p.(type) {
		case ai.ToolFailed:
			failed = append(failed, p)
		case ai.ToolResult:
			modelSaw = p.Output.Text
		}
	}
	if len(failed) != 1 || !errors.Is(failed[0].Err, secret) || failed[0].Name != "t" {
		t.Errorf("ToolFailed parts = %+v, want one carrying the real error", failed)
	}
	if modelSaw != "tool failed" {
		t.Errorf("ToolResult text = %q, want the generic message", modelSaw)
	}
}

func TestStreamText_earlyBreakLeaksNoGoroutines(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewModel(aitest.Turn(aitest.TextChunks("a", "b", "c"), aitest.Delay(time.Hour), aitest.Text("never")))
		s := ai.StreamText(t.Context(), m, ai.TextRequest{Timeouts: ai.Timeouts{Chunk: time.Minute, Step: time.Hour}})
		for range s.Parts() {
			break
		}
		synctest.Wait()
	})
}

func TestStreamText_noGoroutinesAfterCompletionOrError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ok := aitest.NewModel(aitest.Turn(aitest.Text("fine")))
		if _, err := ai.StreamText(t.Context(), ok, ai.TextRequest{Timeouts: ai.Timeouts{FirstChunk: time.Second, Chunk: time.Second}}).Result(); err != nil {
			t.Fatalf("Result() error = %v", err)
		}
		bad := aitest.NewModel(aitest.Turn(aitest.Text("x"), aitest.Err(errors.New("boom"))))
		_, _ = ai.StreamText(t.Context(), bad, ai.TextRequest{}).Result()
		synctest.Wait()
	})
}

func TestGenerateText_toolsRunConcurrentlyWithinStepOnly(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		sleepy := ai.NewTool("sleepy", "d", func(context.Context, struct{}) (int, error) {
			time.Sleep(time.Second)
			return 1, nil
		})
		m := aitest.NewModel(
			aitest.Turn(aitest.ToolCall("sleepy", `{}`), aitest.ToolCall("sleepy", `{}`), aitest.ToolCall("sleepy", `{}`)),
			aitest.Turn(aitest.Text("done")),
		)
		start := time.Now()
		mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{sleepy}})
		if got := time.Since(start); got != time.Second {
			t.Errorf("three parallel 1s tools took %v, want 1s", got)
		}
	})
}

func TestGenerateText_telemetryOnError(t *testing.T) {
	rec := &recorder{}
	boom := errors.New("boom")
	m := aitest.NewModel(aitest.Turn(aitest.Err(boom)))
	_, err := ai.GenerateText(t.Context(), m, ai.TextRequest{Telemetry: rec, MaxRetries: new(0)})
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v", err)
	}
	if diff := cmp.Diff([]string{"step-start:0", "step-end:0"}, rec.events); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateText_nilModel(t *testing.T) {
	if _, err := ai.GenerateText(t.Context(), nil, ai.TextRequest{}); err == nil {
		t.Error("GenerateText(nil model) error = nil, want an error")
	}
}

func TestGenerateText_lengthFinishStopsTheLoop(t *testing.T) {
	m := aitest.NewModel(
		aitest.Turn(aitest.ToolCall("search", `{"query":"q"}`), aitest.Finish(ai.FinishLength)),
		aitest.Turn(aitest.Text("never")),
	)
	res := mustGenerate(t, m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})
	if len(res.Steps) != 1 || res.FinishReason != ai.FinishLength {
		t.Errorf("steps = %d, finish = %q; want 1 step ending with length", len(res.Steps), res.FinishReason)
	}
	if got := len(resultsOf(res.Messages)); got != 1 {
		t.Errorf("results = %d, want 1 (the call still gets an answer)", got)
	}
}

func TestGenerateText_emptyStepAddsNoMessage(t *testing.T) {
	res := mustGenerate(t, aitest.NewModel(aitest.Turn()), ai.TextRequest{})
	if len(res.Messages) != 0 || res.Text != "" {
		t.Errorf("Messages = %v, Text = %q; want none", res.Messages, res.Text)
	}
}

func TestGenerateText_toolChoiceAndSettingsReachModel(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, m, ai.TextRequest{
		ToolChoice:       ai.ToolNamed("search"),
		Tools:            []ai.Tool{searchTool()},
		MaxOutputTokens:  99,
		Temperature:      new(0.7),
		TopP:             new(0.8),
		TopK:             5,
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.2),
		Seed:             new(3),
		StopSequences:    []string{"STOP"},
		Reasoning:        ai.ReasoningHigh,
		Headers:          http.Header{"X-A": {"1"}},
		Provider:         ai.ProviderData{}.WithRaw("p", rawJSON(`1`)),
	})
	c := m.Calls()[0]
	if c.ToolChoice.Mode() != ai.ToolChoiceNamed || c.ToolChoice.Name() != "search" || c.MaxOutputTokens != 99 || *c.Temperature != 0.7 || *c.TopP != 0.8 || c.TopK != 5 ||
		*c.PresencePenalty != 0.1 || *c.FrequencyPenalty != 0.2 || *c.Seed != 3 || c.StopSequences[0] != "STOP" || c.Reasoning != ai.ReasoningHigh || c.Headers.Get("X-A") != "1" || len(c.Provider) != 1 {
		t.Errorf("call = %+v, want every request setting forwarded", c)
	}
}
