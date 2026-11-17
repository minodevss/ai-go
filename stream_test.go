package ai_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func partTypes(s *ai.TextStream) []string {
	var out []string
	for p := range s.Parts() {
		out = append(out, fmt.Sprintf("%T", p))
	}
	return out
}

func TestStreamText_partsOrder(t *testing.T) {
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.Text("done")))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})

	want := []string{
		"ai.StreamStart",
		"ai.StepStart", "ai.ToolInputStart", "ai.ToolInputDelta", "ai.ToolInputEnd", "ai.ToolCall", "ai.ToolResult", "ai.StepEnd",
		"ai.StepStart", "ai.TextStart", "ai.TextDelta", "ai.TextEnd", "ai.StepEnd",
		"ai.StreamFinish",
	}
	if diff := cmp.Diff(want, partTypes(s)); diff != "" {
		t.Errorf("part types mismatch (-want +got):\n%s", diff)
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v, want nil", err)
	}
}

func TestStreamText_text(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks("Hel", "lo ", "world")))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{})

	got := slices.Collect(s.Text())
	if diff := cmp.Diff([]string{"Hel", "lo ", "world"}, got); diff != "" {
		t.Errorf("Text() mismatch (-want +got):\n%s", diff)
	}
	res, err := s.Result()
	if err != nil || res.Text != "Hello world" {
		t.Errorf("Result() = %+v, %v; want Text %q", res, err, "Hello world")
	}
}

func TestStreamText_isLazy(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("x")))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{})
	if got := len(m.Calls()); got != 0 {
		t.Fatalf("model calls before consumption = %d, want 0", got)
	}
	if _, err := s.Result(); err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if got := len(m.Calls()); got != 1 {
		t.Errorf("model calls after Result() = %d, want 1", got)
	}
}

func TestStreamText_snapshotsRequestAtCreation(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text("x")))
	req := ai.TextRequest{Messages: []ai.Message{ai.User("original")}}
	s := ai.StreamText(t.Context(), m, req)
	req.Messages[0] = ai.User("mutated")
	if _, err := s.Result(); err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if got := m.Calls()[0].Messages[0]; !cmp.Equal(got, ai.User("original")) {
		t.Errorf("model saw %+v, want the original message", got)
	}
}

func TestStreamText_singleUse(t *testing.T) {
	tests := []struct {
		name    string
		consume func(s *ai.TextStream) error
		want    error
	}{
		{
			name: "second Parts after completion",
			consume: func(s *ai.TextStream) error {
				for range s.Parts() {
				}
				for range s.Parts() {
				}
				return s.Err()
			},
			want: ai.ErrStreamConsumed,
		},
		{
			name: "Text after Parts",
			consume: func(s *ai.TextStream) error {
				for range s.Parts() {
				}
				for range s.Text() {
				}
				return s.Err()
			},
			want: ai.ErrStreamConsumed,
		},
		{
			name: "Parts inside Parts",
			consume: func(s *ai.TextStream) error {
				for range s.Parts() {
					for range s.Parts() {
					}
					break
				}
				return s.Err()
			},
			want: ai.ErrStreamActive,
		},
		{
			name: "Result inside Parts",
			consume: func(s *ai.TextStream) error {
				var err error
				for range s.Parts() {
					_, err = s.Result()
					break
				}
				return err
			},
			want: ai.ErrStreamActive,
		},
		{
			name: "Parts after Result",
			consume: func(s *ai.TextStream) error {
				if _, err := s.Result(); err != nil {
					return err
				}
				for range s.Parts() {
				}
				return s.Err()
			},
			want: ai.ErrStreamConsumed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := ai.StreamText(t.Context(), aitest.NewModel(aitest.Turn(aitest.Text("x"))), ai.TextRequest{})
			if got := tt.consume(s); !errors.Is(got, tt.want) {
				t.Errorf("error = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStreamText_resultAfterPartsStillAvailable(t *testing.T) {
	s := ai.StreamText(t.Context(), aitest.NewModel(aitest.Turn(aitest.Text("x"))), ai.TextRequest{})
	for range s.Parts() {
	}
	res, err := s.Result()
	if err != nil || res.Text != "x" {
		t.Errorf("Result() = %+v, %v; want Text %q", res, err, "x")
	}
	if again, _ := s.Result(); again != res {
		t.Error("Result() returned a different aggregate on the second call")
	}
}

func TestStreamText_earlyBreak(t *testing.T) {
	m := aitest.NewModel(toolTurn("search", `{"query":"a"}`), aitest.Turn(aitest.TextChunks("a", "b", "c")))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})
	for p := range s.Parts() {
		if _, ok := p.(ai.TextDelta); ok {
			break
		}
	}
	res, err := s.Result()
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Result() error = %v, want context.Canceled", err)
	}
	if got, want := len(res.Steps), 1; got != want {
		t.Errorf("completed steps = %d, want %d", got, want)
	}
}

func TestStreamText_errorMidStream(t *testing.T) {
	boom := errors.New("connection reset")
	m := aitest.NewModel(aitest.Turn(aitest.Text("partial"), aitest.Err(boom)))
	s := ai.StreamText(t.Context(), m, ai.TextRequest{})

	text := strings.Join(slices.Collect(s.Text()), "")
	if text != "partial" {
		t.Errorf("streamed text = %q, want %q", text, "partial")
	}
	if !errors.Is(s.Err(), boom) {
		t.Errorf("Err() = %v, want %v", s.Err(), boom)
	}
	res, err := s.Result()
	if !errors.Is(err, boom) || len(res.Steps) != 0 || res.Text != "" {
		t.Errorf("Result() = %+v, %v; want no completed steps and the error", res, err)
	}
}

func TestStreamText_matchesGenerate(t *testing.T) {
	script := func() *aitest.Model {
		return aitest.NewModel(
			aitest.Turn(aitest.Reasoning("thinking"), aitest.ToolCall("search", `{"query":"a"}`)),
			aitest.Turn(aitest.Text("answer"), aitest.Usage(3, 4)),
		)
	}
	req := ai.TextRequest{Tools: []ai.Tool{searchTool()}, Messages: []ai.Message{ai.User("q")}}

	gen := mustGenerate(t, script(), req)
	got, err := ai.StreamText(t.Context(), script(), req).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	opts := []cmp.Option{cmpopts.IgnoreUnexported(ai.TextResult{}), cmpopts.IgnoreFields(ai.Step{}, "Header")}
	if diff := cmp.Diff(gen, got, opts...); diff != "" {
		t.Errorf("stream aggregate differs from Generate (-generate +stream):\n%s", diff)
	}
}

func TestStreamText_aggregatesCitationsAndSignatures(t *testing.T) {
	sig := ai.ProviderData{}.WithRaw("anthropic", rawJSON(`{"signature":"abc"}`))
	m := aitest.NewModel(aitest.Turn(
		aitest.StreamPart(ai.ReasoningStart{ID: "r"}),
		aitest.StreamPart(ai.ReasoningEnd{ID: "r", Provider: sig}),
		aitest.StreamPart(ai.TextStart{ID: "t"}),
		aitest.StreamPart(ai.TextDelta{ID: "t", Text: "cited"}),
		aitest.StreamPart(ai.Citation{URL: "https://example.com", Title: "Ex"}),
		aitest.StreamPart(ai.TextEnd{ID: "t"}),
	))
	res, err := ai.StreamText(t.Context(), m, ai.TextRequest{}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	want := ai.Parts{
		ai.Reasoning{Provider: sig},
		ai.Text{Text: "cited", Citations: []ai.Citation{{URL: "https://example.com", Title: "Ex"}}},
	}
	if diff := cmp.Diff(want, res.Content); diff != "" {
		t.Errorf("Content mismatch (-want +got):\n%s", diff)
	}
}

func TestStreamText_missingFinishIsAnError(t *testing.T) {
	m := &truncatedModel{}
	res, err := ai.StreamText(t.Context(), m, ai.TextRequest{MaxRetries: new(0)}).Result()
	if err == nil {
		t.Fatalf("Result() = %+v, nil; want an error for a stream without StepEnd", res)
	}
}

func TestStreamText_timeouts(t *testing.T) {
	tests := []struct {
		name     string
		turn     aitest.TurnScript
		timeouts ai.Timeouts
		wantText string
	}{
		{
			name:     "first chunk",
			turn:     aitest.Turn(aitest.Delay(time.Minute), aitest.Text("late")),
			timeouts: ai.Timeouts{FirstChunk: time.Second},
			wantText: "no data from model for 1s",
		},
		{
			name:     "chunk gap",
			turn:     aitest.Turn(aitest.Text("early"), aitest.Delay(time.Minute), aitest.Text("late")),
			timeouts: ai.Timeouts{Chunk: time.Second},
			wantText: "no data from model for 1s",
		},
		{
			name:     "step",
			turn:     aitest.Turn(aitest.Text("early"), aitest.Delay(time.Minute), aitest.Text("late")),
			timeouts: ai.Timeouts{Step: 5 * time.Second},
			wantText: "step timed out after 5s",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				m := aitest.NewModel(tt.turn)
				_, err := ai.StreamText(t.Context(), m, ai.TextRequest{Timeouts: tt.timeouts, MaxRetries: new(0)}).Result()
				if err == nil || !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("error = %v, want a deadline error", err)
				}
				if got := err.Error(); !containsStr(got, tt.wantText) {
					t.Errorf("error = %q, want it to contain %q", got, tt.wantText)
				}
			})
		})
	}
}

func TestGenerateText_stepTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := aitest.NewModel(aitest.Turn(aitest.Delay(time.Minute), aitest.Text("late")))
		_, err := ai.GenerateText(t.Context(), m, ai.TextRequest{Timeouts: ai.Timeouts{Step: time.Second}})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("error = %v, want a deadline error", err)
		}
	})
}

func TestStreamText_parentCancel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		m := aitest.NewModel(aitest.Turn(aitest.Text("early"), aitest.Delay(time.Hour), aitest.Text("late")))
		time.AfterFunc(time.Second, cancel)
		_, err := ai.StreamText(ctx, m, ai.TextRequest{}).Result()
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want context.Canceled", err)
		}
	})
}
