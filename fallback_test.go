package ai_test

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func TestFallback_generate(t *testing.T) {
	tests := []struct {
		name      string
		first     error
		wantText  string
		wantErr   bool
		wantCalls [2]int
	}{
		{name: "503 falls back", first: apiErr(503), wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "rate limit falls back", first: apiErr(429), wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "quota falls back", first: &ai.APIError{StatusCode: 429, Code: "insufficient_quota"}, wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "auth falls back", first: apiErr(401), wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "context too long falls back", first: &ai.APIError{StatusCode: 400, Code: "context_length_exceeded"}, wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "plain 400 stops", first: apiErr(400), wantErr: true, wantCalls: [2]int{1, 0}},
		{name: "unknown error stops", first: errors.New("bug"), wantErr: true, wantCalls: [2]int{1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := aitest.NewModel(aitest.Turn(aitest.Err(tt.first)))
			b := aitest.NewModel(aitest.Turn(aitest.Text("second")))
			res, err := ai.GenerateText(t.Context(), ai.Fallback(a, b), ai.TextRequest{MaxRetries: new(0)})
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && res.Text != tt.wantText {
				t.Errorf("Text = %q, want %q", res.Text, tt.wantText)
			}
			if got := [2]int{len(a.Calls()), len(b.Calls())}; got != tt.wantCalls {
				t.Errorf("calls = %v, want %v", got, tt.wantCalls)
			}
		})
	}
}

func TestFallback_allFail(t *testing.T) {
	e1, e2 := apiErr(503), apiErr(529)
	a := aitest.NewModel(aitest.Turn(aitest.Err(e1)))
	b := aitest.NewModel(aitest.Turn(aitest.Err(e2)))
	_, err := ai.GenerateText(t.Context(), ai.Fallback(a, b), ai.TextRequest{MaxRetries: new(0)})
	if !errors.Is(err, e1) || !errors.Is(err, e2) {
		t.Errorf("error = %v, want both failures wrapped", err)
	}
}

func TestFallback_noModels(t *testing.T) {
	if _, err := ai.GenerateText(t.Context(), ai.Fallback(), ai.TextRequest{}); err == nil {
		t.Error("error = nil, want an error for an empty fallback")
	}
}

func TestFallback_stream(t *testing.T) {
	tests := []struct {
		name      string
		first     aitest.TurnScript
		wantText  string
		wantErr   bool
		wantCalls [2]int
	}{
		{name: "error before any output", first: aitest.Turn(aitest.Err(apiErr(503))), wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "error before first part inside the stream", first: aitest.Turn(aitest.Delay(0), aitest.Err(apiErr(529))), wantText: "second", wantCalls: [2]int{1, 1}},
		{name: "error after output is surfaced", first: aitest.Turn(aitest.Text("partial"), aitest.Err(apiErr(503))), wantErr: true, wantCalls: [2]int{1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := aitest.NewModel(tt.first)
			b := aitest.NewModel(aitest.Turn(aitest.Text("second")))
			res, err := ai.StreamText(t.Context(), ai.Fallback(a, b), ai.TextRequest{MaxRetries: new(0)}).Result()
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && res.Text != tt.wantText {
				t.Errorf("Text = %q, want %q", res.Text, tt.wantText)
			}
			if got := [2]int{len(a.Calls()), len(b.Calls())}; got != tt.wantCalls {
				t.Errorf("calls = %v, want %v", got, tt.wantCalls)
			}
		})
	}
}

func TestFallback_unwrapOrder(t *testing.T) {
	a, b := aitest.NewModel(), aitest.NewModel()
	un, ok := ai.Fallback(a, b).(interface{ Unwrap() []ai.LanguageModel })
	if !ok {
		t.Fatal("Fallback does not implement Unwrap() []LanguageModel")
	}
	got := un.Unwrap()
	if len(got) != 2 || got[0] != ai.LanguageModel(a) || got[1] != ai.LanguageModel(b) {
		t.Errorf("Unwrap() = %v, want [a b]", got)
	}
	got[0] = nil
	if again := un.Unwrap(); again[0] == nil {
		t.Error("Unwrap exposes the internal slice")
	}
}

func TestFallback_identity(t *testing.T) {
	m := ai.Fallback(aitest.NewModel(), aitest.NewModel())
	if diff := cmp.Diff("fallback", m.Provider()); diff != "" {
		t.Errorf("Provider() mismatch (-want +got):\n%s", diff)
	}
	if got, want := m.ModelID(), "aitest:scripted,aitest:scripted"; got != want {
		t.Errorf("ModelID() = %q, want %q", got, want)
	}
}
