package ai_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func TestWrapModel_transformRequestSeesCloneOnly(t *testing.T) {
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	m := ai.WrapModel(inner, ai.Middleware{TransformRequest: func(_ context.Context, c *ai.Call) error {
		c.Instructions = append(c.Instructions, ai.Text{Text: "injected"})
		c.Messages[0] = ai.User("rewritten")
		return nil
	}})
	call := &ai.Call{Messages: []ai.Message{ai.User("original")}}
	if _, err := m.Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if !cmp.Equal(call.Messages[0], ai.User("original")) || len(call.Instructions) != 0 {
		t.Errorf("caller's call was mutated: %+v", call)
	}
	seen := inner.Calls()[0]
	if len(seen.Instructions) != 1 || !cmp.Equal(seen.Messages[0], ai.User("rewritten")) {
		t.Errorf("inner model saw %+v, want the transformed call", seen)
	}
}

func TestWrapModel_transformError(t *testing.T) {
	boom := errors.New("rejected")
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	m := ai.WrapModel(inner, ai.Middleware{TransformRequest: func(context.Context, *ai.Call) error { return boom }})

	if _, err := m.Generate(t.Context(), &ai.Call{}); !errors.Is(err, boom) {
		t.Errorf("Generate() error = %v, want %v", err, boom)
	}
	if _, err := m.Stream(t.Context(), &ai.Call{}); !errors.Is(err, boom) {
		t.Errorf("Stream() error = %v, want %v", err, boom)
	}
	if len(inner.Calls()) != 0 {
		t.Error("inner model was called after a transform error")
	}
}

func TestWrapModel_wrapOrderAndIdentity(t *testing.T) {
	var order []string
	tag := func(name string) ai.Middleware {
		return ai.Middleware{WrapGenerate: func(ctx context.Context, c *ai.Call, next func(context.Context, *ai.Call) (*ai.Response, error)) (*ai.Response, error) {
			order = append(order, name+">")
			resp, err := next(ctx, c)
			order = append(order, "<"+name)
			return resp, err
		}}
	}
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	m := ai.WrapModel(ai.WrapModel(inner, tag("inner")), tag("outer"))
	if _, err := ai.GenerateText(t.Context(), m, ai.TextRequest{}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if diff := cmp.Diff([]string{"outer>", "inner>", "<inner", "<outer"}, order); diff != "" {
		t.Errorf("order mismatch (-want +got):\n%s", diff)
	}
	if m.Provider() != "aitest" || m.ModelID() != "scripted" {
		t.Errorf("wrapped identity = %s/%s, want aitest/scripted", m.Provider(), m.ModelID())
	}
}

func TestWrapModel_wrapStream(t *testing.T) {
	var called bool
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	m := ai.WrapModel(inner, ai.Middleware{WrapStream: func(ctx context.Context, c *ai.Call, next func(context.Context, *ai.Call) (*ai.StreamResponse, error)) (*ai.StreamResponse, error) {
		called = true
		return next(ctx, c)
	}})
	res, err := ai.StreamText(t.Context(), m, ai.TextRequest{}).Result()
	if err != nil || res.Text != "ok" || !called {
		t.Errorf("Result() = %+v, %v, called %v", res, err, called)
	}
}

type marker interface{ Marker() string }

type markedModel struct {
	ai.LanguageModel
	name string
}

func (m markedModel) Marker() string { return m.name }

func TestAs(t *testing.T) {
	base := markedModel{LanguageModel: aitest.NewModel(), name: "base"}
	other := markedModel{LanguageModel: aitest.NewModel(), name: "other"}
	passthrough := ai.Middleware{}
	tests := []struct {
		name   string
		model  any
		want   string
		wantOK bool
	}{
		{name: "direct", model: base, want: "base", wantOK: true},
		{name: "through wrappers", model: ai.WrapModel(ai.WrapModel(base, passthrough), passthrough), want: "base", wantOK: true},
		{name: "fallback children in order", model: ai.Fallback(aitest.NewModel(), base, other), want: "base", wantOK: true},
		{name: "depth first before siblings", model: ai.Fallback(ai.WrapModel(other, passthrough), base), want: "other", wantOK: true},
		{name: "wrapped fallback", model: ai.WrapModel(ai.Fallback(base), passthrough), want: "base", wantOK: true},
		{name: "absent", model: ai.WrapModel(aitest.NewModel(), passthrough)},
		{name: "nil", model: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ai.As[marker](tt.model)
			if ok != tt.wantOK {
				t.Fatalf("As ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && got.Marker() != tt.want {
				t.Errorf("As found %q, want %q", got.Marker(), tt.want)
			}
		})
	}
}

func TestAs_concreteType(t *testing.T) {
	inner := aitest.NewModel()
	got, ok := ai.As[*aitest.Model](ai.WrapModel(inner, ai.Middleware{}))
	if !ok || got != inner {
		t.Errorf("As[*aitest.Model] = %v, %v; want the inner model", got, ok)
	}
}

func TestExtractReasoning_generate(t *testing.T) {
	tests := []struct {
		name string
		text string
		want ai.Parts
	}{
		{name: "no tags", text: "plain", want: ai.Parts{ai.Text{Text: "plain"}}},
		{name: "leading think", text: "<think>plan</think>answer", want: ai.Parts{ai.Reasoning{Text: "plan"}, ai.Text{Text: "answer"}}},
		{name: "text around", text: "a<think>b</think>c", want: ai.Parts{ai.Text{Text: "a"}, ai.Reasoning{Text: "b"}, ai.Text{Text: "c"}}},
		{name: "two blocks", text: "<think>1</think><think>2</think>x", want: ai.Parts{ai.Reasoning{Text: "1"}, ai.Reasoning{Text: "2"}, ai.Text{Text: "x"}}},
		{name: "unclosed", text: "<think>never ends", want: ai.Parts{ai.Reasoning{Text: "never ends"}}},
		{name: "only reasoning", text: "<think>solo</think>", want: ai.Parts{ai.Reasoning{Text: "solo"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := ai.WrapModel(aitest.NewModel(aitest.Turn(aitest.Text(tt.text))), ai.ExtractReasoning("think"))
			res := mustGenerate(t, m, ai.TextRequest{})
			if diff := cmp.Diff(tt.want, res.Content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractReasoning_stream(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   ai.Parts
	}{
		{name: "tags in one chunk", chunks: []string{"<think>plan</think>answer"}, want: ai.Parts{ai.Reasoning{Text: "plan"}, ai.Text{Text: "answer"}}},
		{name: "tags split across chunks", chunks: []string{"<th", "ink>pl", "an</thi", "nk>ans", "wer"}, want: ai.Parts{ai.Reasoning{Text: "plan"}, ai.Text{Text: "answer"}}},
		{name: "one byte at a time", chunks: splitBytes("a<think>b</think>c"), want: ai.Parts{ai.Text{Text: "a"}, ai.Reasoning{Text: "b"}, ai.Text{Text: "c"}}},
		{name: "false start", chunks: []string{"x <th", "is is fine"}, want: ai.Parts{ai.Text{Text: "x <this is fine"}}},
		{name: "no tags", chunks: []string{"he", "llo"}, want: ai.Parts{ai.Text{Text: "hello"}}},
		{name: "unclosed at end", chunks: []string{"<think>abc"}, want: ai.Parts{ai.Reasoning{Text: "abc"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := ai.WrapModel(aitest.NewModel(aitest.Turn(aitest.TextChunks(tt.chunks...))), ai.ExtractReasoning("think"))
			res, err := ai.StreamText(t.Context(), m, ai.TextRequest{}).Result()
			if err != nil {
				t.Fatalf("Result() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, res.Content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func splitBytes(s string) []string {
	out := make([]string, len(s))
	for i := range s {
		out[i] = s[i : i+1]
	}
	return out
}

func TestExtractReasoning_streamKeepsOtherParts(t *testing.T) {
	m := ai.WrapModel(aitest.NewModel(aitest.Turn(aitest.ToolCall("search", `{"query":"q"}`))), ai.ExtractReasoning("think"))
	res, err := ai.StreamText(t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}, MaxSteps: 1}).Result()
	if err != nil || len(res.Steps[0].ToolCalls) != 1 {
		t.Errorf("Result() = %+v, %v; want the tool call to pass through", res, err)
	}
}

func TestExtractReasoning_propagatesStreamErrors(t *testing.T) {
	boom := errors.New("cut")
	m := ai.WrapModel(aitest.NewModel(aitest.Turn(aitest.Text("<think>a"), aitest.Err(boom))), ai.ExtractReasoning("think"))
	if _, err := ai.StreamText(t.Context(), m, ai.TextRequest{}).Result(); !errors.Is(err, boom) {
		t.Errorf("Result() error = %v, want %v", err, boom)
	}
}

func TestDefaultSettings(t *testing.T) {
	defaults := ai.TextRequest{
		MaxOutputTokens: 500,
		Temperature:     new(0.3),
		TopP:            new(0.9),
		TopK:            40,
		Seed:            new(7),
		StopSequences:   []string{"END"},
		Reasoning:       ai.ReasoningLow,
		Headers:         http.Header{"X-Default": {"d"}, "X-Both": {"default"}},
		Provider:        ai.ProviderData{}.WithRaw("a", rawJSON(`1`)).WithRaw("b", rawJSON(`2`)),
	}
	tests := []struct {
		name string
		req  ai.TextRequest
		want func(c ai.Call) error
	}{
		{
			name: "fills everything unset",
			req:  ai.TextRequest{},
			want: func(c ai.Call) error {
				if c.MaxOutputTokens != 500 || *c.Temperature != 0.3 || *c.TopP != 0.9 || c.TopK != 40 || *c.Seed != 7 || c.Reasoning != ai.ReasoningLow || !slices.Equal(c.StopSequences, []string{"END"}) {
					return fmt.Errorf("settings not applied: %+v", c)
				}
				return nil
			},
		},
		{
			name: "request values win",
			req:  ai.TextRequest{MaxOutputTokens: 10, Temperature: new(0.0), TopK: 1, Reasoning: ai.ReasoningHigh, StopSequences: []string{"X"}},
			want: func(c ai.Call) error {
				if c.MaxOutputTokens != 10 || *c.Temperature != 0 || c.TopK != 1 || c.Reasoning != ai.ReasoningHigh || !slices.Equal(c.StopSequences, []string{"X"}) {
					return fmt.Errorf("request values overridden: %+v", c)
				}
				return nil
			},
		},
		{
			name: "headers and provider data merge",
			req:  ai.TextRequest{Headers: http.Header{"X-Both": {"mine"}}, Provider: ai.ProviderData{}.WithRaw("b", rawJSON(`20`))},
			want: func(c ai.Call) error {
				if c.Headers.Get("X-Default") != "d" || c.Headers.Get("X-Both") != "mine" {
					return fmt.Errorf("headers = %v", c.Headers)
				}
				a, _ := ai.ProviderValue[int](c.Provider, "a")
				b, _ := ai.ProviderValue[int](c.Provider, "b")
				if a != 1 || b != 20 {
					return fmt.Errorf("provider data a=%d b=%d, want 1, 20", a, b)
				}
				return nil
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
			m := ai.WrapModel(inner, ai.DefaultSettings(defaults))
			mustGenerate(t, m, tt.req)
			if err := tt.want(inner.Calls()[0]); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestDefaultSettings_isolatedFromLaterMutation(t *testing.T) {
	defaults := ai.TextRequest{Temperature: new(0.3)}
	mw := ai.DefaultSettings(defaults)
	*defaults.Temperature = 1.0
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	mustGenerate(t, ai.WrapModel(inner, mw), ai.TextRequest{})
	if got := *inner.Calls()[0].Temperature; got != 0.3 {
		t.Errorf("Temperature = %v, want 0.3", got)
	}
}
