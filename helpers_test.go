package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type searchIn struct {
	Query string `json:"query" jsonschema:"the search query"`
}

type searchOut struct {
	Hits []string `json:"hits"`
}

func searchTool() ai.Tool {
	return ai.NewTool("search", "web search", func(_ context.Context, in searchIn) (searchOut, error) {
		return searchOut{Hits: []string{in.Query + "-1"}}, nil
	})
}

func mustGenerate(t *testing.T, m ai.LanguageModel, req ai.TextRequest) *ai.TextResult {
	t.Helper()
	res, err := ai.GenerateText(t.Context(), m, req)
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	return res
}

func toolTurn(name, input string) aitest.TurnScript {
	return aitest.Turn(aitest.ToolCall(name, input))
}

func resultsOf(msgs []ai.Message) []ai.ToolResult {
	var out []ai.ToolResult
	for _, m := range msgs {
		for _, p := range m.Content {
			if r, ok := p.(ai.ToolResult); ok {
				out = append(out, r)
			}
		}
	}
	return out
}

func rawJSON(s string) json.RawMessage { return json.RawMessage(s) }

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return b
}

func mustUnmarshal(t *testing.T, b []byte, v any) {
	t.Helper()
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", b, err)
	}
}

func containsStr(s, sub string) bool { return strings.Contains(s, sub) }

type truncatedModel struct{}

func (*truncatedModel) Provider() string { return "test" }

func (*truncatedModel) ModelID() string { return "truncated" }

func (*truncatedModel) Generate(context.Context, *ai.Call) (*ai.Response, error) {
	return nil, errors.New("unused")
}

func (*truncatedModel) Stream(context.Context, *ai.Call) (*ai.StreamResponse, error) {
	return &ai.StreamResponse{Parts: func(yield func(ai.StreamPart, error) bool) {
		yield(ai.TextDelta{ID: "t", Text: "cut off"}, nil)
	}}, nil
}

type capture struct {
	calls []ai.Call
}

func newCapture() *capture { return &capture{} }

func (*capture) Provider() string { return "test" }

func (*capture) ModelID() string { return "capture" }

func (c *capture) Generate(_ context.Context, call *ai.Call) (*ai.Response, error) {
	c.calls = append(c.calls, *call.Clone())
	return &ai.Response{Content: ai.Parts{ai.Text{Text: "ok"}}, FinishReason: ai.FinishStop}, nil
}

func (*capture) Stream(context.Context, *ai.Call) (*ai.StreamResponse, error) {
	return nil, errors.New("unused")
}

func newScripted(t *testing.T, toolInputs ...string) *aitest.Model {
	t.Helper()
	turns := make([]aitest.ModelOption, 0, len(toolInputs)+1)
	for _, in := range toolInputs {
		turns = append(turns, aitest.Turn(aitest.ToolCall("w", in)))
	}
	turns = append(turns, aitest.Turn(aitest.Text("done")))
	return aitest.NewModel(turns...)
}
