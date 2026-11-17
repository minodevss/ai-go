package ai_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func TestStreamText_customPart(t *testing.T) {
	model := aitest.NewModel(aitest.Turn(
		aitest.StreamPart(ai.Custom{Kind: "x.list", Data: []byte(`{"a":1}`)}),
		aitest.Text("done"),
	))
	res, err := ai.StreamText(t.Context(), model, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	want := ai.Parts{ai.Custom{Kind: "x.list", Data: []byte(`{"a":1}`)}, ai.Text{Text: "done"}}
	if diff := cmp.Diff(want, res.Content); diff != "" {
		t.Errorf("content mismatch (-want +got):\n%s", diff)
	}
}

func TestStreamText_citationFollowsEndedText(t *testing.T) {
	cite := ai.Citation{Kind: "url", URL: "https://x.test"}
	model := aitest.NewModel(aitest.Turn(
		aitest.StreamPart(ai.TextStart{ID: "a"}),
		aitest.StreamPart(ai.TextStart{ID: "b"}),
		aitest.StreamPart(ai.TextDelta{ID: "a", Text: "first"}),
		aitest.StreamPart(ai.TextDelta{ID: "b", Text: "second"}),
		aitest.StreamPart(ai.TextEnd{ID: "a"}),
		aitest.StreamPart(cite),
		aitest.StreamPart(ai.TextEnd{ID: "b"}),
	))
	res, err := ai.StreamText(t.Context(), model, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}).Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	want := ai.Parts{ai.Text{Text: "first", Citations: []ai.Citation{cite}}, ai.Text{Text: "second"}}
	if diff := cmp.Diff(want, res.Content); diff != "" {
		t.Errorf("content mismatch (-want +got):\n%s", diff)
	}
}
