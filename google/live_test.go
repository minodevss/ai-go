//go:build live

package google_test

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
	"github.com/minodevss/ai-go/google"
)

type liveWeatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

func liveProvider(t *testing.T, name, key string) *google.Provider {
	t.Helper()
	client := &http.Client{}
	if record.Updating() {
		client = &http.Client{Transport: record.New(t, "testdata/live/"+name+".json")}
	}
	return google.New(google.WithAPIKey(key), google.WithHTTPClient(client))
}

func TestLive_roundTrip(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY is not set")
	}
	model := liveProvider(t, "roundtrip", key).Model("gemini-3-flash-preview")
	weather := ai.NewTool("get_weather", "Current weather for a city", func(_ context.Context, in liveWeatherInput) (string, error) {
		return "Sunny, 21C in " + in.City, nil
	})
	request := func(msgs []ai.Message) ai.TextRequest {
		return ai.TextRequest{
			Instructions:    ai.Texts("Always call get_weather before answering a weather question. Answer in two short sentences."),
			Messages:        msgs,
			Tools:           []ai.Tool{weather},
			Reasoning:       ai.ReasoningHigh,
			MaxSteps:        4,
			MaxOutputTokens: 2000,
		}
	}
	question := ai.User("What is the weather in Seoul, and what is 17*23? Use the tool for the weather.")
	res, err := ai.StreamText(t.Context(), model, request([]ai.Message{question})).Result()
	if err != nil {
		t.Fatalf("first turn error = %v", err)
	}
	if len(res.Steps) < 2 || strings.TrimSpace(res.Text) == "" {
		t.Fatalf("first turn took %d steps and produced %q, want a tool call step and an answer", len(res.Steps), res.Text)
	}
	history := append([]ai.Message{question}, res.Messages...)
	history = append(history, ai.User("Should I bring an umbrella? One short sentence."))
	next, err := ai.StreamText(t.Context(), model, request(history)).Result()
	if err != nil {
		t.Fatalf("second turn with replayed history error = %v", err)
	}
	if strings.TrimSpace(next.Text) == "" {
		t.Error("second turn produced no text")
	}
}

func TestLive_authError(t *testing.T) {
	p := liveProvider(t, "autherror", "invalid-key")
	_, err := ai.GenerateText(t.Context(), p.Model("gemini-3-flash-preview"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	if _, ok := errors.AsType[*ai.APIError](err); !ok {
		t.Errorf("GenerateText() error = %v, want an *ai.APIError", err)
	}
}
