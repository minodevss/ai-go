package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
	"github.com/minodevss/ai-go/openai"
)

const roundTripFixture = "testdata/live/roundtrip.json"

type liveWeatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

func roundTripRequest(msgs []ai.Message) ai.TextRequest {
	weather := ai.NewTool("get_weather", "Current weather for a city", func(_ context.Context, in liveWeatherInput) (string, error) {
		return "Sunny, 21C in " + in.City, nil
	})
	return ai.TextRequest{
		Instructions:    ai.Texts("Always call get_weather before answering a weather question. Answer in two short sentences."),
		Messages:        msgs,
		Tools:           []ai.Tool{weather},
		Reasoning:       ai.ReasoningHigh,
		MaxSteps:        4,
		MaxOutputTokens: 2000,
		Provider:        ai.Providers(openai.Options{Store: new(false), ReasoningSummary: "auto"}),
	}
}

func runRoundTrip(t *testing.T, model ai.LanguageModel) {
	t.Helper()
	first := ai.StreamText(t.Context(), model, roundTripRequest([]ai.Message{ai.User("What is the weather in Seoul, and what is 17*23? Use the tool for the weather.")}))
	res, err := first.Result()
	if err != nil {
		t.Fatalf("first turn error = %v", err)
	}
	if len(res.Steps) < 2 {
		t.Fatalf("first turn took %d steps, want at least 2 (tool call, then answer)", len(res.Steps))
	}
	var sawReasoning, sawCall bool
	for _, m := range res.Messages {
		for _, p := range m.Content {
			switch v := p.(type) {
			case ai.Reasoning:
				data, ok := ai.ProviderValue[map[string]any](v.Provider, "openai")
				item, _ := data["item"].(map[string]any)
				if !ok || item["type"] != "reasoning" || item["encrypted_content"] == nil {
					t.Errorf("reasoning part has no raw item with encrypted content: %v", data)
				}
				sawReasoning = true
			case ai.ToolCall:
				sawCall = true
			}
		}
	}
	if !sawReasoning || !sawCall {
		t.Fatalf("first turn reasoning = %v, tool call = %v; want both", sawReasoning, sawCall)
	}
	if strings.TrimSpace(res.Text) == "" {
		t.Fatal("first turn produced no text")
	}

	history := append([]ai.Message{ai.User("What is the weather in Seoul, and what is 17*23? Use the tool for the weather.")}, res.Messages...)
	history = append(history, ai.User("Should I bring an umbrella? One short sentence."))
	second := ai.StreamText(t.Context(), model, roundTripRequest(history))
	next, err := second.Result()
	if err != nil {
		t.Fatalf("second turn with replayed history error = %v", err)
	}
	if strings.TrimSpace(next.Text) == "" {
		t.Error("second turn produced no text")
	}
	t.Logf("first: %q (input %d, output %d, reasoning %d)", res.Text, res.Usage.Input, res.Usage.Output, res.Usage.Reasoning)
	t.Logf("second: %q (input %d, output %d)", next.Text, next.Usage.Input, next.Usage.Output)
}

func TestRecordedRoundTrip(t *testing.T) {
	rt := record.New(t, roundTripFixture, record.WithBodyMatch())
	model := openai.New(openai.WithAPIKey("test-key"), openai.WithBaseURL("http://gateway.invalid/v1"), openai.WithHTTPClient(&http.Client{Transport: rt})).Model("gpt-6-luna")
	runRoundTrip(t, model)
	if got := len(rt.Requests()); got != 4 {
		t.Errorf("sent %d requests, want 4", got)
	}
}

func TestRecordedRoundTrip_wireIdentity(t *testing.T) {
	file, err := record.Load(roundTripFixture)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	for i := 1; i < len(file.Exchanges); i++ {
		sent, err := requestItems(file.Exchanges[i].Request.Bytes())
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		got := compactStrings(t, sent)
		for j := range i {
			items, err := streamItems(file.Exchanges[j].Response.Bytes())
			if err != nil {
				t.Fatalf("response %d: %v", j, err)
			}
			for _, item := range compactStrings(t, items) {
				if !slices.Contains(got, item) {
					t.Errorf("request %d does not replay the item from response %d byte for byte:\n%s", i, j, item)
				}
			}
		}
	}
}

func compactStrings(t *testing.T, items []json.RawMessage) []string {
	t.Helper()
	out := make([]string, len(items))
	for i, it := range items {
		var b bytes.Buffer
		if err := json.Compact(&b, it); err != nil {
			t.Fatalf("Compact() error = %v", err)
		}
		out[i] = b.String()
	}
	return out
}

func TestRecordedAuthError(t *testing.T) {
	rt := record.New(t, "testdata/live/autherror.json")
	p := openai.New(openai.WithAPIKey("invalid-key"), openai.WithBaseURL("http://gateway.invalid/v1"), openai.WithHTTPClient(&http.Client{Transport: rt}))
	_, err := ai.GenerateText(t.Context(), p.Model("gpt-6-luna"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || !errors.Is(err, ai.ErrAuth) || apiErr.Code != "invalid_api_key" {
		t.Errorf("GenerateText() error = %v, want an invalid_api_key authentication error", err)
	}
}
