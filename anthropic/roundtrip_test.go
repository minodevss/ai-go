package anthropic_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
	"github.com/minodevss/ai-go/anthropic"
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
		Reasoning:       ai.ReasoningMedium,
		MaxSteps:        4,
		MaxRetries:      new(0),
		MaxOutputTokens: 4000,
	}
}

func runRoundTrip(t *testing.T, model ai.LanguageModel) {
	t.Helper()
	question := ai.User("What is the weather in Seoul? Use the tool.")
	first := ai.StreamText(t.Context(), model, roundTripRequest([]ai.Message{question}))
	res, err := first.Result()
	if errors.Is(err, ai.ErrRateLimit) {
		t.Skipf("the account behind the gateway is rate limited, so the thinking round trip was not exercised live: %v", err)
	}
	if err != nil {
		t.Fatalf("first turn error = %v", err)
	}
	if len(res.Steps) < 2 {
		t.Fatalf("first turn took %d steps, want at least 2 (tool call, then answer)", len(res.Steps))
	}
	var sawThinking, sawCall bool
	for _, m := range res.Messages {
		for _, p := range m.Content {
			switch v := p.(type) {
			case ai.Reasoning:
				data, ok := ai.ProviderValue[map[string]any](v.Provider, "anthropic")
				block, _ := data["block"].(map[string]any)
				if !ok || block["signature"] == nil || block["signature"] == "" {
					t.Errorf("reasoning part has no signed raw block: %v", data)
				}
				sawThinking = true
			case ai.ToolCall:
				sawCall = true
			}
		}
	}
	if !sawThinking || !sawCall {
		t.Fatalf("first turn thinking = %v, tool call = %v; want both", sawThinking, sawCall)
	}
	if strings.TrimSpace(res.Text) == "" {
		t.Fatal("first turn produced no text")
	}

	history := append([]ai.Message{question}, res.Messages...)
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
	if _, err := os.Stat(roundTripFixture); err != nil {
		t.Skipf("%s was not recorded because the live account was rate limited", roundTripFixture)
	}
	rt := record.New(t, roundTripFixture, record.WithBodyMatch())
	model := anthropic.New(anthropic.WithAPIKey("test-key"), anthropic.WithBaseURL("http://gateway.invalid"), anthropic.WithHTTPClient(&http.Client{Transport: rt})).Model("claude-haiku-5-5")
	runRoundTrip(t, model)
	if got := len(rt.Requests()); got < 3 {
		t.Errorf("sent %d requests, want at least 3", got)
	}
}

func TestRecordedRoundTrip_wireIdentity(t *testing.T) {
	if _, err := os.Stat(roundTripFixture); err != nil {
		t.Skipf("%s was not recorded because the live account was rate limited", roundTripFixture)
	}
	file, err := record.Load(roundTripFixture)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	for i := 1; i < len(file.Exchanges); i++ {
		sent, err := requestBlocks(file.Exchanges[i].Request.Bytes())
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		got := compactStrings(t, sent)
		for j := range i {
			items, err := streamBlocks(file.Exchanges[j].Response.Bytes())
			if err != nil {
				t.Fatalf("response %d: %v", j, err)
			}
			for _, item := range compactStrings(t, items) {
				if !slices.Contains(got, item) && !isTextBlock(item) {
					t.Errorf("request %d does not replay the block from response %d byte for byte:\n%s", i, j, item)
				}
			}
		}
	}
}

func isTextBlock(block string) bool { return strings.HasPrefix(block, `{"type":"text"`) }

func TestRecordedAuthError(t *testing.T) {
	rt := record.New(t, "testdata/live/autherror.json")
	p := anthropic.New(anthropic.WithAPIKey("invalid-key"), anthropic.WithBaseURL("http://gateway.invalid"), anthropic.WithHTTPClient(&http.Client{Transport: rt}))
	_, err := ai.GenerateText(t.Context(), p.Model("claude-haiku-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("GenerateText() error = %v, want an authentication error", err)
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

func TestRecordedRateLimit(t *testing.T) {
	rt := record.New(t, "testdata/live/ratelimit.json")
	p := anthropic.New(anthropic.WithAPIKey("test-key"), anthropic.WithBaseURL("http://gateway.invalid"), anthropic.WithHTTPClient(&http.Client{Transport: rt}))
	_, err := ai.GenerateText(t.Context(), p.Model("claude-haiku-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || !errors.Is(err, ai.ErrRateLimit) || apiErr.Code != "rate_limit_error" || apiErr.RequestID == "" || !apiErr.Retryable() {
		t.Errorf("GenerateText() error = %v, want a retryable rate_limit_error with a request id", err)
	}
	if got := len(rt.Requests()); got != 1 {
		t.Errorf("sent %d requests, want 1", got)
	}
}
