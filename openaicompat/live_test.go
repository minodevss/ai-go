//go:build live

package openaicompat_test

import (
	"os"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openaicompat"
)

func liveModel(t *testing.T) ai.LanguageModel {
	t.Helper()
	base, key := os.Getenv("GW_BASE_URL"), os.Getenv("GW_API_KEY")
	if base == "" || key == "" {
		t.Skip("GW_BASE_URL and GW_API_KEY are not set")
	}
	return openaicompat.New(base+"/v1", openaicompat.WithAPIKey(key), openaicompat.WithName("gateway")).Model("gpt-6-luna")
}

func TestLive_stream(t *testing.T) {
	res := ai.StreamText(t.Context(), liveModel(t), ai.TextRequest{
		Messages:        []ai.Message{ai.User("Reply with the single word: pong")},
		MaxOutputTokens: 200,
		MaxSteps:        1,
		MaxRetries:      new(0),
	})
	got, err := res.Result()
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if !strings.Contains(strings.ToLower(got.Text), "pong") {
		t.Errorf("Text = %q, want it to contain pong", got.Text)
	}
	if got.Usage.Input == 0 || got.Usage.Output == 0 {
		t.Errorf("Usage = %+v, want input and output tokens", got.Usage)
	}
}
