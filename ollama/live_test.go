//go:build live

package ollama_test

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ollama"
)

func pulledModel(t *testing.T, host string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, host+"/api/tags", nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Skipf("no Ollama server at %s: %v", host, err)
	}
	defer resp.Body.Close()
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Skipf("unreadable /api/tags: %v", err)
	}
	if want := os.Getenv("OLLAMA_LIVE_MODEL"); want != "" {
		return want
	}
	if len(tags.Models) == 0 {
		t.Skip("no model is pulled; this test never pulls one")
	}
	return tags.Models[0].Name
}

func TestLive_generate(t *testing.T) {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	p := ollama.New(ollama.WithBaseURL(host))
	model := pulledModel(t, host)
	res, err := ai.GenerateText(t.Context(), p.Model(model), ai.TextRequest{
		Messages:        []ai.Message{ai.User("Reply with one word: hello.")},
		MaxOutputTokens: 16,
		MaxRetries:      new(0),
	})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Usage.Output == 0 {
		t.Errorf("usage = %+v, want output tokens", res.Usage)
	}
}
