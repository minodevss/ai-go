//go:build live

package openai_test

import (
	"errors"
	"net/http"
	"os"
	"regexp"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
	"github.com/minodevss/ai-go/openai"
)

var identifiers = regexp.MustCompile(`"(safety_identifier|prompt_cache_key|user)":"[^"]+"`)

func liveProvider(t *testing.T, name string, key string) *openai.Provider {
	t.Helper()
	base := os.Getenv("GW_BASE_URL")
	if base == "" {
		t.Skip("GW_BASE_URL is not set")
	}
	client := &http.Client{}
	if record.Updating() {
		rt := record.New(t, "testdata/live/"+name+".json", record.WithBodyRewrite(func(b []byte) []byte {
			return identifiers.ReplaceAll(b, []byte(`"$1":"REDACTED"`))
		}))
		client = &http.Client{Transport: rt}
	}
	return openai.New(openai.WithAPIKey(key), openai.WithBaseURL(base+"/v1"), openai.WithHTTPClient(client))
}

func TestLive_roundTrip(t *testing.T) {
	key := os.Getenv("GW_API_KEY")
	if key == "" {
		t.Skip("GW_API_KEY is not set")
	}
	runRoundTrip(t, liveProvider(t, "roundtrip", key).Model("gpt-6-luna"))
}

func TestLive_authError(t *testing.T) {
	p := liveProvider(t, "autherror", "invalid-key")
	_, err := ai.GenerateText(t.Context(), p.Model("gpt-6-luna"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("GenerateText() error = %v, want an authentication error", err)
	}
}
