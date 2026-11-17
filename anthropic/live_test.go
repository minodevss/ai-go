//go:build live

package anthropic_test

import (
	"errors"
	"net/http"
	"os"
	"regexp"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
	"github.com/minodevss/ai-go/anthropic"
)

var identifiers = regexp.MustCompile(`"(user_id)":"[^"]+"`)

func liveProvider(t *testing.T, name string, key string) *anthropic.Provider {
	t.Helper()
	base := os.Getenv("GW_BASE_URL")
	if base == "" {
		t.Skip("GW_BASE_URL is not set")
	}
	client := &http.Client{}
	if record.Updating() {
		rt := record.New(t, "testdata/live/"+name+".json", record.WithRedactedHeaders("anthropic-organization-id", "anthropic-workspace-id", "x-gw-account", "x-gw-reason", "traceresponse", "cf-ray"), record.WithBodyRewrite(func(b []byte) []byte {
			return identifiers.ReplaceAll(b, []byte(`"$1":"REDACTED"`))
		}))
		client = &http.Client{Transport: rt}
	}
	return anthropic.New(anthropic.WithAPIKey(key), anthropic.WithBaseURL(base), anthropic.WithHTTPClient(client))
}

func TestLive_roundTrip(t *testing.T) {
	key := os.Getenv("GW_API_KEY")
	if key == "" {
		t.Skip("GW_API_KEY is not set")
	}
	runRoundTrip(t, liveProvider(t, "roundtrip", key).Model("claude-haiku-5-5"))
}

func TestLive_authError(t *testing.T) {
	p := liveProvider(t, "autherror", "invalid-key")
	_, err := ai.GenerateText(t.Context(), p.Model("claude-haiku-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	if !errors.Is(err, ai.ErrAuth) {
		t.Errorf("GenerateText() error = %v, want an authentication error", err)
	}
}
