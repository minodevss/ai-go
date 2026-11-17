package cohere_test

import (
	"net/http"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
)

func TestWithHeaders(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalChat))
	p := srv.provider(cohere.WithHeaders(http.Header{"X-Team": {"a"}}))
	_, err := p.Model("command-a-03-2025").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}, Headers: http.Header{"X-Call": {"b"}}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	got := srv.last(t).Header
	if got.Get("X-Team") != "a" || got.Get("X-Call") != "b" || got.Get("Authorization") != "Bearer test-key" {
		t.Errorf("headers = %v, want X-Team, X-Call and the bearer key", got)
	}
}
