package anthropic_test

import (
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func TestCache_marksEveryCacheablePartKind(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	call := ai.ToolCall{ID: "call_1", Name: "lookup", Input: []byte(`{"q":"x"}`)}
	result := ai.ToolResult{CallID: "call_1", Name: "lookup", Output: ai.ToolOutput{Text: "found"}}
	generate(t, srv, &ai.Call{
		Messages: []ai.Message{
			ai.UserParts(anthropic.CacheFor(ai.File{MediaType: "image/png", Data: []byte{1}}, anthropic.TTL1Hour), anthropic.Cache(ai.Text{Text: "doc"})),
			{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "ok"}, anthropic.Cache(call)}},
			{Role: ai.RoleTool, Content: ai.Parts{anthropic.Cache(result)}},
		},
		Tools: []ai.Tool{{Name: "first", InputSchema: []byte(`{"type":"object"}`)}},
	})
	equalJSON(t, `[
		{"role":"user","content":[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQ=="},"cache_control":{"type":"ephemeral","ttl":"1h"}},
			{"type":"text","text":"doc","cache_control":{"type":"ephemeral"}}]},
		{"role":"assistant","content":[
			{"type":"text","text":"ok"},
			{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"x"},"cache_control":{"type":"ephemeral"}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_1","content":"found","cache_control":{"type":"ephemeral"}}]}]`, []byte(field(t, srv, "messages")))
}

func TestCache_skipsPartsTheAPICannotCache(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	thinking := ai.Reasoning{Text: "hmm", Provider: ai.ProviderData{"anthropic": []byte(`{"block":{"type":"thinking","thinking":"hmm","signature":"s"}}`)}}
	redactedAsCustom := ai.Custom{Kind: "anthropic.redacted_thinking", Provider: ai.ProviderData{"anthropic": []byte(`{"block":{"type":"redacted_thinking","data":"d"}}`)}}
	generate(t, srv, &ai.Call{Messages: []ai.Message{
		ai.UserParts(anthropic.Cache(ai.Text{Text: ""}), ai.Text{Text: "hi"}),
		{Role: ai.RoleAssistant, Content: ai.Parts{anthropic.Cache(thinking), anthropic.Cache(redactedAsCustom), anthropic.Cache(ai.Text{Text: ""}), ai.Text{Text: "answer"}}},
		ai.User("more"),
	}})
	if raw := field(t, srv, "messages"); strings.Contains(raw, "cache_control") {
		t.Errorf("messages = %s, want no cache_control on thinking, redacted thinking or empty blocks", raw)
	}
}

func TestCache_keepsFileIDAndReplayBlock(t *testing.T) {
	f := anthropic.Cache(ai.File{MediaType: "application/pdf", Provider: anthropic.FileID("file_1")})
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: []ai.Message{ai.UserParts(f)}})
	equalJSON(t, `[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_1"},"cache_control":{"type":"ephemeral"}}]}]`, []byte(field(t, srv, "messages")))
}

func TestCacheTool_marksCustomAndHostedTools(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Tools: []ai.Tool{
			{Name: "plain", InputSchema: []byte(`{"type":"object"}`)},
			anthropic.CacheToolFor(anthropic.WebSearch(anthropic.WebSearchOptions{MaxUses: 2}), anthropic.TTL1Hour),
			anthropic.CacheTool(ai.Tool{Name: "last", Description: "d", InputSchema: []byte(`{"type":"object"}`)}),
		},
	})
	equalJSON(t, `[
		{"name":"plain","input_schema":{"type":"object"}},
		{"type":"web_search_20260318","name":"web_search","max_uses":2,"allowed_callers":["direct"],"cache_control":{"type":"ephemeral","ttl":"1h"}},
		{"name":"last","description":"d","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]`, []byte(field(t, srv, "tools")))
}

func TestCache_breakpointLimit(t *testing.T) {
	tests := []struct {
		name     string
		call     *ai.Call
		wantErr  string
		wantWarn string
	}{
		{
			name: "four breakpoints pass",
			call: &ai.Call{
				Instructions: []ai.Text{anthropic.CachedText("sys")},
				Tools:        []ai.Tool{anthropic.CacheTool(ai.Tool{Name: "t", InputSchema: []byte(`{"type":"object"}`)})},
				Messages:     []ai.Message{ai.UserParts(anthropic.Cache(ai.Text{Text: "a"}), anthropic.Cache(ai.Text{Text: "b"}))},
			},
		},
		{
			name: "top-level automatic caching counts",
			call: &ai.Call{
				Instructions: []ai.Text{anthropic.CachedText("sys")},
				Provider:     ai.Providers(anthropic.Options{CacheControl: &anthropic.CacheControl{}}),
				Messages:     []ai.Message{ai.UserParts(anthropic.Cache(ai.Text{Text: "a"}), anthropic.Cache(ai.Text{Text: "b"}), anthropic.Cache(ai.Text{Text: "c"}))},
			},
			wantErr: "5 cache breakpoints exceed the limit of 4",
		},
		{
			name: "tools system and messages together",
			call: &ai.Call{
				Instructions: []ai.Text{anthropic.CachedText("sys"), anthropic.CachedText("sys2")},
				Tools:        []ai.Tool{anthropic.CacheTool(ai.Tool{Name: "t", InputSchema: []byte(`{"type":"object"}`)})},
				Messages:     []ai.Message{ai.UserParts(anthropic.Cache(ai.Text{Text: "a"}), anthropic.Cache(ai.Text{Text: "b"}))},
			},
			wantErr: "5 cache breakpoints",
		},
		{
			name: "one hour after five minutes warns",
			call: &ai.Call{
				Instructions: []ai.Text{anthropic.CachedText("sys"), anthropic.CachedTextFor("long", anthropic.TTL1Hour)},
				Messages:     []ai.Message{ai.User("hi")},
			},
			wantWarn: "longer TTLs to come first",
		},
		{
			name: "one hour before five minutes is fine",
			call: &ai.Call{
				Instructions: []ai.Text{anthropic.CachedTextFor("long", anthropic.TTL1Hour), anthropic.CachedText("sys")},
				Messages:     []ai.Message{ai.User("hi")},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalMessage))
			resp, err := srv.provider().Model("claude-sonnet-5-5").Generate(t.Context(), tt.call)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) || srv.count() != 0 {
					t.Fatalf("Generate() error = %v with %d requests, want %q before sending", err, srv.count(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			got := strings.Join(warningsOf(resp), "\n")
			if (tt.wantWarn == "") != (got == "") || !strings.Contains(got, tt.wantWarn) {
				t.Errorf("warnings = %q, want %q", got, tt.wantWarn)
			}
		})
	}
}
