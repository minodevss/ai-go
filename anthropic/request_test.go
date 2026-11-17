package anthropic_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func generate(t *testing.T, srv *fakeServer, c *ai.Call, opts ...anthropic.Option) *ai.Response {
	t.Helper()
	resp, err := srv.provider(opts...).Model("claude-sonnet-5-5").Generate(t.Context(), c)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	return resp
}

func body(t *testing.T, srv *fakeServer) map[string]json.RawMessage {
	t.Helper()
	var out map[string]json.RawMessage
	if err := json.Unmarshal(srv.last(t).Body, &out); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	return out
}

func field(t *testing.T, srv *fakeServer, name string) string {
	t.Helper()
	return string(body(t, srv)[name])
}

func warningsOf(r *ai.Response) []string {
	var out []string
	for _, w := range r.Warnings {
		out = append(out, w.Message)
	}
	return out
}

func TestLanguageModel_basicRequest(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{
		Instructions:  []ai.Text{anthropic.CachedText("You are terse."), {Text: "Answer in English."}, {Text: ""}, anthropic.CachedTextFor("Long context.", anthropic.TTL1Hour)},
		Messages:      []ai.Message{ai.User("hi")},
		StopSequences: []string{"END"},
	})
	equalJSON(t, `{
		"model":"claude-sonnet-5-5","max_tokens":16000,
		"system":[
			{"type":"text","text":"You are terse.","cache_control":{"type":"ephemeral"}},
			{"type":"text","text":"Answer in English."},
			{"type":"text","text":"Long context.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"stop_sequences":["END"]}`, srv.last(t).Body)
	r := srv.last(t)
	if r.Path != "/v1/messages" || r.Header.Get("X-Api-Key") != "test-key" || r.Header.Get("Anthropic-Version") != "2026-10-21" || r.Header.Get("User-Agent") != "ai-go" {
		t.Errorf("path %q headers %v, want /v1/messages with api key, version and user agent", r.Path, r.Header)
	}
	if r.Header.Get("Anthropic-Beta") != "" || r.Header.Get("Authorization") != "" {
		t.Errorf("unexpected beta or authorization headers: %v", r.Header)
	}
	if resp.FinishReason != ai.FinishStop || resp.Usage.Input != 3 || resp.Usage.Output != 2 {
		t.Errorf("response = %+v, want stop with 3 input and 2 output tokens", resp)
	}
	data, ok := ai.ProviderValue[anthropic.ResponseData](resp.Provider, "anthropic")
	if !ok || data.ID != "msg_1" {
		t.Errorf("response provider data = %+v, want message id msg_1", data)
	}
}

func TestLanguageModel_maxTokens(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, MaxOutputTokens: 77})
	if got := field(t, srv, "max_tokens"); got != "77" {
		t.Errorf("max_tokens = %s, want 77", got)
	}
}

func TestLanguageModel_reasoning(t *testing.T) {
	tests := []struct {
		name         string
		reasoning    ai.ReasoningEffort
		opts         anthropic.Options
		wantThinking string
		wantOutput   string
	}{
		{name: "unset sends nothing"},
		{name: "none disables thinking", reasoning: ai.ReasoningNone, wantThinking: `{"type":"disabled"}`},
		{name: "minimal is low", reasoning: ai.ReasoningMinimal, wantThinking: `{"type":"adaptive","display":"summarized"}`, wantOutput: `{"effort":"low"}`},
		{name: "medium", reasoning: ai.ReasoningMedium, wantThinking: `{"type":"adaptive","display":"summarized"}`, wantOutput: `{"effort":"medium"}`},
		{name: "max", reasoning: ai.ReasoningMax, wantThinking: `{"type":"adaptive","display":"summarized"}`, wantOutput: `{"effort":"max"}`},
		{
			name:         "options override",
			reasoning:    ai.ReasoningHigh,
			opts:         anthropic.Options{Thinking: &anthropic.Thinking{Type: anthropic.ThinkingAdaptive, Display: anthropic.DisplayOmitted}, Effort: "xhigh"},
			wantThinking: `{"type":"adaptive","display":"omitted"}`,
			wantOutput:   `{"effort":"xhigh"}`,
		},
		{
			name:         "manual budget",
			opts:         anthropic.Options{Thinking: &anthropic.Thinking{Type: anthropic.ThinkingEnabled, BudgetTokens: 2048}},
			wantThinking: `{"type":"enabled","budget_tokens":2048}`,
		},
		{name: "between tools", opts: anthropic.Options{Thinking: &anthropic.Thinking{Type: anthropic.ThinkingBetweenTools}}, wantThinking: `{"type":"between_tools"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalMessage))
			generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Reasoning: tt.reasoning, Provider: ai.Providers(tt.opts)})
			if got := field(t, srv, "thinking"); got != tt.wantThinking {
				t.Errorf("thinking = %q, want %q", got, tt.wantThinking)
			}
			if got := field(t, srv, "output_config"); got != tt.wantOutput {
				t.Errorf("output_config = %q, want %q", got, tt.wantOutput)
			}
		})
	}
}

func TestLanguageModel_options(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Tools:    []ai.Tool{ai.NewClientTool[weatherInput]("get_weather", "Weather")},
		Provider: ai.Providers(anthropic.Options{
			CacheControl:           &anthropic.CacheControl{TTL: anthropic.TTL1Hour},
			UserID:                 "user-123",
			ServiceTier:            "standard_only",
			DisableParallelToolUse: true,
			Container:              "container_abc",
			Betas:                  []string{"context-management-2025-06-27", "extra"},
			MCPServers: []anthropic.MCPServer{
				{Name: "calendar", URL: "https://mcp.example.com/sse", AuthorizationToken: "tok", AllowedTools: []string{"search_events"}},
				{Name: "wiki", URL: "https://wiki.example.com/mcp", DisabledTools: []string{"delete_page"}},
			},
		}),
	}, anthropic.WithBeta("provider-beta", "extra"))
	got := srv.last(t)
	equalJSON(t, `{
		"model":"claude-sonnet-5-5","max_tokens":16000,
		"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"cache_control":{"type":"ephemeral","ttl":"1h"},
		"metadata":{"user_id":"user-123"},
		"service_tier":"standard_only",
		"container":"container_abc",
		"tool_choice":{"type":"auto","disable_parallel_tool_use":true},
		"mcp_servers":[
			{"type":"url","url":"https://mcp.example.com/sse","name":"calendar","authorization_token":"tok"},
			{"type":"url","url":"https://wiki.example.com/mcp","name":"wiki"}],
		"tools":[
			{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"city":{"type":"string","description":"city name"}},"required":["city"],"additionalProperties":false}},
			{"type":"mcp_toolset","mcp_server_name":"calendar","default_config":{"enabled":false},"configs":{"search_events":{"enabled":true}}},
			{"type":"mcp_toolset","mcp_server_name":"wiki","configs":{"delete_page":{"enabled":false}}}]}`, got.Body)
	if beta := got.Header.Get("Anthropic-Beta"); beta != "provider-beta,extra,context-management-2025-06-27,mcp-client-2025-11-20" {
		t.Errorf("anthropic-beta = %q, want provider, extra, options and MCP betas without duplicates", beta)
	}
}

func TestLanguageModel_headers(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Headers: http.Header{"Anthropic-Beta": {"call-beta"}, "X-Trace": {"1"}}},
		anthropic.WithHeaders(http.Header{"Anthropic-Version": {"2023-06-01"}, "Anthropic-Beta": {"header-beta"}, "X-Team": {"core"}}))
	h := srv.last(t).Header
	if h.Get("Anthropic-Version") != "2023-06-01" || h.Get("X-Team") != "core" || h.Get("X-Trace") != "1" || h.Get("Anthropic-Beta") != "header-beta,call-beta" {
		t.Errorf("headers = %v, want overridden version, team, trace and merged betas", h)
	}
}

func TestLanguageModel_unsupportedSettings(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{
		Messages:         []ai.Message{ai.User("hi")},
		Temperature:      new(0.2),
		TopP:             new(0.9),
		TopK:             5,
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.1),
		Seed:             new(7),
	})
	got := strings.Join(warningsOf(resp), "\n")
	for _, name := range []string{"temperature", "topP", "topK", "presencePenalty", "frequencyPenalty", "seed"} {
		if !strings.Contains(got, name) {
			t.Errorf("warnings do not mention %s:\n%s", name, got)
		}
	}
	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if _, ok := body(t, srv)[key]; ok {
			t.Errorf("request carries %s, which API version 2026-10-21 rejects", key)
		}
	}
}

func TestLanguageModel_tools(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather")
	strict := ai.Tool{
		Name:        "lookup",
		Description: "Lookup",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer","minimum":1},"tags":{"type":"array","items":{"type":"string"},"minItems":1}},"required":["n"]}`),
		Strict:      true,
	}
	bare := ai.Tool{Name: "ping"}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Tools: []ai.Tool{
			weather, strict, bare,
			anthropic.WebSearch(anthropic.WebSearchOptions{MaxUses: 3, AllowedDomains: []string{"go.dev"}, UserLocation: &anthropic.UserLocation{Country: "KR", City: "Seoul"}, AllowedCallers: []string{"direct"}}),
			anthropic.WebFetch(anthropic.WebFetchOptions{Version: "web_fetch_20250910", Citations: true, MaxContentTokens: 5000, BlockedDomains: []string{"x.com"}}),
			anthropic.CodeExecution(),
		},
	})
	equalJSON(t, `[
		{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"city":{"type":"string","description":"city name"}},"required":["city"],"additionalProperties":false}},
		{"name":"lookup","description":"Lookup","strict":true,"input_schema":{"type":"object","properties":{"n":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"},"minItems":1}},"required":["n"],"additionalProperties":false}},
		{"name":"ping","input_schema":{"type":"object","properties":{}}},
		{"type":"web_search_20260318","name":"web_search","max_uses":3,"allowed_domains":["go.dev"],"user_location":{"type":"approximate","city":"Seoul","country":"KR"},"allowed_callers":["direct"]},
		{"type":"web_fetch_20250910","name":"web_fetch","blocked_domains":["x.com"],"citations":{"enabled":true},"max_content_tokens":5000},
		{"type":"code_execution_20260521","name":"code_execution"}]`, body(t, srv)["tools"])
	if got := strings.Join(warningsOf(resp), "\n"); !strings.Contains(got, `"minimum"`) {
		t.Errorf("warnings = %q, want the removed minimum keyword", got)
	}
}

func TestWebSearch_defaultsToDirectCallers(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Tools: []ai.Tool{
			anthropic.WebSearch(anthropic.WebSearchOptions{}),
			anthropic.WebSearch(anthropic.WebSearchOptions{AllowedCallers: []string{"code_execution_20260120"}}),
		},
	})
	equalJSON(t, `[
		{"type":"web_search_20260318","name":"web_search","allowed_callers":["direct"]},
		{"type":"web_search_20260318","name":"web_search","allowed_callers":["code_execution_20260120"]}]`, body(t, srv)["tools"])
}

func TestLanguageModel_toolChoice(t *testing.T) {
	tools := []ai.Tool{ai.NewClientTool[weatherInput]("get_weather", "Weather"), anthropic.WebSearch(anthropic.WebSearchOptions{})}
	tests := []struct {
		name   string
		choice ai.ToolChoice
		tools  []ai.Tool
		want   string
	}{
		{name: "auto", choice: ai.ToolChoice{}, tools: tools},
		{name: "required", choice: ai.ToolRequired(), tools: tools, want: `{"type":"any"}`},
		{name: "none", choice: ai.ToolNone(), tools: tools, want: `{"type":"none"}`},
		{name: "named", choice: ai.ToolNamed("get_weather"), tools: tools, want: `{"type":"tool","name":"get_weather"}`},
		{name: "named hosted tool", choice: ai.ToolNamed("web_search"), tools: tools, want: `{"type":"tool","name":"web_search"}`},
		{name: "no tools", choice: ai.ToolRequired()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalMessage))
			generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Tools: tt.tools, ToolChoice: tt.choice})
			if got := field(t, srv, "tool_choice"); got != tt.want {
				t.Errorf("tool_choice = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLanguageModel_structuredOutput(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string","minLength":1},"age":{"type":"integer"}},"required":["name","age"]}`)
	spec := &ai.OutputSpec{Name: "person", Description: "A person", Schema: schema}
	const fittedSchema = `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}`

	t.Run("native output_config.format", func(t *testing.T) {
		srv := newFakeServer(t, jsonReply(strings.Replace(minimalMessage, `"hi"`, `"{\"name\":\"Ada\",\"age\":36}"`, 1)))
		resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Output: spec, Reasoning: ai.ReasoningLow})
		equalJSON(t, `{"effort":"low","format":{"type":"json_schema","schema":`+fittedSchema+`}}`, body(t, srv)["output_config"])
		if _, ok := body(t, srv)["tools"]; ok {
			t.Error("native structured output must not add a tool")
		}
		if text, ok := resp.Content[0].(ai.Text); !ok || text.Text != `{"name":"Ada","age":36}` {
			t.Errorf("content = %#v, want the JSON text", resp.Content)
		}
		if !strings.Contains(strings.Join(warningsOf(resp), "\n"), `"minLength"`) {
			t.Errorf("warnings = %v, want the dropped minLength keyword", warningsOf(resp))
		}
	})

	forcedReply := `{"id":"msg_f","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_f","name":"person","input":{"name":"Ada","age":36}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":9}}`
	t.Run("forced tool when requested", func(t *testing.T) {
		srv := newFakeServer(t, jsonReply(forcedReply))
		resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Output: spec, Provider: ai.Providers(anthropic.Options{OutputViaTool: true})})
		equalJSON(t, `[{"name":"person","description":"A person","input_schema":`+fittedSchema+`}]`, body(t, srv)["tools"])
		if got := field(t, srv, "tool_choice"); got != `{"type":"tool","name":"person"}` {
			t.Errorf("tool_choice = %s, want the forced tool", got)
		}
		if _, ok := body(t, srv)["output_config"]; ok {
			t.Error("forced tool output must not send output_config")
		}
		if text, ok := resp.Content[0].(ai.Text); len(resp.Content) != 1 || !ok || text.Text != `{"name":"Ada","age":36}` {
			t.Errorf("content = %#v, want the tool input as text", resp.Content)
		}
		if resp.FinishReason != ai.FinishStop {
			t.Errorf("finish = %q, want stop for a forced output tool", resp.FinishReason)
		}
	})

	t.Run("forced tool replays as plain text", func(t *testing.T) {
		srv := newFakeServer(t, jsonReply(forcedReply))
		first := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Output: spec, Provider: ai.Providers(anthropic.Options{OutputViaTool: true})})
		history := []ai.Message{ai.User("hi"), {Role: ai.RoleAssistant, Content: first.Content, Provider: first.Provider}, ai.User("again")}
		generate(t, srv, &ai.Call{Messages: history})
		equalJSON(t, `[
			{"role":"user","content":[{"type":"text","text":"hi"}]},
			{"role":"assistant","content":[{"type":"text","text":"{\"name\":\"Ada\",\"age\":36}"}]},
			{"role":"user","content":[{"type":"text","text":"again"}]}]`, body(t, srv)["messages"])
	})

	t.Run("citations force a tool", func(t *testing.T) {
		srv := newFakeServer(t, jsonReply(forcedReply))
		resp := generate(t, srv, &ai.Call{
			Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "text/plain", Data: []byte("doc"), Citations: true}, ai.Text{Text: "extract"})},
			Output:   spec,
		})
		if got := field(t, srv, "tool_choice"); got != `{"type":"tool","name":"person"}` {
			t.Errorf("tool_choice = %s, want the forced tool", got)
		}
		if !strings.Contains(strings.Join(warningsOf(resp), "\n"), "citations cannot be combined") {
			t.Errorf("warnings = %v, want the citations notice", warningsOf(resp))
		}
	})
}

func TestLanguageModel_userContent(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.UserParts(
		anthropic.CachedText("see attachments"),
		ai.File{MediaType: "image/png", Data: png},
		ai.File{MediaType: "image/jpeg", URL: "https://example.com/a.jpg"},
		ai.File{MediaType: "image/webp", Provider: anthropic.FileID("file_img")},
		ai.File{MediaType: "application/pdf", Filename: "spec.pdf", Data: []byte("%PDF"), Citations: true},
		ai.File{MediaType: "application/pdf", URL: "https://example.com/b.pdf"},
		ai.File{MediaType: "application/pdf", Provider: anthropic.FileID("file_pdf")},
		ai.File{MediaType: "text/plain", Filename: "notes.txt", Data: []byte("notes")},
		ai.Text{Text: ""},
	)}})
	equalJSON(t, `[{"role":"user","content":[
		{"type":"text","text":"see attachments","cache_control":{"type":"ephemeral"}},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw=="}},
		{"type":"image","source":{"type":"url","url":"https://example.com/a.jpg"}},
		{"type":"image","source":{"type":"file","file_id":"file_img"}},
		{"type":"document","title":"spec.pdf","citations":{"enabled":true},"source":{"type":"base64","media_type":"application/pdf","data":"JVBERg=="}},
		{"type":"document","citations":{"enabled":true},"source":{"type":"url","url":"https://example.com/b.pdf"}},
		{"type":"document","citations":{"enabled":true},"source":{"type":"file","file_id":"file_pdf"}},
		{"type":"document","title":"notes.txt","citations":{"enabled":true},"source":{"type":"text","media_type":"text/plain","data":"notes"}}]}]`, body(t, srv)["messages"])
	if !strings.Contains(strings.Join(warningsOf(resp), "\n"), "all or none") {
		t.Errorf("warnings = %v, want the all-or-none citations notice", warningsOf(resp))
	}
}

func TestLanguageModel_unsupportedFile(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	_, err := srv.provider().Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "audio/mp3", Data: []byte("x")})}})
	if err == nil || !strings.Contains(err.Error(), "unsupported file media type") {
		t.Errorf("Generate() error = %v, want an unsupported media type error", err)
	}
	if srv.count() != 0 {
		t.Errorf("sent %d requests, want none", srv.count())
	}
}

func TestLanguageModel_toolResults(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	history := []ai.Message{
		ai.User("go"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.ToolCall{ID: "toolu_a", Name: "a", Input: json.RawMessage(`{"x":1}`)},
			ai.ToolCall{ID: "toolu_b", Name: "b"},
			ai.ToolCall{ID: "toolu_c", Name: "c", Input: json.RawMessage(`"not an object"`)},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{
			ai.ToolResult{CallID: "toolu_b", Name: "b", Output: ai.ToolOutput{JSON: json.RawMessage(`{"ok":true}`)}},
			ai.ToolResult{CallID: "toolu_a", Name: "a", Output: ai.ToolOutput{Text: "boom"}, IsError: true},
			ai.ToolResult{CallID: "toolu_zzz", Name: "ghost", Output: ai.ToolOutput{Text: "orphan"}},
		}},
		ai.User("thanks"),
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "toolu_d", Name: "shot"}}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "toolu_d", Name: "shot", Output: ai.ToolOutput{Parts: ai.Parts{ai.Text{Text: "screen"}, ai.File{MediaType: "image/png", Data: png}}}}}},
	}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{Messages: history})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"go"}]},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"toolu_a","name":"a","input":{"x":1}},
			{"type":"tool_use","id":"toolu_b","name":"b","input":{}},
			{"type":"tool_use","id":"toolu_c","name":"c","input":{}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_c","content":"error: this tool call has no result","is_error":true},
			{"type":"tool_result","tool_use_id":"toolu_b","content":"{\"ok\":true}"},
			{"type":"tool_result","tool_use_id":"toolu_a","content":"boom","is_error":true},
			{"type":"text","text":"thanks"}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_d","name":"shot","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_d","content":[
			{"type":"text","text":"screen"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw=="}}]}]}]`, body(t, srv)["messages"])
	if !strings.Contains(strings.Join(warningsOf(resp), "\n"), "without a matching tool call were dropped") {
		t.Errorf("warnings = %v, want the orphan result notice", warningsOf(resp))
	}
}

func TestLanguageModel_unansweredCallAtEnd(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: []ai.Message{
		ai.User("go"),
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "toolu_a", Name: "a"}}},
	}})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"go"}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_a","name":"a","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"error: this tool call has no result","is_error":true}]}]`, body(t, srv)["messages"])
}

func TestLanguageModel_assistantReplay(t *testing.T) {
	cite := json.RawMessage(`{"type":"char_location","cited_text":"x","document_index":0,"document_title":"T","start_char_index":0,"end_char_index":1,"file_id":null}`)
	thinking := json.RawMessage(`{"type":"thinking","thinking":"hm","signature":"SIG"}`)
	redacted := json.RawMessage(`{"type":"redacted_thinking","data":"DATA"}`)
	custom := json.RawMessage(`{"type":"compaction","content":"summary"}`)
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Reasoning{Text: "hm", Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"block":` + string(thinking) + `}`)}},
			ai.Reasoning{Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"block":` + string(redacted) + `}`)}},
			ai.Reasoning{Text: "foreign", Provider: ai.ProviderData{"openai": json.RawMessage(`{"id":"rs_1"}`)}},
			ai.Text{Text: ""},
			ai.Text{Text: "cited", Citations: []ai.Citation{{Kind: "char", Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"block":` + string(cite) + `}`)}}, {Kind: "url", URL: "https://x"}}},
			ai.Custom{Kind: "anthropic.compaction", Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"block":` + string(custom) + `}`)}},
			ai.Custom{Kind: "openai.x"},
		}},
		ai.User("next"),
	}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: history})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"q"}]},
		{"role":"assistant","content":[`+string(thinking)+`,`+string(redacted)+`,{"type":"text","text":"cited","citations":[`+string(cite)+`]},`+string(custom)+`]},
		{"role":"user","content":[{"type":"text","text":"next"}]}]`, body(t, srv)["messages"])
}

func TestLanguageModel_consecutiveTurnsMerge(t *testing.T) {
	srv := newFakeServer(t, jsonReply(minimalMessage))
	generate(t, srv, &ai.Call{Messages: []ai.Message{
		ai.User("a"), ai.User("b"),
		ai.Assistant("c"), ai.Assistant("d"),
		{Role: ai.RoleUser, Content: ai.Parts{ai.ToolApprovalResponse{ApprovalID: "x", CallID: "y", Approved: true}}},
		ai.User("e"),
	}})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
		{"role":"assistant","content":[{"type":"text","text":"c"},{"type":"text","text":"d"}]},
		{"role":"user","content":[{"type":"text","text":"e"}]}]`, body(t, srv)["messages"])
}

func TestLanguageModel_containerReuse(t *testing.T) {
	assistant := func(expires string) ai.Message {
		return ai.Message{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "ran"}}, Provider: ai.Providers().WithRaw("anthropic", json.RawMessage(`{"id":"msg_1","container":{"id":"container_live","expiresAt":"`+expires+`"}}`))}
	}
	tests := []struct {
		name  string
		tools []ai.Tool
		msgs  []ai.Message
		opts  anthropic.Options
		want  string
	}{
		{name: "reused", tools: []ai.Tool{anthropic.CodeExecution()}, msgs: []ai.Message{ai.User("a"), assistant("2099-01-01T00:00:00Z"), ai.User("b")}, want: `"container_live"`},
		{name: "expired", tools: []ai.Tool{anthropic.CodeExecution()}, msgs: []ai.Message{ai.User("a"), assistant("2001-01-01T00:00:00Z"), ai.User("b")}},
		{name: "no hosted tool", msgs: []ai.Message{ai.User("a"), assistant("2099-01-01T00:00:00Z"), ai.User("b")}},
		{name: "option wins", tools: []ai.Tool{anthropic.CodeExecution()}, msgs: []ai.Message{ai.User("a"), assistant("2099-01-01T00:00:00Z"), ai.User("b")}, opts: anthropic.Options{Container: "container_opt"}, want: `"container_opt"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(minimalMessage))
			generate(t, srv, &ai.Call{Messages: tt.msgs, Tools: tt.tools, Provider: ai.Providers(tt.opts)})
			if got := field(t, srv, "container"); got != tt.want {
				t.Errorf("container = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLanguageModel_serverToolsReplayVerbatim(t *testing.T) {
	reply := `{"id":"msg_s","type":"message","role":"assistant","model":"m","content":[
		{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"},"caller":{"type":"direct"}},
		{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://a","title":"A","encrypted_content":"ENC","page_age":null}]},
		{"type":"text","text":"done","citations":[{"type":"web_search_result_location","url":"https://a","title":"A","encrypted_index":"IDX","cited_text":"c"}]}],
		"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`
	srv := newFakeServer(t, jsonReply(reply))
	first := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("q")}})
	history := []ai.Message{ai.User("q"), {Role: ai.RoleAssistant, Content: first.Content, Provider: first.Provider}, ai.User("more")}
	generate(t, srv, &ai.Call{Messages: history})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"q"}]},
		{"role":"assistant","content":[
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"},"caller":{"type":"direct"}},
			{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://a","title":"A","encrypted_content":"ENC","page_age":null}]},
			{"type":"text","text":"done","citations":[{"type":"web_search_result_location","url":"https://a","title":"A","encrypted_index":"IDX","cited_text":"c"}]}]},
		{"role":"user","content":[{"type":"text","text":"more"}]}]`, body(t, srv)["messages"])
	call, ok := first.Content[0].(ai.ToolCall)
	if !ok || !call.ProviderExecuted || call.Name != "web_search" {
		t.Errorf("first part = %#v, want a provider-executed web_search call", first.Content[0])
	}
	res, ok := first.Content[1].(ai.ToolResult)
	if !ok || !res.ProviderExecuted || res.CallID != "srvtoolu_1" || res.Name != "web_search" || res.IsError {
		t.Errorf("second part = %#v, want the provider-executed result of srvtoolu_1", first.Content[1])
	}
	text, ok := first.Content[2].(ai.Text)
	if !ok || len(text.Citations) != 1 || text.Citations[0].Kind != "web" || text.Citations[0].URL != "https://a" || text.Citations[0].CitedText != "c" {
		t.Errorf("third part = %#v, want text with one web citation", first.Content[2])
	}
}

func TestLanguageModel_responseBlocks(t *testing.T) {
	reply := `{"id":"msg_r","type":"message","role":"assistant","model":"m","content":[
		{"type":"mcp_tool_use","id":"mcptoolu_1","name":"echo","server_name":"srv","input":{"a":1}},
		{"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":true,"content":[{"type":"text","text":"bad"}]},
		{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_old","content":{"type":"web_fetch_tool_result_error","error_code":"unavailable"}},
		{"type":"container_upload","file_id":"file_1"},
		{"type":"text","text":"","citations":null},
		{"type":"tool_use","id":"toolu_1","name":"get_weather","input":null}],
		"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`
	srv := newFakeServer(t, jsonReply(reply))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("q")}})
	if len(resp.Content) != 5 {
		t.Fatalf("content has %d parts, want 5 (empty text dropped): %#v", len(resp.Content), resp.Content)
	}
	if c, ok := resp.Content[0].(ai.ToolCall); !ok || c.Name != "mcp.srv.echo" || !c.ProviderExecuted {
		t.Errorf("part 0 = %#v, want the provider-executed MCP call", resp.Content[0])
	}
	if r, ok := resp.Content[1].(ai.ToolResult); !ok || !r.IsError || r.Name != "mcp.srv.echo" || string(r.Output.JSON) != `[{"type":"text","text":"bad"}]` {
		t.Errorf("part 1 = %#v, want an error MCP result", resp.Content[1])
	}
	if r, ok := resp.Content[2].(ai.ToolResult); !ok || !r.IsError || r.Name != "web_fetch" {
		t.Errorf("part 2 = %#v, want an error web_fetch result named from its type", resp.Content[2])
	}
	if c, ok := resp.Content[3].(ai.Custom); !ok || c.Kind != "anthropic.container_upload" {
		t.Errorf("part 3 = %#v, want a custom container_upload part", resp.Content[3])
	}
	if c, ok := resp.Content[4].(ai.ToolCall); !ok || string(c.Input) != "{}" || c.ProviderExecuted {
		t.Errorf("part 4 = %#v, want a client call with empty input", resp.Content[4])
	}
	if resp.FinishReason != ai.FinishToolCalls {
		t.Errorf("finish = %q, want tool-calls", resp.FinishReason)
	}
}

func TestLanguageModel_usage(t *testing.T) {
	reply := `{"id":"m","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","stop_sequence":null,
		"usage":{"input_tokens":10,"cache_creation_input_tokens":200,"cache_read_input_tokens":3000,"output_tokens":50,"output_tokens_details":{"thinking_tokens":40},"service_tier":"priority","inference_geo":"us","server_tool_use":{"web_search_requests":2}}}`
	srv := newFakeServer(t, jsonReply(reply))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("q")}})
	want := ai.Usage{Input: 3210, Output: 50, CacheRead: 3000, CacheWrite: 200, Reasoning: 40}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	data, _ := ai.ProviderValue[anthropic.ResponseData](resp.Provider, "anthropic")
	if data.ServiceTier != "priority" || data.InferenceGeo != "us" || string(data.ServerToolUse) != `{"web_search_requests":2}` {
		t.Errorf("response data = %+v, want tier, geo and server tool usage", data)
	}
}

func TestLanguageModel_unsupportedToolOutput(t *testing.T) {
	history := []ai.Message{
		ai.User("go"),
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: "toolu_a", Name: "a"}}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "toolu_a", Name: "a", Output: ai.ToolOutput{Parts: ai.Parts{ai.File{MediaType: "audio/mp3", Data: []byte("x")}}}}}},
	}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{Messages: history})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"go"}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_a","name":"a","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"error: unsupported tool output","is_error":true}]}]`, body(t, srv)["messages"])
	if !strings.Contains(strings.Join(warningsOf(resp), "\n"), "cannot take as tool output") {
		t.Errorf("warnings = %v, want the unsupported output notice", warningsOf(resp))
	}
}

func TestLanguageModel_strictSchemaCombinators(t *testing.T) {
	tool := ai.Tool{
		Name: "pick",
		InputSchema: json.RawMessage(`{"type":"object","properties":{
			"v":{"oneOf":[{"type":"string","maxLength":3},{"type":"object","properties":{"k":{"type":"string"}}}]},
			"list":{"type":"array","items":{"type":"object","properties":{"x":{"type":["integer","null"]}},"additionalProperties":true},"minItems":2}},
			"$defs":{"d":{"type":"object","properties":{"y":{"type":"string"}}}}}`),
		Strict: true,
	}
	srv := newFakeServer(t, jsonReply(minimalMessage))
	resp := generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{tool}})
	var tools []struct {
		Schema json.RawMessage `json:"input_schema"`
	}
	if err := json.Unmarshal(body(t, srv)["tools"], &tools); err != nil {
		t.Fatal(err)
	}
	equalJSON(t, `{"type":"object","properties":{
		"v":{"anyOf":[{"type":"string"},{"type":"object","properties":{"k":{"type":"string"}},"additionalProperties":false}]},
		"list":{"type":"array","items":{"type":"object","properties":{"x":{"type":["integer","null"]}},"additionalProperties":false}}},
		"$defs":{"d":{"type":"object","properties":{"y":{"type":"string"}},"additionalProperties":false}},"additionalProperties":false}`, tools[0].Schema)
	got := strings.Join(warningsOf(resp), "\n")
	for _, want := range []string{`"oneOf"`, `"maxLength"`, `"minItems"`, "changed from true to false"} {
		if !strings.Contains(got, want) {
			t.Errorf("warnings do not mention %s:\n%s", want, got)
		}
	}
}
