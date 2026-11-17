package openai_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

const chatMinimal = `{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`

func TestChatModel_request(t *testing.T) {
	longID := strings.Repeat("a", 60)
	tests := []struct {
		name string
		call ai.Call
		want string
	}{
		{
			name: "settings",
			call: ai.Call{
				Instructions:     ai.Texts("Be brief."),
				Messages:         []ai.Message{ai.User("hi")},
				MaxOutputTokens:  50,
				Temperature:      new(0.2),
				TopP:             new(0.8),
				PresencePenalty:  new(0.1),
				FrequencyPenalty: new(0.2),
				Seed:             new(7),
				StopSequences:    []string{"END"},
				Reasoning:        ai.ReasoningLow,
				Provider:         ai.Providers(openai.Options{Store: new(false), ServiceTier: "flex", ParallelToolCalls: new(false), PromptCacheKey: "k", User: "u", TextVerbosity: "low"}),
			},
			want: `{"model":"m","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"hi"}],"max_completion_tokens":50,"temperature":0.2,"top_p":0.8,
			"presence_penalty":0.1,"frequency_penalty":0.2,"seed":7,"stop":["END"],"reasoning_effort":"low","store":false,"service_tier":"flex","parallel_tool_calls":false,
			"prompt_cache_key":"k","user":"u","verbosity":"low"}`,
		},
		{
			name: "multimodal user",
			call: ai.Call{Messages: []ai.Message{ai.UserParts(
				ai.Text{Text: "look"},
				ai.File{MediaType: "image/png", URL: "https://x.test/i.png"},
				ai.File{MediaType: "audio/mpeg", Data: []byte("a")},
				ai.File{MediaType: "application/pdf", Data: []byte("p"), Filename: "a.pdf"},
				ai.File{MediaType: "application/pdf", Provider: openai.FileID("file-1")},
			)}},
			want: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://x.test/i.png"}},
			{"type":"input_audio","input_audio":{"data":"YQ==","format":"mp3"}},{"type":"file","file":{"file_data":"data:application/pdf;base64,cA==","filename":"a.pdf"}},{"type":"file","file":{"file_id":"file-1"}}]}]}`,
		},
		{
			name: "tools and history",
			call: ai.Call{
				Messages: []ai.Message{
					ai.User("q"),
					{Role: ai.RoleAssistant, Content: ai.Parts{
						ai.Reasoning{Text: "dropped foreign"},
						ai.Reasoning{Text: "kept", Provider: ai.ProviderData{"openai": json.RawMessage(`{"field":"reasoning_content"}`)}},
						ai.Text{Text: "calling"},
						ai.ToolCall{ID: "call_1", Name: "f", Input: json.RawMessage(`{"a":1}`)},
						ai.ToolCall{ID: longID, Name: "g"},
						ai.ToolCall{ID: "call_3", Name: "h"},
					}},
					{Role: ai.RoleTool, Content: ai.Parts{
						ai.ToolResult{CallID: "call_1", Name: "f", Output: ai.ToolOutput{Text: "r1"}},
						ai.ToolResult{CallID: longID, Name: "g", Output: ai.ToolOutput{JSON: json.RawMessage(`{"x":1}`)}},
					}},
				},
				Tools:      []ai.Tool{{Name: "f", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}, openai.WebSearch(openai.WebSearchOptions{})},
				ToolChoice: ai.ToolNamed("f"),
			},
			want: `{"model":"m","messages":[{"role":"user","content":"q"},
			{"role":"assistant","content":"calling","reasoning_content":"kept","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}},
				{"id":"call_AAAA","type":"function","function":{"name":"g","arguments":"{}"}},
				{"id":"call_3","type":"function","function":{"name":"h","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_3","content":"error: this tool call has no result"},
			{"role":"tool","tool_call_id":"call_1","content":"r1"},
			{"role":"tool","tool_call_id":"call_AAAA","content":"{\"x\":1}"}],
			"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"},"strict":false}}],"tool_choice":{"type":"function","function":{"name":"f"}}}`,
		},
		{
			name: "structured output",
			call: ai.Call{Messages: []ai.Message{ai.User("hi")}, Output: &ai.OutputSpec{Name: "out", Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)}},
			want: `{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"out","strict":true,
			"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(chatMinimal))
			if _, err := srv.provider().ChatModel("m").Generate(t.Context(), &tt.call); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			got := srv.last(t)
			if got.Path != "/v1/chat/completions" {
				t.Errorf("path = %s, want /v1/chat/completions", got.Path)
			}
			equalJSON(t, strings.ReplaceAll(tt.want, "call_AAAA", callIDOf(t, longID)), got.Body)
		})
	}
}

func callIDOf(t *testing.T, long string) string {
	t.Helper()
	srv := newFakeServer(t, jsonReply(chatMinimal))
	call := &ai.Call{Messages: []ai.Message{
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.ToolCall{ID: long, Name: "g"}}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: long, Name: "g", Output: ai.ToolOutput{Text: "x"}}}},
	}}
	if _, err := srv.provider().ChatModel("m").Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	var req struct {
		Messages []struct {
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(srv.last(t).Body, &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	id := req.Messages[1].ToolCallID
	if len(id) > 40 || !strings.HasPrefix(id, "call_") {
		t.Errorf("mapped call id %q, want at most 40 chars starting with call_", id)
	}
	return id
}

func TestChatModel_response(t *testing.T) {
	body := `{"id":"chatcmpl-2","service_tier":"default","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"thinking","content":"answer",
	"annotations":[{"type":"url_citation","url_citation":{"start_index":0,"end_index":6,"url":"https://x.test","title":"X"}}],
	"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}}],
	"usage":{"prompt_tokens":20,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":8},"completion_tokens_details":{"reasoning_tokens":4}}}`
	srv := newFakeServer(t, jsonReply(body))
	resp, err := srv.provider().ChatModel("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := len(resp.Content); got != 3 {
		t.Fatalf("content has %d parts, want 3", got)
	}
	if r, ok := resp.Content[0].(ai.Reasoning); !ok || r.Text != "thinking" {
		t.Errorf("content[0] = %#v, want reasoning", resp.Content[0])
	}
	text, ok := resp.Content[1].(ai.Text)
	if !ok || text.Text != "answer" || len(text.Citations) != 1 || text.Citations[0].URL != "https://x.test" || text.Citations[0].End != 6 {
		t.Errorf("content[1] = %#v, want text with one url citation", resp.Content[1])
	}
	if c, ok := resp.Content[2].(ai.ToolCall); !ok || c.ID != "call_1" || string(c.Input) != `{"a":1}` {
		t.Errorf("content[2] = %#v, want tool call", resp.Content[2])
	}
	got := resp.Usage
	got.Raw = nil
	if diff := cmp.Diff(ai.Usage{Input: 20, Output: 9, CacheRead: 8, Reasoning: 4}, got); diff != "" {
		t.Errorf("usage mismatch (-want +got):\n%s", diff)
	}
	if resp.FinishReason != ai.FinishToolCalls || resp.FinishRaw != "tool_calls" {
		t.Errorf("finish = %q/%q, want tool-calls/tool_calls", resp.FinishReason, resp.FinishRaw)
	}
}

func TestChatModel_finishReasons(t *testing.T) {
	tests := []struct {
		raw     string
		message string
		want    ai.FinishReason
	}{
		{raw: "stop", message: `"content":"x"`, want: ai.FinishStop},
		{raw: "length", message: `"content":"x"`, want: ai.FinishLength},
		{raw: "content_filter", message: `"content":""`, want: ai.FinishContentFilter},
		{raw: "function_call", message: `"content":null`, want: ai.FinishToolCalls},
		{raw: "stop", message: `"content":null,"refusal":"no"`, want: ai.FinishRefusal},
		{raw: "weird", message: `"content":"x"`, want: ai.FinishOther},
	}
	for _, tt := range tests {
		t.Run(tt.raw+" "+tt.message, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(`{"id":"c","choices":[{"message":{`+tt.message+`},"finish_reason":"`+tt.raw+`"}]}`))
			resp, err := srv.provider().ChatModel("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if resp.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, tt.want)
			}
		})
	}
}

func TestChatModel_noChoices(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"c","choices":[]}`))
	if _, err := srv.provider().ChatModel("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err == nil {
		t.Error("Generate() error = nil, want error")
	}
}

func TestChatModel_stream(t *testing.T) {
	chunk := func(s string) string { return "data: " + s + "\n\n" }
	body := chunk(`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"th"}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"reasoning_content":"ink"}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"content":"Hel"}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"f","arguments":""}}]}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"g","arguments":"{\"y\""}}]}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":":2}"}}]}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`) +
		chunk(`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`) +
		chunk(`{"id":"c1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":1}}}`) +
		"data: [DONE]\n\n"
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().ChatModel("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	equalJSON(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`, srv.last(t).Body)
	want := []string{
		"ReasoningStart", "ReasoningDelta", "ReasoningDelta", "ReasoningEnd", "TextStart", "TextDelta", "TextDelta", "TextEnd",
		"ToolInputStart", "ToolInputStart", "ToolInputDelta", "ToolInputDelta", "ToolInputDelta", "ToolInputDelta",
		"ToolInputEnd", "ToolCall", "ToolInputEnd", "ToolCall", "StepEnd",
	}
	if diff := cmp.Diff(want, partTypes(parts)); diff != "" {
		t.Fatalf("part types mismatch (-want +got):\n%s", diff)
	}
	var calls []string
	for _, p := range parts {
		if c, ok := p.(ai.ToolCall); ok {
			calls = append(calls, c.ID+"="+string(c.Input))
		}
	}
	if diff := cmp.Diff([]string{`call_a={"x":1}`, `call_b={"y":2}`}, calls); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}
	end, ok := parts[len(parts)-1].(ai.StepEnd)
	if !ok || end.FinishReason != ai.FinishToolCalls || end.Usage.Input != 7 || end.Usage.CacheRead != 2 || end.Usage.Reasoning != 1 {
		t.Errorf("StepEnd = %#v, want tool-calls with usage 7/cache 2/reasoning 1", parts[len(parts)-1])
	}
}

func TestChatModel_streamEndsWithoutDone(t *testing.T) {
	body := "data: {\"id\":\"c\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n"
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().ChatModel("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	if _, ok := parts[len(parts)-1].(ai.StepEnd); !ok {
		t.Errorf("last part = %T, want StepEnd", parts[len(parts)-1])
	}
}

func TestChatModel_streamErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "truncated", body: "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n", want: "ended before the response finished"},
		{name: "error chunk", body: "data: {\"error\":{\"message\":\"nope\",\"type\":\"server_error\",\"code\":null}}\n\n", want: "server_error"},
		{name: "bad chunk", body: "data: {x\n\n", want: "decode stream chunk"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, sseReply(tt.body))
			_, err := streamAll(t, srv.provider().ChatModel("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("stream error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestChatModel_streamRefusal(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"refusal\":\"no\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	srv := newFakeServer(t, sseReply(body))
	parts, err := streamAll(t, srv.provider().ChatModel("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("stream error = %v", err)
	}
	end, ok := parts[len(parts)-1].(ai.StepEnd)
	if !ok || end.FinishReason != ai.FinishRefusal {
		t.Errorf("last part = %#v, want StepEnd with refusal", parts[len(parts)-1])
	}
}
