package bedrock_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
)

func generate(t *testing.T, f *fakeServer, req ai.TextRequest) (*ai.TextResult, error) {
	t.Helper()
	return ai.GenerateText(t.Context(), f.provider().Model(testModel), req)
}

func TestLanguageModel_Generate_request(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	tests := []struct {
		name string
		req  ai.TextRequest
		want string
	}{
		{
			name: "text with system and inference config",
			req: ai.TextRequest{
				Instructions:    ai.Texts("You are terse."),
				Messages:        []ai.Message{ai.User("hi")},
				MaxOutputTokens: 100,
				Temperature:     new(0.5),
				TopP:            new(0.9),
				StopSequences:   []string{"END"},
			},
			want: `{"system":[{"text":"You are terse."}],"messages":[{"role":"user","content":[{"text":"hi"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.5,"topP":0.9,"stopSequences":["END"]}}`,
		},
		{
			name: "cache points on system, user text and tools",
			req: ai.TextRequest{
				Instructions: []ai.Text{bedrock.CachedText("Long policy."), {Text: "short", Provider: bedrock.CachePoint(bedrock.TTL1Hour)}},
				Messages:     []ai.Message{ai.UserParts(bedrock.CachedTextFor("Big document.", bedrock.TTL1Hour), ai.Text{Text: "question"})},
				Tools:        []ai.Tool{weather},
			},
			want: `{"system":[{"text":"Long policy."},{"cachePoint":{"type":"default"}},{"text":"short"},{"cachePoint":{"type":"default","ttl":"1h"}}],"messages":[{"role":"user","content":[{"text":"Big document."},{"cachePoint":{"type":"default","ttl":"1h"}},{"text":"question"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Weather by city","inputSchema":{"json":{"additionalProperties":false,"properties":{"city":{"description":"city name","type":"string"}},"required":["city"],"type":"object"}}}}]}}`,
		},
		{
			name: "required tool choice",
			req:  ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{weather}, ToolChoice: ai.ToolRequired()},
			want: `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Weather by city","inputSchema":{"json":{"additionalProperties":false,"properties":{"city":{"description":"city name","type":"string"}},"required":["city"],"type":"object"}}}}],"toolChoice":{"any":{}}}}`,
		},
		{
			name: "named tool choice",
			req:  ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{weather}, ToolChoice: ai.ToolNamed("get_weather")},
			want: `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Weather by city","inputSchema":{"json":{"additionalProperties":false,"properties":{"city":{"description":"city name","type":"string"}},"required":["city"],"type":"object"}}}}],"toolChoice":{"tool":{"name":"get_weather"}}}}`,
		},
		{
			name: "images and documents",
			req: ai.TextRequest{Messages: []ai.Message{ai.UserParts(
				ai.File{MediaType: "image/png", Data: []byte("png")},
				ai.File{MediaType: "application/pdf", Filename: "Q3 report.pdf", Data: []byte("pdf"), Citations: true},
				ai.File{MediaType: "text/markdown", URL: "s3://bucket/notes.md"},
				ai.Text{Text: "summarize"},
			)}},
			want: `{"messages":[{"role":"user","content":[{"image":{"format":"png","source":{"bytes":"cG5n"}}},{"document":{"name":"Q3 report pdf","format":"pdf","source":{"bytes":"cGRm"},"citations":{"enabled":true}}},{"document":{"name":"document 2","format":"md","source":{"s3Location":{"uri":"s3://bucket/notes.md"}}}},{"text":"summarize"}]}]}`,
		},
		{
			name: "tool round trip with error, JSON and image results",
			req: ai.TextRequest{Messages: []ai.Message{
				ai.User("weather?"),
				{Role: ai.RoleAssistant, Content: ai.Parts{
					ai.ToolCall{ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"city":"Seoul"}`)},
					ai.ToolCall{ID: "call_2", Name: "get_weather", Input: json.RawMessage(`{"city":"Busan"}`)},
					ai.ToolCall{ID: "call_3", Name: "get_weather", Input: json.RawMessage(`{"city":"Daegu"}`)},
				}},
				{Role: ai.RoleTool, Content: ai.Parts{
					ai.ToolResult{CallID: "call_1", Name: "get_weather", Output: ai.ToolOutput{JSON: json.RawMessage(`{"temp":21}`)}},
					ai.ToolResult{CallID: "call_2", Name: "get_weather", Output: ai.ToolOutput{Text: "city not found"}, IsError: true},
					ai.ToolResult{CallID: "call_3", Name: "get_weather", Output: ai.ToolOutput{Parts: ai.Parts{ai.Text{Text: "chart"}, ai.File{MediaType: "image/jpeg", Data: []byte("jpg")}}}},
				}},
			}, Tools: []ai.Tool{weather}},
			want: `{"messages":[{"role":"user","content":[{"text":"weather?"}]},{"role":"assistant","content":[{"toolUse":{"toolUseId":"call_1","name":"get_weather","input":{"city":"Seoul"}}},{"toolUse":{"toolUseId":"call_2","name":"get_weather","input":{"city":"Busan"}}},{"toolUse":{"toolUseId":"call_3","name":"get_weather","input":{"city":"Daegu"}}}]},{"role":"user","content":[{"toolResult":{"toolUseId":"call_1","content":[{"json":{"temp":21}}]}},{"toolResult":{"toolUseId":"call_2","content":[{"text":"city not found"}],"status":"error"}},{"toolResult":{"toolUseId":"call_3","content":[{"text":"chart"},{"image":{"format":"jpeg","source":{"bytes":"anBn"}}}]}}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Weather by city","inputSchema":{"json":{"additionalProperties":false,"properties":{"city":{"description":"city name","type":"string"}},"required":["city"],"type":"object"}}}}]}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(minimalResponse))
			if _, err := generate(t, f, withOneStep(tt.req)); err != nil {
				t.Fatalf("GenerateText() error = %v", err)
			}
			equalJSON(t, tt.want, f.last(t).Body)
		})
	}
}

func withOneStep(r ai.TextRequest) ai.TextRequest {
	r.MaxSteps = 1
	return r
}

func TestLanguageModel_Generate_path(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	if _, err := generate(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if got, want := f.last(t).Path, "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse"; got != want {
		t.Errorf("request path = %q, want %q", got, want)
	}
}

func TestLanguageModel_Generate_arnModelID(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	arn := "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
	if _, err := ai.GenerateText(t.Context(), f.provider().Model(arn), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	want := "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse"
	if got := f.last(t).Path; got != want {
		t.Errorf("request path = %q, want %q", got, want)
	}
}

func TestLanguageModel_Generate_options(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	_, err := generate(t, f, ai.TextRequest{
		Messages: []ai.Message{ai.User("hi")},
		Provider: ai.Providers(bedrock.Options{
			Guardrail:                         &bedrock.Guardrail{Identifier: "gr1abcd2efgh", Version: "3", Trace: bedrock.TraceEnabled, StreamProcessingMode: bedrock.ProcessingAsync},
			AdditionalModelRequestFields:      json.RawMessage(`{"thinking":{"type":"enabled","budget_tokens":1024}}`),
			AdditionalModelResponseFieldPaths: []string{"/stop_sequence"},
			PerformanceLatency:                bedrock.LatencyOptimized,
			ServiceTier:                       bedrock.TierPriority,
			RequestMetadata:                   map[string]string{"team": "search"},
		}),
	})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	equalJSON(t, `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"guardrailConfig":{"guardrailIdentifier":"gr1abcd2efgh","guardrailVersion":"3","trace":"enabled"},"additionalModelRequestFields":{"thinking":{"type":"enabled","budget_tokens":1024}},"additionalModelResponseFieldPaths":["/stop_sequence"],"performanceConfig":{"latency":"optimized"},"serviceTier":{"type":"priority"},"requestMetadata":{"team":"search"}}`, f.last(t).Body)
}

func TestLanguageModel_Generate_structuredOutput(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}`)
	tests := []struct {
		name     string
		opts     bedrock.Options
		spec     *ai.OutputSpec
		reply    string
		wantBody string
		wantText string
	}{
		{
			name:     "native json schema",
			spec:     &ai.OutputSpec{Name: "extract data", Description: "Extract fields", Schema: schema},
			reply:    `{"output":{"message":{"role":"assistant","content":[{"text":"{\"title\":\"x\"}"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`,
			wantBody: `{"messages":[{"role":"user","content":[{"text":"go"}]}],"outputConfig":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":"{\"additionalProperties\":false,\"properties\":{\"title\":{\"type\":\"string\"}},\"required\":[\"title\"],\"type\":\"object\"}","name":"extractdata","description":"Extract fields"}}}}}`,
			wantText: `{"title":"x"}`,
		},
		{
			name:     "forced tool when requested",
			opts:     bedrock.Options{OutputViaTool: true},
			spec:     &ai.OutputSpec{Name: "extract", Schema: schema},
			reply:    `{"output":{"message":{"role":"assistant","content":[{"toolUse":{"toolUseId":"tooluse_1","name":"extract","input":{"title":"x"}}}]}},"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`,
			wantBody: `{"messages":[{"role":"user","content":[{"text":"go"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"extract","inputSchema":{"json":{"additionalProperties":false,"properties":{"title":{"type":"string"}},"required":["title"],"type":"object"}}}}],"toolChoice":{"tool":{"name":"extract"}}}}`,
			wantText: `{"title":"x"}`,
		},
		{
			name:     "forced tool without a schema",
			spec:     &ai.OutputSpec{Name: "free"},
			reply:    `{"output":{"message":{"role":"assistant","content":[{"toolUse":{"toolUseId":"tooluse_1","name":"free","input":{"a":1}}}]}},"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`,
			wantBody: `{"messages":[{"role":"user","content":[{"text":"go"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"free","inputSchema":{"json":{"type":"object"}}}}],"toolChoice":{"tool":{"name":"free"}}}}`,
			wantText: `{"a":1}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(tt.reply))
			model := f.provider().Model(testModel)
			resp, err := model.Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("go")}, Output: tt.spec, Provider: ai.Providers(tt.opts)})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			equalJSON(t, tt.wantBody, f.last(t).Body)
			text, ok := resp.Content[0].(ai.Text)
			if !ok || text.Text != tt.wantText {
				t.Errorf("Generate() content = %#v, want text %q", resp.Content, tt.wantText)
			}
			if resp.FinishReason != ai.FinishStop {
				t.Errorf("FinishReason = %q, want %q (forced tool calls are not tool calls)", resp.FinishReason, ai.FinishStop)
			}
		})
	}
}

func TestLanguageModel_Generate_warnings(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	res, err := generate(t, f, ai.TextRequest{
		Messages:         []ai.Message{ai.User("hi")},
		TopK:             5,
		Reasoning:        ai.ReasoningHigh,
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.1),
		Seed:             new(7),
	})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	var all []string
	for _, w := range res.Warnings {
		all = append(all, w.Message)
	}
	joined := strings.Join(all, "\n")
	for _, want := range []string{"topK", "reasoning", "presencePenalty", "frequencyPenalty", "seed", "AdditionalModelRequestFields"} {
		if !strings.Contains(joined, want) {
			t.Errorf("warnings do not mention %q:\n%s", want, joined)
		}
	}
}

func TestLanguageModel_Generate_toolChoiceNone(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	f := newFakeServer(t, jsonReply(minimalResponse))
	res, err := generate(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{weather}, ToolChoice: ai.ToolNone()})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if strings.Contains(string(f.last(t).Body), "toolConfig") {
		t.Errorf("request body has toolConfig, want none for ToolNone without tool history:\n%s", f.last(t).Body)
	}
	if len(res.Warnings) == 0 {
		t.Error("Warnings is empty, want a tool choice warning")
	}
}

func TestLanguageModel_Generate_invalidInput(t *testing.T) {
	tests := []struct {
		name string
		req  ai.TextRequest
		want string
	}{
		{name: "tool name with a dot", req: ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{{Name: "mcp.search", Description: "d"}}}, want: "tool name"},
		{name: "unsupported media type", req: ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "audio/mpeg", Data: []byte("x")})}}, want: "unsupported file media type"},
		{name: "http file url", req: ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "image/png", URL: "https://example.com/a.png"})}}, want: "s3://"},
		{name: "file without data", req: ai.TextRequest{Messages: []ai.Message{ai.UserParts(ai.File{MediaType: "image/png"})}}, want: "no data"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(minimalResponse))
			_, err := generate(t, f, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateText() error = %v, want %q", err, tt.want)
			}
			if f.count() != 0 {
				t.Errorf("server saw %d requests, want 0", f.count())
			}
		})
	}
}

func TestLanguageModel_Generate_response(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"output":{"message":{"role":"assistant","content":[{"text":"done"}]}},"stopReason":"end_turn","usage":{"inputTokens":10,"outputTokens":5,"totalTokens":55,"cacheReadInputTokens":30,"cacheWriteInputTokens":10},"metrics":{"latencyMs":99},"additionalModelResponseFields":{"stop_sequence":"x"},"performanceConfig":{"latency":"optimized"},"serviceTier":{"type":"priority"}}`))
	res, err := generate(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Usage.Input != 50 || res.Usage.CacheRead != 30 || res.Usage.CacheWrite != 10 || res.Usage.Output != 5 {
		t.Errorf("Usage = %+v, want input 50 (10 + 30 + 10), cache read 30, cache write 10, output 5", res.Usage)
	}
	data, ok := ai.ProviderValue[bedrock.ResponseData](res.Provider, "bedrock")
	if !ok {
		t.Fatalf("Provider[bedrock] is missing: %v", res.Provider)
	}
	if string(data.Metrics) != `{"latencyMs":99}` || string(data.ServiceTier) != `{"type":"priority"}` || string(data.PerformanceConfig) != `{"latency":"optimized"}` || string(data.AdditionalModelResponseFields) != `{"stop_sequence":"x"}` {
		t.Errorf("ResponseData = %+v, want metrics, tier, performance config and additional fields", data)
	}
}

func TestLanguageModel_Generate_unknownBlockIsKept(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"output":{"message":{"role":"assistant","content":[{"audio":{"format":"mp3","source":{"bytes":"AAAA"}}},{"text":"x"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`))
	resp, err := f.provider().Model(testModel).Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	custom, ok := resp.Content[0].(ai.Custom)
	if !ok || custom.Kind != "bedrock.audio" || !strings.Contains(string(custom.Data), "AAAA") {
		t.Errorf("Content[0] = %#v, want a bedrock.audio custom part with the raw block", resp.Content[0])
	}
}

func TestLanguageModel_replaysReasoningOnlyForTheSameModel(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"output":{"message":{"role":"assistant","content":[{"reasoningContent":{"reasoningText":{"text":"think","signature":"SIG-1"}}},{"reasoningContent":{"reasoningText":{"text":"unsigned"}}},{"text":"answer"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`))
	first, err := generate(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("q")}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	history := append([]ai.Message{ai.User("q")}, first.Messages...)
	history = append(history, ai.User("next"))
	if _, err := generate(t, f, ai.TextRequest{Messages: history}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	body := string(f.last(t).Body)
	if !strings.Contains(body, `"signature":"SIG-1"`) || strings.Contains(body, "unsigned") {
		t.Errorf("same-model replay body = %s, want the signed reasoning block and no unsigned one", body)
	}
	other := f.provider().Model("amazon.nova-pro-v1:0")
	if _, err := ai.GenerateText(t.Context(), other, ai.TextRequest{Messages: history}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if strings.Contains(string(f.last(t).Body), "SIG-1") {
		t.Errorf("other-model replay body = %s, want the signed reasoning dropped", f.last(t).Body)
	}
}

func TestLanguageModel_Generate_headers(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	if _, err := generate(t, f, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Headers: map[string][]string{"X-Trace": {"abc"}}}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if got := f.last(t).Header.Get("X-Trace"); got != "abc" {
		t.Errorf("X-Trace = %q, want abc", got)
	}
}

func TestSystemTool(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	_, err := generate(t, f, withOneStep(ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: []ai.Tool{bedrock.SystemTool("nova_grounding")}}))
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	equalJSON(t, `{"messages":[{"role":"user","content":[{"text":"hi"}]}],"toolConfig":{"tools":[{"systemTool":{"name":"nova_grounding"}}]}}`, f.last(t).Body)
}
