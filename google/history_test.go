package google_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

const parallelReply = `{"candidates":[{"content":{"role":"model","parts":[
	{"functionCall":{"name":"get_weather","args":{"city":"Seoul"}},"thoughtSignature":"SIG-A"},
	{"functionCall":{"name":"get_weather","args":{"city":"Busan"}}}]},"finishReason":"STOP"}],"responseId":"resp-par"}`

const idReply = `{"candidates":[{"content":{"role":"model","parts":[
	{"functionCall":{"id":"srv-1","name":"get_weather","args":{"city":"Seoul"}},"thoughtSignature":"SIG-A"},
	{"functionCall":{"id":"srv-2","name":"get_weather","args":{"city":"Busan"}}}]},"finishReason":"STOP"}],"responseId":"resp-id"}`

func staticToken(context.Context) (string, error) { return "tok", nil }

func callOf(t *testing.T, resp *ai.Response) []ai.ToolCall {
	t.Helper()
	var out []ai.ToolCall
	for _, p := range resp.Content {
		if c, ok := p.(ai.ToolCall); ok {
			out = append(out, c)
		}
	}
	return out
}

func sendHistory(t *testing.T, f *fakeServer, model string, history []ai.Message) *ai.Response {
	t.Helper()
	resp, err := f.provider().Model(model).Generate(t.Context(), &ai.Call{Messages: history})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	return resp
}

func TestLanguageModel_generatedCallIDsAreDeterministic(t *testing.T) {
	f := newFakeServer(t, jsonReply(parallelReply))
	first := callOf(t, sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")}))
	second := callOf(t, sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")}))
	if len(first) != 2 || first[0].ID == "" || first[0].ID == first[1].ID {
		t.Fatalf("call ids = %q, %q; want two distinct ids", first[0].ID, first[1].ID)
	}
	if first[0].ID != second[0].ID || first[1].ID != second[1].ID {
		t.Errorf("call ids changed between identical responses: %v vs %v", first, second)
	}
	s := newFakeServer(t, sseReply(
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Seoul"}},"thoughtSignature":"SIG-A"}]}}],"responseId":"resp-par"}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Busan"}}}]},"finishReason":"STOP"}],"responseId":"resp-par"}`,
	))
	res, err := ai.StreamText(t.Context(), s.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxSteps: 1}).Result()
	if err != nil {
		t.Fatalf("StreamText().Result() error = %v", err)
	}
	if len(res.Steps[0].ToolCalls) != 2 || res.Steps[0].ToolCalls[0].ID != first[0].ID || res.Steps[0].ToolCalls[1].ID != first[1].ID {
		t.Errorf("stream call ids = %v, want %q and %q", res.Steps[0].ToolCalls, first[0].ID, first[1].ID)
	}
}

func TestLanguageModel_functionResponses(t *testing.T) {
	tests := []struct {
		name    string
		reply   string
		results func(calls []ai.ToolCall) ai.Message
		want    string
	}{
		{
			name:  "generated ids are not echoed and order follows the calls",
			reply: parallelReply,
			results: func(calls []ai.ToolCall) ai.Message {
				return ai.Message{Role: ai.RoleTool, Content: ai.Parts{
					ai.ToolResult{CallID: calls[1].ID, Name: "get_weather", Output: ai.ToolOutput{Text: "rain"}},
					ai.ToolResult{CallID: calls[0].ID, Name: "get_weather", Output: ai.ToolOutput{JSON: json.RawMessage(`{"temp":21}`)}},
				}}
			},
			want: `[{"functionResponse":{"name":"get_weather","response":{"temp":21}}},{"functionResponse":{"name":"get_weather","response":{"output":"rain"}}}]`,
		},
		{
			name:  "server ids are echoed",
			reply: idReply,
			results: func(calls []ai.ToolCall) ai.Message {
				return ai.Message{Role: ai.RoleTool, Content: ai.Parts{
					ai.ToolResult{CallID: calls[0].ID, Name: "get_weather", Output: ai.ToolOutput{JSON: json.RawMessage(`[1,2]`)}},
					ai.ToolResult{CallID: calls[1].ID, Name: "get_weather", IsError: true, Output: ai.ToolOutput{Text: "boom"}},
				}}
			},
			want: `[{"functionResponse":{"id":"srv-1","name":"get_weather","response":{"output":[1,2]}}},{"functionResponse":{"id":"srv-2","name":"get_weather","response":{"error":"boom"}}}]`,
		},
		{
			name:  "missing results become errors",
			reply: idReply,
			results: func(calls []ai.ToolCall) ai.Message {
				return ai.Message{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: calls[1].ID, Name: "get_weather", Output: ai.ToolOutput{Text: "ok"}}}}
			},
			want: `[{"functionResponse":{"id":"srv-1","name":"get_weather","response":{"error":"error: this tool call has no result"}}},{"functionResponse":{"id":"srv-2","name":"get_weather","response":{"output":"ok"}}}]`,
		},
		{
			name:  "result parts carry files",
			reply: idReply,
			results: func(calls []ai.ToolCall) ai.Message {
				return ai.Message{Role: ai.RoleTool, Content: ai.Parts{
					ai.ToolResult{CallID: calls[0].ID, Name: "get_weather", Output: ai.ToolOutput{Parts: ai.Parts{ai.Text{Text: "map"}, ai.File{MediaType: "image/png", Data: []byte{1, 2}}}}},
					ai.ToolResult{CallID: calls[1].ID, Name: "get_weather", Output: ai.ToolOutput{Text: "ok"}},
				}}
			},
			want: `[{"functionResponse":{"id":"srv-1","name":"get_weather","response":{"output":"map","attachments":[{"$ref":"file-1"}]},"parts":[{"inlineData":{"displayName":"file-1","mimeType":"image/png","data":"AQI="}}]}},{"functionResponse":{"id":"srv-2","name":"get_weather","response":{"output":"ok"}}}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(tt.reply))
			first := sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")})
			history := []ai.Message{
				ai.User("hi"),
				{Role: ai.RoleAssistant, Content: first.Content, Provider: first.Provider},
				tt.results(callOf(t, first)),
			}
			sendHistory(t, f, "gemini-3-flash", history)
			var req struct {
				Contents []struct {
					Role  string            `json:"role"`
					Parts []json.RawMessage `json:"parts"`
				} `json:"contents"`
			}
			if err := json.Unmarshal(f.last(t).Body, &req); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}
			if len(req.Contents) != 3 || req.Contents[1].Role != "model" || req.Contents[2].Role != "user" {
				t.Fatalf("contents = %+v, want user, model, user", req.Contents)
			}
			raw, err := json.Marshal(req.Contents[2].Parts)
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			equalJSON(t, tt.want, raw)
			if !strings.Contains(string(req.Contents[1].Parts[0]), `"thoughtSignature":"SIG-A"`) {
				t.Errorf("first function call lost its signature: %s", req.Contents[1].Parts[0])
			}
			if strings.Contains(string(req.Contents[1].Parts[1]), "thoughtSignature") {
				t.Errorf("second parallel call gained a signature: %s", req.Contents[1].Parts[1])
			}
		})
	}
}

func TestLanguageModel_dummySignatureForForeignCalls(t *testing.T) {
	history := []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.ToolCall{ID: "toolu_1", Name: "a", Input: json.RawMessage(`{}`)},
			ai.ToolCall{ID: "toolu_2", Name: "b", Input: json.RawMessage(`{}`)},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{
			ai.ToolResult{CallID: "toolu_1", Name: "a", Output: ai.ToolOutput{Text: "x"}},
			ai.ToolResult{CallID: "toolu_2", Name: "b", Output: ai.ToolOutput{Text: "y"}},
		}},
	}
	tests := []struct {
		model string
		want  int
	}{
		{model: "gemini-3-flash", want: 1},
		{model: "gemini-4-argon", want: 1},
		{model: "gemini-2.5-flash", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(textReply))
			resp := sendHistory(t, f, tt.model, history)
			if got := strings.Count(string(f.last(t).Body), `"thoughtSignature":"skip_thought_signature_validator"`); got != tt.want {
				t.Errorf("dummy signatures = %d, want %d", got, tt.want)
			}
			warned := false
			for _, w := range resp.Warnings {
				warned = warned || strings.Contains(w.Message, "skip_thought_signature_validator")
			}
			if warned != (tt.want == 1) {
				t.Errorf("dummy signature warning = %v, want %v", warned, tt.want == 1)
			}
		})
	}
}

func TestLanguageModel_signaturesAreNotSharedBetweenEndpoints(t *testing.T) {
	studio := newFakeServer(t, jsonReply(`{"candidates":[{"content":{"role":"model","parts":[
		{"text":"SUMMARY-THOUGHT","thought":true,"thoughtSignature":"SIG-STUDIO-THOUGHT"},
		{"text":"answer","thoughtSignature":"SIG-STUDIO-TEXT"},
		{"functionCall":{"name":"lookup","args":{}},"thoughtSignature":"SIG-STUDIO-CALL"}]},"finishReason":"STOP"}],"responseId":"r-studio"}`))
	first := sendHistory(t, studio, "gemini-3-flash", []ai.Message{ai.User("q")})
	assistant := ai.Message{Role: ai.RoleAssistant, Content: first.Content, Provider: first.Provider}
	history := []ai.Message{ai.User("q"), assistant, {Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: callOf(t, first)[0].ID, Name: "lookup", Output: ai.ToolOutput{Text: "r"}}}}}

	sendHistory(t, studio, "gemini-3-flash", history)
	same := string(studio.last(t).Body)
	for _, want := range []string{"SIG-STUDIO-THOUGHT", "SIG-STUDIO-TEXT", "SIG-STUDIO-CALL", "SUMMARY-THOUGHT"} {
		if !strings.Contains(same, want) {
			t.Errorf("same endpoint request lost %q", want)
		}
	}

	vertex := newFakeServer(t, jsonReply(textReply))
	resp, err := google.New(google.WithVertex("p", "global"), google.WithBaseURL(vertex.URL), google.WithTokenSource(staticToken)).Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: history})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	other := string(vertex.last(t).Body)
	for _, forbidden := range []string{"SIG-STUDIO-THOUGHT", "SIG-STUDIO-TEXT", "SIG-STUDIO-CALL", "SUMMARY-THOUGHT"} {
		if strings.Contains(other, forbidden) {
			t.Errorf("Vertex request contains AI Studio data %q", forbidden)
		}
	}
	if !strings.Contains(other, `"text":"answer"`) || !strings.Contains(other, `"thoughtSignature":"skip_thought_signature_validator"`) {
		t.Errorf("Vertex request should keep the answer text and use the dummy signature:\n%s", other)
	}
	if len(resp.Warnings) == 0 {
		t.Error("Vertex response has no warning about the replaced signature")
	}
	if _, ok := ai.ProviderValue[map[string]any](resp.Provider, "google-vertex"); !ok {
		t.Errorf("Vertex response data is not under google-vertex: %v", resp.Provider)
	}
	if _, ok := ai.ProviderValue[map[string]any](first.Provider, "google"); !ok {
		t.Errorf("AI Studio response data is not under google: %v", first.Provider)
	}
}

func TestLanguageModel_emptyParts(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	signed := ai.Text{Provider: ai.ProviderData{"google": json.RawMessage(`{"part":{"text":"","thoughtSignature":"SIG-TAIL"}}`)}}
	unsigned := ai.Text{Provider: ai.ProviderData{"google": json.RawMessage(`{"part":{"text":""}}`)}}
	sendHistory(t, f, "gemini-3-flash", []ai.Message{
		ai.User("q"),
		{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "a"}, unsigned, signed, ai.Text{}}},
		ai.User("next"),
	})
	var req struct {
		Contents []struct {
			Parts []json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(f.last(t).Body, &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	raw, err := json.Marshal(req.Contents[1].Parts)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	equalJSON(t, `[{"text":"a"},{"text":"","thoughtSignature":"SIG-TAIL"}]`, raw)
}

func TestLanguageModel_userFiles(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.UserParts(
		ai.Text{Text: "look"},
		ai.File{MediaType: "image/png", Data: []byte{1, 2}},
		ai.File{MediaType: "application/pdf", URL: "https://generativelanguage.googleapis.com/v1beta/files/abc"},
	)})
	var req struct {
		Contents []struct {
			Parts []json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(f.last(t).Body, &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	raw, err := json.Marshal(req.Contents[0].Parts)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	equalJSON(t, `[{"text":"look"},{"inlineData":{"mimeType":"image/png","data":"AQI="}},{"fileData":{"mimeType":"application/pdf","fileUri":"https://generativelanguage.googleapis.com/v1beta/files/abc"}}]`, raw)

	_, err = f.provider().Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.UserParts(ai.File{Data: []byte{1}})}})
	if err == nil {
		t.Error("Generate() with a file that has no media type error = nil, want an error")
	}
}

func TestLanguageModel_replayAcrossEndpoints(t *testing.T) {
	const reply = `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"answer","thoughtSignature":"SIG-TEXT"},
		{"functionCall":{"name":"lookup","args":{}},"thoughtSignature":"SIG-CALL"}]},"finishReason":"STOP"}],"responseId":"r-1"}`
	endpoints := map[string]func(f *fakeServer) *google.Provider{
		"google": func(f *fakeServer) *google.Provider { return f.provider() },
		"google-vertex": func(f *fakeServer) *google.Provider {
			return google.New(google.WithVertex("p", "global"), google.WithBaseURL(f.URL), google.WithTokenSource(staticToken))
		},
	}
	tests := []struct {
		name          string
		from          string
		to            string
		wantSignature bool
		wantDummy     bool
	}{
		{name: "ai studio to ai studio", from: "google", to: "google", wantSignature: true},
		{name: "vertex to vertex", from: "google-vertex", to: "google-vertex", wantSignature: true},
		{name: "ai studio to vertex", from: "google", to: "google-vertex", wantDummy: true},
		{name: "vertex to ai studio", from: "google-vertex", to: "google", wantDummy: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := newFakeServer(t, jsonReply(reply))
			first, err := endpoints[tt.from](source).Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("q")}})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if _, ok := ai.ProviderValue[map[string]any](first.Provider, tt.from); !ok {
				t.Fatalf("response data is not under %q: %v", tt.from, first.Provider)
			}
			history := []ai.Message{
				ai.User("q"),
				{Role: ai.RoleAssistant, Content: first.Content, Provider: first.Provider},
				{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: callOf(t, first)[0].ID, Name: "lookup", Output: ai.ToolOutput{Text: "r"}}}},
			}
			target := newFakeServer(t, jsonReply(textReply))
			resp, err := endpoints[tt.to](target).Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: history})
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			sent := string(target.last(t).Body)
			if got := strings.Contains(sent, "SIG-TEXT") && strings.Contains(sent, "SIG-CALL"); got != tt.wantSignature {
				t.Errorf("request keeps source signatures = %v, want %v", got, tt.wantSignature)
			}
			if got := strings.Contains(sent, `"thoughtSignature":"skip_thought_signature_validator"`); got != tt.wantDummy {
				t.Errorf("request uses the dummy signature = %v, want %v", got, tt.wantDummy)
			}
			if !strings.Contains(sent, `"text":"answer"`) {
				t.Errorf("request lost the answer text:\n%s", sent)
			}
			if _, ok := ai.ProviderValue[map[string]any](resp.Provider, tt.to); !ok {
				t.Errorf("response data is not under %q: %v", tt.to, resp.Provider)
			}
		})
	}
}
