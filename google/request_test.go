package google_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

const textReply = `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1},"responseId":"r1"}`

func generate(t *testing.T, f *fakeServer, id string, req ai.TextRequest) (*ai.TextResult, error) {
	t.Helper()
	req.MaxRetries = new(0)
	return ai.GenerateText(t.Context(), f.provider().Model(id), req)
}

func requestBody(t *testing.T, f *fakeServer) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(f.last(t).Body, &out); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	return out
}

func TestLanguageModel_requestShape(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	_, err := generate(t, f, "gemini-3-flash", ai.TextRequest{
		Instructions:     ai.Texts("Be brief.", "Be kind."),
		Messages:         []ai.Message{ai.User("hello")},
		Tools:            []ai.Tool{google.Search(), weather},
		MaxOutputTokens:  100,
		Temperature:      new(0.5),
		TopP:             new(0.9),
		TopK:             40,
		Seed:             new(7),
		StopSequences:    []string{"END"},
		PresencePenalty:  new(0.1),
		FrequencyPenalty: new(0.2),
		Provider: ai.Providers(google.Options{
			SafetySettings:     []google.SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH"}},
			CachedContent:      "cachedContents/abc",
			ResponseModalities: []string{"TEXT"},
			Labels:             map[string]string{"team": "a"},
			ServiceTier:        "flex",
		}),
	})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	got := f.last(t)
	if got.Path != "/v1beta/models/gemini-3-flash:generateContent" || got.Header.Get("x-goog-api-key") != "test-key" {
		t.Errorf("request = %s with key %q", got.Path, got.Header.Get("x-goog-api-key"))
	}
	equalJSON(t, `{
		"contents":[{"role":"user","parts":[{"text":"hello"}]}],
		"systemInstruction":{"parts":[{"text":"Be brief."},{"text":"Be kind."}]},
		"tools":[{"googleSearch":{}},{"functionDeclarations":[{"name":"get_weather","description":"Weather by city","parametersJsonSchema":{"type":"object","properties":{"city":{"type":"string","description":"city name"}},"required":["city"],"additionalProperties":false}}]}],
		"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_ONLY_HIGH"}],
		"cachedContent":"cachedContents/abc",
		"labels":{"team":"a"},
		"serviceTier":"flex",
		"generationConfig":{"temperature":0.5,"topP":0.9,"topK":40,"maxOutputTokens":100,"stopSequences":["END"],"seed":7,"presencePenalty":0.1,"frequencyPenalty":0.2,"responseModalities":["TEXT"]}
	}`, got.Body)
}

func TestLanguageModel_streamRequest(t *testing.T) {
	f := newFakeServer(t, sseReply(`{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`))
	res, err := ai.StreamText(t.Context(), f.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hello")}}).Result()
	if err != nil {
		t.Fatalf("StreamText().Result() error = %v", err)
	}
	got := f.last(t)
	if res.Text != "hi" || got.Path != "/v1beta/models/gemini-3-flash:streamGenerateContent" || got.Query != "alt=sse" {
		t.Errorf("text = %q, request = %s?%s", res.Text, got.Path, got.Query)
	}
}

func TestLanguageModel_thinking(t *testing.T) {
	tests := []struct {
		name   string
		model  string
		req    ai.TextRequest
		want   string
		wantEr bool
	}{
		{name: "gemini 3 high", model: "gemini-3-flash", req: ai.TextRequest{Reasoning: ai.ReasoningHigh}, want: `{"includeThoughts":true,"thinkingLevel":"high"}`},
		{name: "gemini 3 none", model: "gemini-3-flash", req: ai.TextRequest{Reasoning: ai.ReasoningNone}, want: `{"includeThoughts":true,"thinkingLevel":"minimal"}`},
		{name: "gemini 4 minimal", model: "gemini-4-argon", req: ai.TextRequest{Reasoning: ai.ReasoningMinimal}, want: `{"includeThoughts":true,"thinkingLevel":"low"}`},
		{name: "gemini 4 hydrogen max", model: "gemini-4-hydrogen", req: ai.TextRequest{Reasoning: ai.ReasoningMax}, want: `{"includeThoughts":true,"thinkingLevel":"xhigh"}`},
		{name: "gemini 2.5 none", model: "gemini-2.5-flash", req: ai.TextRequest{Reasoning: ai.ReasoningNone}, want: `{"thinkingBudget":0}`},
		{name: "gemini 2.5 pro none clamps to 128", model: "gemini-2.5-pro", req: ai.TextRequest{Reasoning: ai.ReasoningNone}, want: `{"includeThoughts":true,"thinkingBudget":128}`},
		{name: "gemini 2.5 pro minimal keeps 512", model: "gemini-2.5-pro", req: ai.TextRequest{Reasoning: ai.ReasoningMinimal}, want: `{"includeThoughts":true,"thinkingBudget":512}`},
		{name: "gemini 2.5 medium", model: "gemini-2.5-flash", req: ai.TextRequest{Reasoning: ai.ReasoningMedium}, want: `{"includeThoughts":true,"thinkingBudget":8192}`},
		{name: "options budget wins", model: "gemini-2.5-pro", req: ai.TextRequest{Reasoning: ai.ReasoningLow, Provider: ai.Providers(google.Options{ThinkingBudget: new(-1), IncludeThoughts: new(false)})}, want: `{"includeThoughts":false,"thinkingBudget":-1}`},
		{name: "options level", model: "gemini-3-pro", req: ai.TextRequest{Provider: ai.Providers(google.Options{ThinkingLevel: "low"})}, want: `{"includeThoughts":true,"thinkingLevel":"low"}`},
		{name: "budget and level", model: "gemini-3-pro", req: ai.TextRequest{Provider: ai.Providers(google.Options{ThinkingBudget: new(10), ThinkingLevel: "low"})}, wantEr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(textReply))
			tt.req.Messages = []ai.Message{ai.User("hi")}
			_, err := generate(t, f, tt.model, tt.req)
			if tt.wantEr {
				if err == nil {
					t.Fatal("GenerateText() error = nil, want an error")
				}
				return
			}
			if err != nil {
				t.Fatalf("GenerateText() error = %v", err)
			}
			cfg, _ := requestBody(t, f)["generationConfig"].(map[string]any)
			raw, err := json.Marshal(cfg["thinkingConfig"])
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			equalJSON(t, tt.want, raw)
		})
	}
}

func TestLanguageModel_toolConfig(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather")
	strict := weather
	strict.Strict = true
	tests := []struct {
		name string
		req  ai.TextRequest
		want string
	}{
		{name: "auto has none", req: ai.TextRequest{Tools: []ai.Tool{weather}}, want: `null`},
		{name: "required", req: ai.TextRequest{Tools: []ai.Tool{weather}, ToolChoice: ai.ToolRequired()}, want: `{"functionCallingConfig":{"mode":"ANY"}}`},
		{name: "none", req: ai.TextRequest{Tools: []ai.Tool{weather}, ToolChoice: ai.ToolNone()}, want: `{"functionCallingConfig":{"mode":"NONE"}}`},
		{name: "named", req: ai.TextRequest{Tools: []ai.Tool{weather}, ToolChoice: ai.ToolNamed("get_weather")}, want: `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["get_weather"]}}`},
		{name: "strict is validated", req: ai.TextRequest{Tools: []ai.Tool{strict}}, want: `{"functionCallingConfig":{"mode":"VALIDATED"}}`},
		{name: "server side", req: ai.TextRequest{Tools: []ai.Tool{weather, google.Search()}, Provider: ai.Providers(google.Options{ServerSideToolInvocations: true})}, want: `{"functionCallingConfig":{"mode":"VALIDATED"},"includeServerSideToolInvocations":true}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(textReply))
			tt.req.Messages = []ai.Message{ai.User("hi")}
			if _, err := generate(t, f, "gemini-3-flash", tt.req); err != nil {
				t.Fatalf("GenerateText() error = %v", err)
			}
			raw, err := json.Marshal(requestBody(t, f)["toolConfig"])
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			equalJSON(t, tt.want, raw)
		})
	}
}

func TestLanguageModel_hostedTools(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	tools := []ai.Tool{google.Search(), google.URLContext(), google.CodeExecution(), google.Maps(), google.FileSearch("fileSearchStores/docs-1")}
	if _, err := generate(t, f, "gemini-3-flash", ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: tools}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	raw, err := json.Marshal(requestBody(t, f)["tools"])
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	equalJSON(t, `[{"googleSearch":{}},{"urlContext":{}},{"codeExecution":{}},{"googleMaps":{}},{"fileSearch":{"fileSearchStoreNames":["fileSearchStores/docs-1"]}}]`, raw)
}

func TestLanguageModel_modelIDIsPathEscaped(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	model := f.provider().Model("tunedModels/a?b#c")
	if _, err := model.Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := f.last(t).Path; got != "/v1beta/models/tunedModels%2Fa%3Fb%23c:generateContent" && got != "/v1beta/models/tunedModels/a?b#c:generateContent" {
		t.Errorf("path = %q, want the id escaped", got)
	}
}

func TestLanguageModel_structuredOutput(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	model := f.provider().Model("models/gemini-3-flash")
	resp, err := model.Generate(t.Context(), &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Output:   &ai.OutputSpec{Name: "out", Schema: json.RawMessage(`{"$schema":"x","type":"object","properties":{"n":{"type":["integer","null"],"minimum":1},"s":{"type":"string","maxLength":3}},"required":["s"],"additionalProperties":false}`)},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if f.last(t).Path != "/v1beta/models/gemini-3-flash:generateContent" {
		t.Errorf("path = %q, want the models/ prefix trimmed", f.last(t).Path)
	}
	var warnings []string
	for _, w := range resp.Warnings {
		warnings = append(warnings, w.Message)
	}
	joined := strings.Join(warnings, "\n")
	for _, want := range []string{`"$schema"`, `"maxLength"`} {
		if !strings.Contains(joined, want) {
			t.Errorf("warnings do not mention %s:\n%s", want, joined)
		}
	}
	equalJSON(t, `{"responseMimeType":"application/json","responseJsonSchema":{"type":"object","properties":{"n":{"type":["integer","null"],"minimum":1},"s":{"type":"string"}},"required":["s"],"additionalProperties":false}}`, mustField(t, f.last(t).Body, "generationConfig"))

	openAPI := ai.Providers(google.Options{OpenAPISchemas: true})
	if _, err := model.Generate(t.Context(), &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Output:   &ai.OutputSpec{Schema: json.RawMessage(`{"type":"object","properties":{"n":{"type":["integer","null"],"minimum":1},"s":{"type":"string"}},"required":["s"],"additionalProperties":false}`)},
		Provider: openAPI,
	}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	equalJSON(t, `{"responseMimeType":"application/json","responseSchema":{"type":"OBJECT","properties":{"n":{"type":"INTEGER","nullable":true,"minimum":1},"s":{"type":"STRING"}},"required":["s"]}}`, mustField(t, f.last(t).Body, "generationConfig"))
}

func mustField(t *testing.T, body []byte, field string) []byte {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	return m[field]
}

func TestNew_environment(t *testing.T) {
	tests := []struct {
		name   string
		gemini string
		google string
		opt    []google.Option
		want   string
	}{
		{name: "gemini key first", gemini: "g1", google: "g2", want: "g1"},
		{name: "google key fallback", google: "g2", want: "g2"},
		{name: "option wins", gemini: "g1", opt: []google.Option{google.WithAPIKey("explicit")}, want: "explicit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GEMINI_API_KEY", tt.gemini)
			t.Setenv("GOOGLE_API_KEY", tt.google)
			f := newFakeServer(t, jsonReply(textReply))
			p := google.New(append([]google.Option{google.WithBaseURL(f.URL + "/v1beta")}, tt.opt...)...)
			if _, err := ai.GenerateText(t.Context(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)}); err != nil {
				t.Fatalf("GenerateText() error = %v", err)
			}
			if got := f.last(t).Header.Get("x-goog-api-key"); got != tt.want {
				t.Errorf("x-goog-api-key = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNew_baseURLFromEnvironment(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	t.Setenv("GOOGLE_GEMINI_BASE_URL", f.URL+"/custom")
	if _, err := ai.GenerateText(t.Context(), google.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if got := f.last(t).Path; got != "/custom/models/gemini-3-flash:generateContent" {
		t.Errorf("path = %q", got)
	}
}

func TestNew_headersAndLimiter(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	l := &countingLimiter{}
	p := f.provider(google.WithHeaders(http.Header{"X-Team": {"a"}}), google.WithLimiter(l))
	if _, err := ai.GenerateText(t.Context(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Headers: http.Header{"X-Call": {"b"}}, MaxRetries: new(0)}); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	h := f.last(t).Header
	if h.Get("X-Team") != "a" || h.Get("X-Call") != "b" || l.n != 1 {
		t.Errorf("headers = %v, limiter waits = %d", h, l.n)
	}
}

type countingLimiter struct{ n int }

func (l *countingLimiter) Wait(context.Context) error {
	l.n++
	return nil
}

func TestProvider_languageModel(t *testing.T) {
	m, err := google.New().LanguageModel("gemini-3-flash")
	if err != nil || m.Provider() != "google" || m.ModelID() != "gemini-3-flash" {
		t.Errorf("LanguageModel() = %v, %v", m, err)
	}
	if diff := cmp.Diff("google-vertex", google.New(google.WithVertex("p", "global")).Model("x").Provider()); diff != "" {
		t.Errorf("vertex provider name mismatch (-want +got):\n%s", diff)
	}
}

func TestShortcuts_useEnvironmentProvider(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "env-key")
	tests := []struct {
		name string
		got  string
		want string
	}{
		{name: "language", got: google.Model("m1").ModelID(), want: "m1"},
		{name: "embedding", got: google.NewEmbeddingModel("m2").ModelID(), want: "m2"},
		{name: "image", got: google.NewImageModel("m3").ModelID(), want: "m3"},
		{name: "video", got: google.NewVideoModel("m4").ModelID(), want: "m4"},
		{name: "speech", got: google.NewSpeechModel("m5").ModelID(), want: "m5"},
		{name: "realtime", got: google.NewRealtimeModel("m6").ModelID(), want: "m6"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("ModelID() = %q, want %q", tt.got, tt.want)
			}
		})
	}
}
