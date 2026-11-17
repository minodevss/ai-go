package openaicompat_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openaicompat"
)

func exampleServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
}

func exampleChat(text string) string {
	return `{"id":"c1","choices":[{"message":{"role":"assistant","content":"` + text + `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`
}

func say(model ai.LanguageModel) {
	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{Messages: []ai.Message{ai.User("Say hello.")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
}

func ExampleNew() {
	srv := exampleServer(exampleChat("Hello from a local server."))
	defer srv.Close()

	p := openaicompat.New(srv.URL+"/v1", openaicompat.WithName("local"))
	say(p.Model("qwen3"))
	// Output:
	// Hello from a local server.
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(`{"data":[{"index":0,"embedding":[0.25,0.5]},{"index":1,"embedding":[0.75,1]}],"usage":{"prompt_tokens":4,"total_tokens":4}}`)
	defer srv.Close()

	p := openaicompat.New(srv.URL + "/v1")
	res, err := ai.Embed(context.Background(), p.EmbeddingModel("bge-m3"), ai.EmbedRequest{Texts: []string{"first", "second"}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), len(res.Embeddings[0]))
	// Output:
	// 2 2
}

func ExampleProvider_Model() {
	srv := exampleServer(exampleChat("Hello from a model."))
	defer srv.Close()

	model := openaicompat.New(srv.URL).Model("any-model")
	fmt.Println(model.Provider(), model.ModelID())
	say(model)
	// Output:
	// openaicompat any-model
	// Hello from a model.
}

func ExampleWithTokenSource() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("Authorization"))
		_, _ = io.WriteString(w, exampleChat("Authenticated."))
	}))
	defer srv.Close()

	token := func(context.Context) (string, error) { return "entra-token", nil }
	say(openaicompat.New(srv.URL, openaicompat.WithTokenSource(token)).Model("m"))
	// Output:
	// Bearer entra-token
	// Authenticated.
}

func ExampleAzure() {
	srv := exampleServer(exampleChat("Hello from Azure."))
	defer srv.Close()

	p := openaicompat.Azure("my-resource", openaicompat.WithAPIKey("azure-key"), openaicompat.WithBaseURL(srv.URL+"/openai/v1/"))
	say(p.Model("my-gpt-deployment"))
	// Output:
	// Hello from Azure.
}

func ExampleMistral() {
	srv := exampleServer(exampleChat("Hello from Mistral."))
	defer srv.Close()

	p := openaicompat.Mistral(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/v1"))
	res, err := ai.GenerateText(context.Background(), p.Model("mistral-large-latest"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
		Provider: ai.Providers(openaicompat.MistralOptions{SafePrompt: new(true)}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Mistral.
}

func ExampleDeepSeek() {
	srv := exampleServer(exampleChat("Hello from DeepSeek."))
	defer srv.Close()

	p := openaicompat.DeepSeek(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("deepseek-v4-pro"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
		Provider: ai.Providers(openaicompat.DeepSeekOptions{Thinking: openaicompat.DeepSeekThinkingDisabled}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from DeepSeek.
}

func ExamplePerplexity() {
	srv := exampleServer(`{"id":"p1","citations":["https://example.test/a"],"choices":[{"message":{"role":"assistant","content":"Hello [1]."},"finish_reason":"stop"}]}`)
	defer srv.Close()

	p := openaicompat.Perplexity(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("sonar"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
		Provider: ai.Providers(openaicompat.PerplexityOptions{SearchRecencyFilter: "week"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range res.Content {
		if text, ok := p.(ai.Text); ok {
			fmt.Println(text.Text, text.Citations[0].URL)
		}
	}
	// Output:
	// Hello [1]. https://example.test/a
}

func ExampleOpenRouter() {
	srv := exampleServer(exampleChat("Hello from OpenRouter."))
	defer srv.Close()

	p := openaicompat.OpenRouter(
		openaicompat.WithAPIKey("key"),
		openaicompat.WithAttribution("https://example.test", "Example"),
		openaicompat.WithBaseURL(srv.URL+"/api/v1"),
	)
	res, err := ai.GenerateText(context.Background(), p.Model("anthropic/claude-sonnet-4.5"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
		Provider: ai.Providers(openaicompat.OpenRouterOptions{
			Provider: &openaicompat.OpenRouterRouting{Order: []string{"anthropic"}, AllowFallbacks: new(false)},
		}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from OpenRouter.
}

func ExampleWithAttribution() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("HTTP-Referer"), r.Header.Get("X-OpenRouter-Title"))
		_, _ = io.WriteString(w, exampleChat("ok"))
	}))
	defer srv.Close()

	say(openaicompat.OpenRouter(openaicompat.WithBaseURL(srv.URL), openaicompat.WithAttribution("https://example.test", "Example")).Model("m"))
	// Output:
	// https://example.test Example
	// ok
}

func ExampleXAI() {
	srv := exampleServer(exampleChat("Hello from Grok."))
	defer srv.Close()

	p := openaicompat.XAI(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/v1"))
	say(p.Model("grok-4.7"))
	// Output:
	// Hello from Grok.
}

func ExampleGroq() {
	srv := exampleServer(exampleChat("Hello from Groq."))
	defer srv.Close()

	p := openaicompat.Groq(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/openai/v1"))
	say(p.Model("openai/gpt-oss-20b"))
	// Output:
	// Hello from Groq.
}

func ExampleTogether() {
	srv := exampleServer(exampleChat("Hello from Together."))
	defer srv.Close()

	p := openaicompat.Together(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/v1"))
	say(p.Model("openai/gpt-oss-120b"))
	// Output:
	// Hello from Together.
}

func ExampleFireworks() {
	srv := exampleServer(exampleChat("Hello from Fireworks."))
	defer srv.Close()

	p := openaicompat.Fireworks(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/inference/v1"))
	say(p.Model("accounts/fireworks/models/kimi-k2-instruct"))
	// Output:
	// Hello from Fireworks.
}

func ExampleCerebras() {
	srv := exampleServer(exampleChat("Hello from Cerebras."))
	defer srv.Close()

	p := openaicompat.Cerebras(openaicompat.WithAPIKey("key"), openaicompat.WithBaseURL(srv.URL+"/v1"))
	say(p.Model("gpt-oss-120b"))
	// Output:
	// Hello from Cerebras.
}

func ExampleVLLM() {
	srv := exampleServer(exampleChat("Hello from vLLM."))
	defer srv.Close()

	p := openaicompat.VLLM(srv.URL + "/v1")
	say(p.Model("Qwen/Qwen3-8B"))
	// Output:
	// Hello from vLLM.
}

func ExampleProvider_LanguageModel() {
	m, err := openaicompat.New("http://localhost:8000/v1").LanguageModel("qwen3")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// openaicompat qwen3
}

type pausingLimiter struct{}

func (pausingLimiter) Wait(context.Context) error {
	fmt.Println("waiting for a slot")
	return nil
}

func ExampleWithLimiter() {
	srv := exampleServer(exampleChat("Hello after the wait."))
	defer srv.Close()

	p := openaicompat.New(srv.URL, openaicompat.WithLimiter(pausingLimiter{}))
	say(p.Model("m"))
	// Output:
	// waiting for a slot
	// Hello after the wait.
}
