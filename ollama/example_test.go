package ollama_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ollama"
)

func exampleServer(response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
}

const exampleChat = `{"model":"llama3.2","created_at":"2026-11-17T09:00:00Z","message":{"role":"assistant","content":"Hello from Ollama."},"done":true,"done_reason":"stop","prompt_eval_count":9,"eval_count":5}`

const exampleEmbed = `{"model":"all-minilm","embeddings":[[0.5,0.25],[0.125,1]],"prompt_eval_count":6}`

func ExampleNew() {
	srv := exampleServer(exampleChat)
	defer srv.Close()

	p := ollama.New(ollama.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("llama3.2"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.Output)
	// Output:
	// Hello from Ollama.
	// 9 5
}

func ExampleModel() {
	srv := exampleServer(exampleChat)
	defer srv.Close()
	if err := os.Setenv("OLLAMA_HOST", srv.URL); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("OLLAMA_HOST")

	res, err := ai.GenerateText(context.Background(), ollama.Model("llama3.2"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Ollama.
}

func ExampleProvider_LanguageModel() {
	m, err := ollama.New().LanguageModel("llama3.2")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// ollama llama3.2
}

func ExampleOptions() {
	srv := exampleServer(exampleChat)
	defer srv.Close()
	model := ollama.New(ollama.WithBaseURL(srv.URL)).Model("qwen3")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages:  []ai.Message{ai.User("Say hello.")},
		Reasoning: ai.ReasoningMedium,
		Provider:  ai.Providers(ollama.Options{NumCtx: 8192, KeepAlive: "10m"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Ollama.
}

func ExampleWithAPIKey() {
	p := ollama.New(ollama.WithBaseURL(ollama.CloudBaseURL), ollama.WithAPIKey("your_api_key"))
	fmt.Println(p.Model("gpt-oss:120b").ModelID())
	// Output:
	// gpt-oss:120b
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(exampleEmbed)
	defer srv.Close()
	m := ollama.New(ollama.WithBaseURL(srv.URL)).EmbeddingModel("all-minilm")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{Texts: []string{"sky", "grass"}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Embeddings[1], res.Usage.Input)
	// Output:
	// 2 [0.125 1] 6
}

func ExampleProvider_Model() {
	m := ollama.New().Model("llama3.2")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// ollama llama3.2
}

func ExampleNewEmbeddingModel() {
	m := ollama.NewEmbeddingModel("all-minilm")
	fmt.Println(m.Provider(), m.ModelID(), m.MaxBatch())
	// Output:
	// ollama all-minilm 0
}

func ExampleEmbeddingOptions() {
	srv := exampleServer(exampleEmbed)
	defer srv.Close()
	m := ollama.New(ollama.WithBaseURL(srv.URL)).EmbeddingModel("all-minilm")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Texts:    []string{"sky", "grass"},
		Provider: ai.Providers(ollama.EmbeddingOptions{Truncate: new(false), KeepAlive: "5m"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings))
	// Output:
	// 2
}
