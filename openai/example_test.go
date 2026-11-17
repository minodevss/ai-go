package openai_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func exampleServer(responses map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, responses[r.URL.Path])
	}))
}

const exampleResponse = `{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello from the Responses API.","annotations":[]}]}],"usage":{"input_tokens":9,"output_tokens":6}}`

func ExampleNew() {
	srv := exampleServer(map[string]string{"/v1/responses": exampleResponse})
	defer srv.Close()

	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))
	res, err := ai.GenerateText(context.Background(), p.Model("gpt-5-mini"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.Output)
	// Output:
	// Hello from the Responses API.
	// 9 6
}

func ExampleModel() {
	srv := exampleServer(map[string]string{"/v1/responses": exampleResponse})
	defer srv.Close()
	if err := os.Setenv("OPENAI_BASE_URL", srv.URL+"/v1"); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("OPENAI_BASE_URL")

	res, err := ai.GenerateText(context.Background(), openai.Model("gpt-5-mini"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from the Responses API.
}

func ExampleOptions() {
	srv := exampleServer(map[string]string{"/v1/responses": exampleResponse})
	defer srv.Close()
	model := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1")).Model("gpt-5-mini")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages:  []ai.Message{ai.User("Think, then answer.")},
		Reasoning: ai.ReasoningLow,
		Provider:  ai.Providers(openai.Options{Store: new(false), ReasoningSummary: "auto", PromptCacheKey: "support-bot"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, _ := ai.ProviderValue[openai.ResponseData](res.Provider, "openai")
	fmt.Println(data.ID)
	// Output:
	// resp_1
}

func ExampleProvider_ChatModel() {
	srv := exampleServer(map[string]string{"/v1/chat/completions": `{"id":"c1","choices":[{"message":{"role":"assistant","content":"Hello from Chat Completions."},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateText(context.Background(), p.ChatModel("gpt-4.1-mini"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Chat Completions.
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(map[string]string{"/v1/embeddings": `{"data":[{"index":0,"embedding":[0.25,0.5]},{"index":1,"embedding":[0.75,1]}],"usage":{"prompt_tokens":4,"total_tokens":4}}`})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.Embed(context.Background(), p.EmbeddingModel("text-embedding-3-small"), ai.EmbedRequest{
		Texts: []string{"first", "second"},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Embeddings[1])
	// Output:
	// 2 [0.75 1]
}

func ExampleWebSearch() {
	srv := exampleServer(map[string]string{"/v1/responses": `{"id":"resp_2","status":"completed","output":[
		{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"go 1.26 release"}},
		{"id":"msg_2","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Go 1.26 shipped in February.","annotations":[{"type":"url_citation","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","start_index":0,"end_index":8}]}]}]}`})
	defer srv.Close()
	model := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1")).Model("gpt-5-mini")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("When did Go 1.26 ship?")},
		Tools:    []ai.Tool{openai.WebSearch(openai.WebSearchOptions{SearchContextSize: "low"})},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range res.Content {
		switch p := p.(type) {
		case ai.ToolCall:
			fmt.Println("searched:", p.Name, p.ProviderExecuted)
		case ai.Text:
			fmt.Println(p.Text, "-", p.Citations[0].URL)
		}
	}
	// Output:
	// searched: web_search true
	// Go 1.26 shipped in February. - https://go.dev/doc/go1.26
}

func ExampleFileID() {
	srv := exampleServer(map[string]string{"/v1/responses": exampleResponse})
	defer srv.Close()
	model := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1")).Model("gpt-5-mini")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.UserParts(
			ai.Text{Text: "Summarize the file."},
			ai.File{MediaType: "application/pdf", Provider: openai.FileID("file-abc123")},
		)},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.FinishReason)
	// Output:
	// stop
}

func ExampleProvider_Model() {
	srv := exampleServer(map[string]string{"/v1/responses": exampleResponse})
	defer srv.Close()
	p := openai.New(openai.WithAPIKey("sk-test"), openai.WithBaseURL(srv.URL+"/v1"))

	res, err := ai.GenerateText(context.Background(), p.Model("gpt-5-mini"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from the Responses API.
}

func ExampleProvider_LanguageModel() {
	m, err := openai.New(openai.WithAPIKey("sk-test")).LanguageModel("gpt-5-mini")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// openai gpt-5-mini
}

func useServer(srv *httptest.Server) func() {
	os.Setenv("OPENAI_BASE_URL", srv.URL+"/v1")
	os.Setenv("OPENAI_API_KEY", "sk-test")
	return func() {
		os.Unsetenv("OPENAI_BASE_URL")
		os.Unsetenv("OPENAI_API_KEY")
	}
}

func ExampleNewChatModel() {
	srv := exampleServer(map[string]string{"/v1/chat/completions": `{"id":"c1","choices":[{"message":{"role":"assistant","content":"Hello from Chat Completions."},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.GenerateText(context.Background(), openai.NewChatModel("gpt-4.1-mini"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Chat Completions.
}

func ExampleNewEmbeddingModel() {
	srv := exampleServer(map[string]string{"/v1/embeddings": `{"data":[{"index":0,"embedding":[0.25,0.5]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`})
	defer srv.Close()
	defer useServer(srv)()

	res, err := ai.Embed(context.Background(), openai.NewEmbeddingModel("text-embedding-3-small"), ai.EmbedRequest{Texts: []string{"first"}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0])
	// Output:
	// [0.25 0.5]
}
