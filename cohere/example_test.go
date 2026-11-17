package cohere_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/cohere"
)

func exampleServer(responses map[string]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, responses[r.URL.Path])
	}))
}

const exampleChat = `{"id":"c1","finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"thinking","thinking":"Greet the user."},{"type":"text","text":"Hello from Command."}]},"usage":{"billed_units":{"input_tokens":5,"output_tokens":4},"tokens":{"input_tokens":9,"output_tokens":6},"cached_tokens":3}}`

const exampleEmbed = `{"id":"e1","embeddings":{"float":[[0.1,0.2],[0.3,0.4]]},"meta":{"billed_units":{"input_tokens":7}}}`

const exampleRerank = `{"id":"r1","results":[{"index":1,"relevance_score":0.93},{"index":0,"relevance_score":0.12}],"meta":{"billed_units":{"search_units":1}}}`

func ExampleNew() {
	srv := exampleServer(map[string]string{"/v2/chat": exampleChat})
	defer srv.Close()

	p := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("command-a-reasoning-08-2025"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.CacheRead, res.Usage.Output)
	// Output:
	// Hello from Command.
	// 9 3 6
}

func ExampleModel() {
	srv := exampleServer(map[string]string{"/v2/chat": exampleChat})
	defer srv.Close()
	if err := os.Setenv("CO_API_URL", srv.URL); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("CO_API_URL")

	res, err := ai.GenerateText(context.Background(), cohere.Model("command-a-03-2025"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Command.
}

func ExampleProvider_LanguageModel() {
	m, err := cohere.New().LanguageModel("command-a-03-2025")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// cohere command-a-03-2025
}

func ExampleOptions() {
	srv := exampleServer(map[string]string{"/v2/chat": exampleChat})
	defer srv.Close()
	model := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL)).Model("command-a-reasoning-08-2025")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("Summarize the document.")},
		Provider: ai.Providers(cohere.Options{
			Thinking:     &cohere.Thinking{Type: cohere.ThinkingEnabled, TokenBudget: 4096},
			CitationMode: cohere.CitationsAccurate,
			Documents:    []cohere.Document{{Data: map[string]any{"title": "Notes", "text": "Cohere serves Command models."}}},
		}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, _ := ai.ProviderValue[cohere.ResponseData](res.Provider, "cohere")
	fmt.Println(data.ID)
	// Output:
	// c1
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(map[string]string{"/v2/embed": exampleEmbed})
	defer srv.Close()
	m := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL)).EmbeddingModel("embed-v4.0")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Texts:     []string{"first document", "second document"},
		InputType: ai.InputDocument,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Embeddings[1], res.Usage.Input)
	// Output:
	// 2 [0.3 0.4] 7
}

func ExampleProvider_RerankModel() {
	srv := exampleServer(map[string]string{"/v2/rerank": exampleRerank})
	defer srv.Close()
	m := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL)).RerankModel("rerank-v3.5")

	res, err := ai.Rerank(context.Background(), m, ai.RerankRequest{
		Query:     "capital of France",
		Documents: []string{"Berlin is in Germany.", "Paris is the capital of France."},
		TopN:      1,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Ranking[0].Index, res.Ranking[0].Score)
	// Output:
	// 1 0.93
}

func ExampleProvider_Model() {
	m := cohere.New(cohere.WithAPIKey("co-test")).Model("command-a-03-2025")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// cohere command-a-03-2025
}

func ExampleNewEmbeddingModel() {
	m := cohere.NewEmbeddingModel("embed-v4.0")
	fmt.Println(m.Provider(), m.ModelID(), m.MaxBatch())
	// Output:
	// cohere embed-v4.0 96
}

func ExampleNewRerankModel() {
	m := cohere.NewRerankModel("rerank-v3.5")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// cohere rerank-v3.5
}

func ExampleEmbeddingOptions() {
	srv := exampleServer(map[string]string{"/v2/embed": exampleEmbed})
	defer srv.Close()
	m := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL)).EmbeddingModel("embed-v4.0")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Texts:    []string{"a", "b"},
		Provider: ai.Providers(cohere.EmbeddingOptions{InputType: cohere.InputClustering, Truncate: cohere.TruncateEnd}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Usage.Input)
	// Output:
	// 2 7
}

func ExampleRerankOptions() {
	srv := exampleServer(map[string]string{"/v2/rerank": exampleRerank})
	defer srv.Close()
	m := cohere.New(cohere.WithAPIKey("co-test"), cohere.WithBaseURL(srv.URL)).RerankModel("rerank-v3.5")

	res, err := ai.Rerank(context.Background(), m, ai.RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b"},
		Provider:  ai.Providers(cohere.RerankOptions{MaxTokensPerDoc: 1024}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Ranking))
	// Output:
	// 2
}

func ExampleWithHeaders() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("X-Team"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleChat)
	}))
	defer srv.Close()
	p := cohere.New(cohere.WithBaseURL(srv.URL), cohere.WithHeaders(http.Header{"X-Team": {"search"}}))

	if _, err := ai.GenerateText(context.Background(), p.Model("command-a-03-2025"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// search
}
