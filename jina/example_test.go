package jina_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/jina"
)

func exampleServer(response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
}

const (
	exampleEmbeddings = `{"model":"jina-embeddings-v3","object":"list","usage":{"total_tokens":9,"prompt_tokens":9},"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]},{"object":"embedding","index":1,"embedding":[0.3,0.4]}]}`
	exampleRerank     = `{"model":"jina-reranker-v3","usage":{"total_tokens":14},"results":[{"index":1,"relevance_score":0.91},{"index":0,"relevance_score":0.12}]}`
)

func ExampleNew() {
	srv := exampleServer(exampleEmbeddings)
	defer srv.Close()

	p := jina.New(jina.WithAPIKey("jina_test"), jina.WithBaseURL(srv.URL))
	res, err := ai.Embed(context.Background(), p.EmbeddingModel("jina-embeddings-v3"), ai.EmbedRequest{
		Texts:      []string{"Go is a programming language.", "Gophers dig burrows."},
		InputType:  ai.InputDocument,
		Dimensions: 512,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Embeddings[1], res.Usage.Input)
	// Output:
	// 2 [0.3 0.4] 9
}

func ExampleEmbeddingModel() {
	srv := exampleServer(exampleEmbeddings)
	defer srv.Close()
	m := jina.New(jina.WithBaseURL(srv.URL)).EmbeddingModel("jina-embeddings-v3")
	fmt.Println(m.Provider(), m.ModelID(), m.MaxBatch())
	// Output:
	// jina jina-embeddings-v3 0
}

func ExampleEmbeddingOptions() {
	srv := exampleServer(`{"data":[{"index":0,"embedding":[0.5]},{"index":1,"embedding":[0.6]}],"usage":{"total_tokens":20}}`)
	defer srv.Close()
	m := jina.New(jina.WithAPIKey("jina_test"), jina.WithBaseURL(srv.URL)).EmbeddingModel("jina-embeddings-v3")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Texts:     []string{"First chunk of a long document.", "Second chunk that refers to it."},
		InputType: ai.InputDocument,
		Provider:  ai.Providers(jina.EmbeddingOptions{LateChunking: new(true), Truncate: new(true)}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Usage.Input)
	// Output:
	// 2 20
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(`{"data":[{"index":0,"embedding":[0.7,0.8]}],"usage":{"total_tokens":31,"prompt_tokens":4,"image_tokens":27}}`)
	defer srv.Close()
	m := jina.New(jina.WithAPIKey("jina_test"), jina.WithBaseURL(srv.URL)).EmbeddingModel("jina-embeddings-v5-omni-small")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Inputs: []ai.Parts{{
			ai.Text{Text: "A red bicycle"},
			ai.File{MediaType: "image/jpeg", URL: "https://example.com/bike.jpg"},
		}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0], res.Usage.Input)
	// Output:
	// [0.7 0.8] 31
}

func ExampleProvider_RerankModel() {
	srv := exampleServer(exampleRerank)
	defer srv.Close()
	m := jina.New(jina.WithAPIKey("jina_test"), jina.WithBaseURL(srv.URL)).RerankModel("jina-reranker-v3")

	res, err := ai.Rerank(context.Background(), m, ai.RerankRequest{
		Query:     "How do I install Go?",
		Documents: []string{"Gophers are rodents.", "Download Go from go.dev/dl."},
		TopN:      2,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, r := range res.Ranking {
		fmt.Println(r.Index, r.Score)
	}
	// Output:
	// 1 0.91
	// 0 0.12
}

func ExampleRerankModel() {
	srv := exampleServer(exampleRerank)
	defer srv.Close()
	m := jina.New(jina.WithBaseURL(srv.URL)).RerankModel("jina-reranker-v3.5")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// jina jina-reranker-v3.5
}

func ExampleRerankOptions() {
	srv := exampleServer(exampleRerank)
	defer srv.Close()
	m := jina.New(jina.WithAPIKey("jina_test"), jina.WithBaseURL(srv.URL)).RerankModel("jina-reranker-v3")

	res, err := ai.Rerank(context.Background(), m, ai.RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b"},
		Provider:  ai.Providers(jina.RerankOptions{MaxDocLength: 4096}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Ranking))
	// Output:
	// 2
}

func ExampleNewEmbeddingModel() {
	m := jina.NewEmbeddingModel("jina-embeddings-v3")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// jina jina-embeddings-v3
}

func ExampleNewRerankModel() {
	m := jina.NewRerankModel("jina-reranker-v3")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// jina jina-reranker-v3
}
