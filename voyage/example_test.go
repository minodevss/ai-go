package voyage_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/voyage"
)

func exampleServer(response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
}

const (
	exampleEmbeddings = `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0},{"object":"embedding","embedding":[0.3,0.4],"index":1}],"model":"voyage-4","usage":{"total_tokens":9}}`
	exampleRerank     = `{"object":"list","data":[{"relevance_score":0.91,"index":1},{"relevance_score":0.12,"index":0}],"model":"rerank-2.5","usage":{"total_tokens":14}}`
)

func ExampleNew() {
	srv := exampleServer(exampleEmbeddings)
	defer srv.Close()

	p := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL))
	res, err := ai.Embed(context.Background(), p.EmbeddingModel("voyage-4"), ai.EmbedRequest{
		Texts:      []string{"Go is a programming language.", "Gophers dig burrows."},
		InputType:  ai.InputDocument,
		Dimensions: 1024,
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
	m := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL)).EmbeddingModel("voyage-4")
	fmt.Println(m.Provider(), m.ModelID(), m.MaxBatch())
	// Output:
	// voyage voyage-4 1000
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(`{"data":[{"embedding":[0.5,0.6],"index":0}],"model":"voyage-multimodal-3.5","usage":{"text_tokens":4,"image_pixels":10000,"total_tokens":22}}`)
	defer srv.Close()
	m := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL)).EmbeddingModel("voyage-multimodal-3.5")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Inputs: []ai.Parts{{
			ai.Text{Text: "A red bicycle"},
			ai.File{MediaType: "image/jpeg", URL: "https://example.com/bike.jpg"},
		}},
		InputType: ai.InputQuery,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0], res.Usage.Input)
	// Output:
	// [0.5 0.6] 22
}

func ExampleProvider_RerankModel() {
	srv := exampleServer(exampleRerank)
	defer srv.Close()
	m := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL)).RerankModel("rerank-2.5")

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
	m := voyage.New(voyage.WithBaseURL(srv.URL)).RerankModel("rerank-2.5-lite")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// voyage rerank-2.5-lite
}

func ExampleRerankOptions() {
	srv := exampleServer(exampleRerank)
	defer srv.Close()
	m := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL)).RerankModel("rerank-2.5")

	res, err := ai.Rerank(context.Background(), m, ai.RerankRequest{
		Query:     "q",
		Documents: []string{"a", "b"},
		Provider:  ai.Providers(voyage.RerankOptions{Truncation: new(false)}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Ranking))
	// Output:
	// 2
}

func ExampleEmbeddingOptions() {
	srv := exampleServer(exampleEmbeddings)
	defer srv.Close()
	m := voyage.New(voyage.WithAPIKey("pa-test"), voyage.WithBaseURL(srv.URL)).EmbeddingModel("voyage-4")

	res, err := ai.Embed(context.Background(), m, ai.EmbedRequest{
		Texts:    []string{"a", "b"},
		Provider: ai.Providers(voyage.EmbeddingOptions{Truncation: new(false)}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Embeddings), res.Usage.Input)
	// Output:
	// 2 9
}

func ExampleNewEmbeddingModel() {
	m := voyage.NewEmbeddingModel("voyage-4")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// voyage voyage-4
}

func ExampleNewRerankModel() {
	m := voyage.NewRerankModel("rerank-2.5")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// voyage rerank-2.5
}
