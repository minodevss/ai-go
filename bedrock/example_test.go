package bedrock_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
)

func exampleServer(status int, response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.Header().Set("X-Amzn-Errortype", "ThrottlingException:http://internal.amazon.com/coral/com.amazon.bedrock/")
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, response)
	}))
}

const exampleResponse = `{"output":{"message":{"role":"assistant","content":[{"text":"Hello from Converse."}]}},"stopReason":"end_turn","usage":{"inputTokens":9,"outputTokens":6,"totalTokens":19,"cacheReadInputTokens":4,"cacheWriteInputTokens":0},"metrics":{"latencyMs":212}}`

func ExampleNew() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()

	p := bedrock.New(
		bedrock.WithRegion("us-east-1"),
		bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"})),
		bedrock.WithBaseURL(srv.URL),
	)
	res, err := ai.GenerateText(context.Background(), p.Model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.CacheRead, res.Usage.Output)
	// Output:
	// Hello from Converse.
	// 13 4 6
}

func ExampleWithAPIKey() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()

	p := bedrock.New(bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-lite-v1:0"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Converse.
}

func ExampleModel() {
	for k, v := range map[string]string{"AWS_REGION": "us-east-1", "AWS_BEARER_TOKEN_BEDROCK": "bedrock-api-key-example"} {
		if err := os.Setenv(k, v); err != nil {
			fmt.Println(err)
			return
		}
		defer os.Unsetenv(k)
	}

	m := bedrock.Model("amazon.nova-pro-v1:0")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// bedrock amazon.nova-pro-v1:0
}

func ExampleProvider_LanguageModel() {
	m, err := bedrock.New().LanguageModel("amazon.nova-pro-v1:0")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// bedrock amazon.nova-pro-v1:0
}

func ExampleOptions() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))

	res, err := ai.GenerateText(context.Background(), p.Model("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), ai.TextRequest{
		Instructions: []ai.Text{bedrock.CachedTextFor("You are a support agent for Acme.", bedrock.TTL1Hour)},
		Messages:     []ai.Message{ai.User("Where is my order?")},
		Provider: ai.Providers(bedrock.Options{
			Guardrail:                    &bedrock.Guardrail{Identifier: "gr1abcd2efgh", Version: "1", Trace: bedrock.TraceEnabled},
			AdditionalModelRequestFields: json.RawMessage(`{"thinking":{"type":"enabled","budget_tokens":2048}}`),
			PerformanceLatency:           bedrock.LatencyOptimized,
			RequestMetadata:              map[string]string{"team": "support"},
		}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, _ := ai.ProviderValue[bedrock.ResponseData](res.Provider, "bedrock")
	fmt.Println(res.Text)
	fmt.Println(string(data.Metrics))
	// Output:
	// Hello from Converse.
	// {"latencyMs":212}
}

func ExampleCachedText() {
	system := []ai.Text{
		bedrock.CachedText("Long, stable policy text that is worth caching."),
		{Text: "Today's date changes every request."},
	}
	fmt.Println(len(system), len(system[0].Provider), len(system[1].Provider))
	// Output:
	// 2 1 0
}

func ExampleCachePoint() {
	doc := ai.Text{Text: "A very long document.", Provider: bedrock.CachePoint(bedrock.TTL5Minutes)}
	fmt.Println(string(doc.Provider["bedrock"]))
	// Output:
	// {"cachePoint":{"ttl":"5m"}}
}

func ExampleProvider_EmbeddingModel() {
	srv := exampleServer(http.StatusOK, `{"embedding":[0.25,0.5,0.75],"inputTextTokenCount":4}`)
	defer srv.Close()
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))

	res, err := ai.Embed(context.Background(), p.EmbeddingModel("amazon.titan-embed-text-v2:0"), ai.EmbedRequest{
		Texts:      []string{"Where is my order?"},
		Dimensions: 256,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0], res.Usage.Input)
	// Output:
	// [0.25 0.5 0.75] 4
}

func ExampleCredentialsFunc() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()
	assumed := bedrock.CredentialsFunc(func(context.Context) (bedrock.Keys, error) {
		return bedrock.Keys{AccessKeyID: "ASIAEXAMPLE", SecretAccessKey: "secret", SessionToken: "session-token"}, nil
	})

	p := bedrock.New(bedrock.WithRegion("eu-central-1"), bedrock.WithCredentials(assumed), bedrock.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-micro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Converse.
}

func ExampleErrNoCredentials() {
	for _, k := range []string{"AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"} {
		if err := os.Setenv(k, ""); err != nil {
			fmt.Println(err)
			return
		}
		defer os.Unsetenv(k)
	}
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithBaseURL("http://127.0.0.1:1"))
	_, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{
		Messages:   []ai.Message{ai.User("hi")},
		MaxRetries: new(0),
	})
	fmt.Println(errors.Is(err, bedrock.ErrNoCredentials))
	// Output:
	// true
}

func ExampleModel_throttling() {
	srv := exampleServer(http.StatusTooManyRequests, `{"message":"Too many requests, please wait before trying again."}`)
	defer srv.Close()
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))

	_, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{
		Messages:   []ai.Message{ai.User("hi")},
		MaxRetries: new(0),
	})
	if apiErr, ok := errors.AsType[*ai.APIError](err); ok {
		fmt.Println(apiErr.Code, apiErr.StatusCode, apiErr.Retryable(), errors.Is(err, ai.ErrRateLimit))
	}
	// Output:
	// ThrottlingException 429 true true
}

func ExampleStaticCredentials() {
	creds := bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret", SessionToken: "session-token"})
	keys, err := creds.Retrieve(context.Background())
	fmt.Println(keys.AccessKeyID, keys.SessionToken, err)
	// Output:
	// AKIDEXAMPLE session-token <nil>
}

func ExampleEnvCredentials() {
	for k, v := range map[string]string{"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "secret", "AWS_SESSION_TOKEN": ""} {
		if err := os.Setenv(k, v); err != nil {
			fmt.Println(err)
			return
		}
		defer os.Unsetenv(k)
	}
	keys, err := bedrock.EnvCredentials().Retrieve(context.Background())
	fmt.Println(keys.AccessKeyID, keys.SessionToken == "", err)
	// Output:
	// AKIDEXAMPLE true <nil>
}

func ExampleSystemTool() {
	tool := bedrock.SystemTool("nova_grounding")
	fmt.Println(tool.Name, string(tool.Provider["bedrock"]))
	// Output:
	// nova_grounding {"systemTool":"nova_grounding"}
}

func ExampleEmbeddingOptions() {
	srv := exampleServer(http.StatusOK, `{"id":"1","embeddings":[[0.5,0.25]],"response_type":"embeddings_floats"}`)
	defer srv.Close()
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))

	res, err := ai.Embed(context.Background(), p.EmbeddingModel("cohere.embed-english-v3"), ai.EmbedRequest{
		Texts:     []string{"cheap flights to Busan"},
		InputType: ai.InputQuery,
		Provider:  ai.Providers(bedrock.EmbeddingOptions{Truncate: "END"}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0])
	// Output:
	// [0.5 0.25]
}

func ExampleWithLimiter() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()
	waits := 0
	limiter := limiterFunc(func(context.Context) error { waits++; return nil })
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL), bedrock.WithLimiter(limiter))

	if _, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(waits)
	// Output:
	// 1
}

func ExampleWithRegion() {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		fmt.Println(r.URL.Host)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(exampleResponse)), Request: r}, nil
	})}

	p := bedrock.New(bedrock.WithRegion("eu-west-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithHTTPClient(client))
	if _, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// bedrock-runtime.eu-west-1.amazonaws.com
}

func ExampleWithCredentials() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(strings.HasPrefix(r.Header.Get("Authorization"), "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleResponse)
	}))
	defer srv.Close()

	creds := bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"})
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithCredentials(creds), bedrock.WithBaseURL(srv.URL))
	if _, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// true
}

func ExampleWithBaseURL() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()

	p := bedrock.New(bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Converse.
}

func ExampleWithHTTPClient() {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		fmt.Println(r.URL.Host)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(exampleResponse)), Request: r}, nil
	})}

	p := bedrock.New(bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL("http://gateway.example"), bedrock.WithHTTPClient(client))
	if _, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// gateway.example
}

func ExampleWithHeaders() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("X-Team"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleResponse)
	}))
	defer srv.Close()

	p := bedrock.New(bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL), bedrock.WithRegion("us-east-1"), bedrock.WithHeaders(http.Header{"X-Team": {"search"}}))
	if _, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// search
}

func ExampleProvider_Model() {
	m := bedrock.New(bedrock.WithRegion("us-east-1")).Model("amazon.nova-pro-v1:0")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// bedrock amazon.nova-pro-v1:0
}

func ExampleNewEmbeddingModel() {
	if err := os.Setenv("AWS_REGION", "us-east-1"); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("AWS_REGION")

	m := bedrock.NewEmbeddingModel("amazon.titan-embed-text-v2:0")
	fmt.Println(m.Provider(), m.ModelID(), m.MaxBatch())
	// Output:
	// bedrock amazon.titan-embed-text-v2:0 1
}

func ExampleEmbeddingModel_Embed() {
	srv := exampleServer(http.StatusOK, `{"embedding":[0.5,0.25],"inputTextTokenCount":2}`)
	defer srv.Close()

	m := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL)).EmbeddingModel("amazon.titan-embed-text-v2:0")
	res, err := m.Embed(context.Background(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "Where is my order?"}}}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Embeddings[0], res.Usage.Input)
	// Output:
	// [0.5 0.25] 2
}

func ExampleCachedTextFor() {
	t := bedrock.CachedTextFor("Long policy.", bedrock.TTL1Hour)
	fmt.Println(t.Text, string(t.Provider["bedrock"]))
	// Output:
	// Long policy. {"cachePoint":{"ttl":"1h"}}
}

func ExampleLanguageModel_Generate() {
	srv := exampleServer(http.StatusOK, exampleResponse)
	defer srv.Close()

	m := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL)).Model("amazon.nova-pro-v1:0")
	res, err := m.Generate(context.Background(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	text, _ := res.Content[0].(ai.Text)
	fmt.Println(text.Text, res.FinishReason)
	// Output:
	// Hello from Converse. stop
}

func ExampleLanguageModel_Stream() {
	frames, err := encodeEventsErr(textStream("Hello"))
	if err != nil {
		fmt.Println(err)
		return
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(frames)
	}))
	defer srv.Close()

	m := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithAPIKey("bedrock-api-key-example"), bedrock.WithBaseURL(srv.URL)).Model("amazon.nova-pro-v1:0")
	stream := ai.StreamText(context.Background(), m, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	for text := range stream.Text() {
		fmt.Println(text)
	}
	if err := stream.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// Hello
}
