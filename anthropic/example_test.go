package anthropic_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func exampleServer(response string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
}

const exampleMessage = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[{"type":"text","text":"Hello from the Messages API."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":9,"cache_read_input_tokens":4,"output_tokens":6}}`

func ExampleNew() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL), anthropic.WithBeta("context-management-2025-06-27"))
	res, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.Usage.Input, res.Usage.CacheRead, res.Usage.Output)
	// Output:
	// Hello from the Messages API.
	// 13 4 6
}

func ExampleModel() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	if err := os.Setenv("ANTHROPIC_BASE_URL", srv.URL); err != nil {
		fmt.Println(err)
		return
	}
	defer os.Unsetenv("ANTHROPIC_BASE_URL")

	res, err := ai.GenerateText(context.Background(), anthropic.Model("claude-sonnet-5-5"), ai.TextRequest{
		Messages: []ai.Message{ai.User("Say hello.")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from the Messages API.
}

func ExampleProvider_LanguageModel() {
	m, err := anthropic.New().LanguageModel("claude-haiku-5-5")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// anthropic claude-haiku-5-5
}

func ExampleOptions() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-opus-5-5")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Instructions: []ai.Text{anthropic.CachedText("You are a support agent for Acme.")},
		Messages:     []ai.Message{ai.User("Where is my order?")},
		Reasoning:    ai.ReasoningHigh,
		Provider: ai.Providers(anthropic.Options{
			CacheControl: &anthropic.CacheControl{TTL: anthropic.TTL1Hour},
			Thinking:     &anthropic.Thinking{Type: anthropic.ThinkingAdaptive, Display: anthropic.DisplaySummarized},
		}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	data, _ := ai.ProviderValue[anthropic.ResponseData](res.Provider, "anthropic")
	fmt.Println(data.ID)
	// Output:
	// msg_1
}

func ExampleCachedText() {
	t := anthropic.CachedText("Long, stable instructions.")
	fmt.Println(t.Text, string(t.Provider["anthropic"]))
	// Output:
	// Long, stable instructions. {"cacheControl":{}}
}

func ExampleCachedTextFor() {
	t := anthropic.CachedTextFor("Reference manual.", anthropic.TTL1Hour)
	fmt.Println(string(t.Provider["anthropic"]))
	// Output:
	// {"cacheControl":{"ttl":"1h"}}
}

func ExampleFileID() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.UserParts(
			ai.Text{Text: "Summarize the file and cite it."},
			ai.File{MediaType: "application/pdf", Citations: true, Provider: anthropic.FileID("file_011CNha8iCJcU1wXNR6q4V8w")},
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

const searchMessage = `{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[
	{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"go 1.26 release"}},
	{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","encrypted_content":"ENC","page_age":null}]},
	{"type":"text","text":"Go 1.26 shipped in February.","citations":[{"type":"web_search_result_location","url":"https://go.dev/doc/go1.26","title":"Go 1.26 Release Notes","encrypted_index":"IDX","cited_text":"Go 1.26 is released"}]}],
	"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":50,"output_tokens":20,"server_tool_use":{"web_search_requests":1}}}`

func ExampleWebSearch() {
	srv := exampleServer(searchMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("When did Go 1.26 ship?")},
		Tools:    []ai.Tool{anthropic.WebSearch(anthropic.WebSearchOptions{MaxUses: 2, AllowedDomains: []string{"go.dev"}})},
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

func ExampleWebFetch() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")

	tool := anthropic.WebFetch(anthropic.WebFetchOptions{Citations: true, MaxUses: 1})
	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("Summarize https://go.dev/doc/go1.26")},
		Tools:    []ai.Tool{tool},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tool.Name, res.FinishReason)
	// Output:
	// web_fetch stop
}

func ExampleCodeExecution() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")

	tool := anthropic.CodeExecution()
	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("What is the standard deviation of 1..10?")},
		Tools:    []ai.Tool{tool},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tool.Name, res.FinishReason)
	// Output:
	// code_execution stop
}

func ExampleMCPServer() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("What is on my calendar today?")},
		Provider: ai.Providers(anthropic.Options{MCPServers: []anthropic.MCPServer{{
			Name:         "calendar",
			URL:          "https://mcp.example.com/sse",
			AllowedTools: []string{"search_events"},
		}}}),
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.FinishReason)
	// Output:
	// stop
}

func ExampleCache() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{
		Tools: []ai.Tool{anthropic.CacheTool(ai.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)})},
		Messages: []ai.Message{
			ai.UserParts(anthropic.CacheFor(ai.Text{Text: "A long shared document."}, anthropic.TTL1Hour), ai.Text{Text: "Summarize it."}),
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from the Messages API.
}

func exampleBatchServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/messages/batches":
			_, _ = io.WriteString(w, `{"id":"msgbatch_1","type":"message_batch","processing_status":"in_progress","request_counts":{"processing":1},"expires_at":"2026-11-18T00:00:00Z","cancel_initiated_at":null,"results_url":null}`)
		case r.Method == http.MethodPost:
			_, _ = io.WriteString(w, `{"id":"msgbatch_1","type":"message_batch","processing_status":"canceling","request_counts":{"processing":1},"expires_at":"2026-11-18T00:00:00Z","cancel_initiated_at":"2026-11-17T00:00:00Z","results_url":null}`)
		case r.URL.Path == "/v1/messages/batches/msgbatch_1/results":
			_, _ = io.WriteString(w, `{"custom_id":"q1","result":{"type":"succeeded","message":{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Paris."}],"stop_reason":"end_turn","usage":{"input_tokens":8,"output_tokens":2}}}}`)
		default:
			_, _ = io.WriteString(w, `{"id":"msgbatch_1","type":"message_batch","processing_status":"ended","request_counts":{"succeeded":1},"expires_at":"2026-11-18T00:00:00Z","cancel_initiated_at":null,"results_url":"https://example.test/results"}`)
		}
	}))
}

func startExampleBatch(ctx context.Context, srv *httptest.Server) (*ai.BatchJob, error) {
	model := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)).Model("claude-haiku-5-5")
	return ai.StartBatch(ctx, model, []ai.BatchItem{{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Capital of France?")}}}})
}

func ExampleLanguageModel_StartBatch() {
	srv := exampleBatchServer()
	defer srv.Close()

	ctx := context.Background()
	job, err := startExampleBatch(ctx, srv)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := job.Wait(ctx, time.Second); err != nil {
		fmt.Println(err)
		return
	}
	for r := range job.Results(ctx) {
		fmt.Println(r.ID, r.Result.Text)
	}
	// Output:
	// q1 Paris.
}

func ExampleLanguageModel_BatchStatus() {
	srv := exampleBatchServer()
	defer srv.Close()

	ctx := context.Background()
	job, err := startExampleBatch(ctx, srv)
	if err != nil {
		fmt.Println(err)
		return
	}
	status, err := job.Status(ctx)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(status.State, status.Total, status.Completed)
	// Output:
	// completed 1 1
}

func ExampleLanguageModel_BatchResults() {
	srv := exampleBatchServer()
	defer srv.Close()

	ctx := context.Background()
	job, err := startExampleBatch(ctx, srv)
	if err != nil {
		fmt.Println(err)
		return
	}
	for r := range job.Results(ctx) {
		fmt.Println(r.ID, r.Result.Text)
	}
	fmt.Println(job.Err())
	// Output:
	// q1 Paris.
	// <nil>
}

func ExampleLanguageModel_CancelBatch() {
	srv := exampleBatchServer()
	defer srv.Close()

	ctx := context.Background()
	job, err := startExampleBatch(ctx, srv)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(job.Cancel(ctx))
	// Output:
	// <nil>
}

func ExampleFileIDOf() {
	srv := exampleServer(`{"id":"file_011CNha8iCJcU1wXNR6q4V8w","type":"file","filename":"report.pdf","mime_type":"application/pdf","size_bytes":4,"created_at":"2026-11-17T00:00:00Z","downloadable":false,"expires_at":null}`)
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL))
	file, err := ai.UploadFile(context.Background(), p, ai.FileUpload{Name: "report.pdf", MediaType: "application/pdf", Data: strings.NewReader("%PDF")})
	if err != nil {
		fmt.Println(err)
		return
	}
	id, ok := anthropic.FileIDOf(file)
	fmt.Println(id, ok, file.MediaType)
	msg := ai.UserParts(file, ai.Text{Text: "Summarize the report."})
	fmt.Println(len(msg.Content))
	// Output:
	// file_011CNha8iCJcU1wXNR6q4V8w true application/pdf
	// 2
}

func fileExampleProvider(response string) (*anthropic.Provider, func()) {
	srv := exampleServer(response)
	return anthropic.New(anthropic.WithAPIKey("sk-ant-test"), anthropic.WithBaseURL(srv.URL)), srv.Close
}

const exampleFile = `{"id":"file_011CNha8iCJcU1wXNR6q4V8w","type":"file","filename":"report.pdf","mime_type":"application/pdf","size_bytes":4,"created_at":"2026-11-17T00:00:00Z","downloadable":false,"expires_at":null}`

func ExampleProvider_UploadFile() {
	p, closeServer := fileExampleProvider(exampleFile)
	defer closeServer()

	file, err := ai.UploadFile(context.Background(), p, ai.FileUpload{Name: "report.pdf", MediaType: "application/pdf", Data: strings.NewReader("%PDF")})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(file.Filename, file.MediaType)
	// Output:
	// report.pdf application/pdf
}

func ExampleProvider_GetFile() {
	p, closeServer := fileExampleProvider(exampleFile)
	defer closeServer()

	file, err := ai.GetFile(context.Background(), p, "file_011CNha8iCJcU1wXNR6q4V8w")
	if err != nil {
		fmt.Println(err)
		return
	}
	id, _ := anthropic.FileIDOf(file)
	fmt.Println(id)
	// Output:
	// file_011CNha8iCJcU1wXNR6q4V8w
}

func ExampleProvider_ListFiles() {
	p, closeServer := fileExampleProvider(`{"data":[` + exampleFile + `],"has_more":false,"next_page":null}`)
	defer closeServer()

	list, err := ai.ListFiles(context.Background(), p, ai.FileListRequest{Limit: 10})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(list.Files), list.Files[0].Filename)
	// Output:
	// 1 report.pdf
}

func ExampleProvider_DeleteFile() {
	p, closeServer := fileExampleProvider(`{"id":"file_011CNha8iCJcU1wXNR6q4V8w","type":"file_deleted"}`)
	defer closeServer()

	fmt.Println(ai.DeleteFile(context.Background(), p, "file_011CNha8iCJcU1wXNR6q4V8w"))
	// Output:
	// <nil>
}

func ExampleCacheFor() {
	text, _ := anthropic.CacheFor(ai.Text{Text: "A long shared document."}, anthropic.TTL1Hour).(ai.Text)
	fmt.Println(string(text.Provider["anthropic"]))
	// Output:
	// {"cacheControl":{"ttl":"1h"}}
}

func ExampleCacheTool() {
	t := anthropic.CacheTool(ai.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)})
	fmt.Println(string(t.Provider["anthropic"]))
	// Output:
	// {"cacheControl":{}}
}

func ExampleCacheToolFor() {
	t := anthropic.CacheToolFor(ai.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}, anthropic.TTL5Minutes)
	fmt.Println(string(t.Provider["anthropic"]))
	// Output:
	// {"cacheControl":{"ttl":"5m"}}
}

func ExampleWithAPIKey() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("X-Api-Key"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleMessage)
	}))
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL))
	if _, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// sk-ant-example
}

func ExampleWithBaseURL() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()

	p := anthropic.New(anthropic.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from the Messages API.
}

type limiterFunc func(context.Context) error

func (f limiterFunc) Wait(ctx context.Context) error { return f(ctx) }

func ExampleWithHTTPClient() {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		fmt.Println(r.URL.Host)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(exampleMessage)), Request: r}, nil
	})}

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL("http://gateway.example"), anthropic.WithHTTPClient(client))
	if _, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// gateway.example
}

func ExampleWithLimiter() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()

	waits := 0
	limiter := limiterFunc(func(context.Context) error { waits++; return nil })
	p := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL), anthropic.WithLimiter(limiter))
	if _, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(waits)
	// Output:
	// 1
}

func ExampleWithHeaders() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("X-Team"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleMessage)
	}))
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL), anthropic.WithHeaders(http.Header{"X-Team": {"search"}}))
	if _, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// search
}

func ExampleWithBeta() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Header.Get("Anthropic-Beta"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, exampleMessage)
	}))
	defer srv.Close()

	p := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL), anthropic.WithBeta("context-management-2025-06-27"))
	if _, err := ai.GenerateText(context.Background(), p.Model("claude-sonnet-5-5"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		fmt.Println(err)
	}
	// Output:
	// context-management-2025-06-27
}

func ExampleProvider_Model() {
	m := anthropic.New().Model("claude-sonnet-5-5")
	fmt.Println(m.Provider(), m.ModelID())
	// Output:
	// anthropic claude-sonnet-5-5
}

func ExampleLanguageModel_Generate() {
	srv := exampleServer(exampleMessage)
	defer srv.Close()

	m := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")
	res, err := m.Generate(context.Background(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	text, _ := res.Content[0].(ai.Text)
	fmt.Println(text.Text, res.FinishReason)
	// Output:
	// Hello from the Messages API. stop
}

func ExampleLanguageModel_Stream() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range []string{
			`{"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}`,
			`{"type":"message_stop"}`,
		} {
			typ, _, _ := strings.Cut(strings.TrimPrefix(e, `{"type":"`), `"`)
			_, _ = io.WriteString(w, "event: "+typ+"\ndata: "+e+"\n\n")
		}
	}))
	defer srv.Close()

	m := anthropic.New(anthropic.WithAPIKey("sk-ant-example"), anthropic.WithBaseURL(srv.URL)).Model("claude-sonnet-5-5")
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
