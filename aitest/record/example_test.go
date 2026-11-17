package record_test

import (
	"context"
	"flag"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/minodevss/ai-go/aitest/record"
)

type exampleTB struct {
	testing.TB
	cleanups []func()
}

func (e *exampleTB) Helper() {}

func (e *exampleTB) Cleanup(f func()) { e.cleanups = append(e.cleanups, f) }

func (e *exampleTB) Fatalf(format string, args ...any) { panic(fmt.Sprintf(format, args...)) }

func (e *exampleTB) Errorf(format string, args ...any) { fmt.Printf(format+"\n", args...) }

func (e *exampleTB) finish() {
	for _, f := range slices.Backward(e.cleanups) {
		f()
	}
}

func exampleDir() (string, func()) {
	dir, err := os.MkdirTemp("", "record")
	if err != nil {
		panic(err)
	}
	return dir, func() { _ = os.RemoveAll(dir) }
}

func exampleSend(rt http.RoundTripper, url string, header http.Header, body string) string {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		panic(err)
	}
	maps.Copy(req.Header, header)
	resp, err := (&http.Client{Transport: rt}).Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}
	return string(out)
}

func exampleRecord(path string, header http.Header, body string, opts ...record.Option) {
	if err := flag.Set("update", "true"); err != nil {
		panic(err)
	}
	defer func() { _ = flag.Set("update", "false") }()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Tenant-Reply", "tenant-reply")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	tb := &exampleTB{}
	rt := record.New(tb, path, append([]record.Option{record.WithUpstream(http.DefaultTransport)}, opts...)...)
	exampleSend(rt, upstream.URL+"/v1/chat", header, body)
	tb.finish()
}

func exampleFixture(path string) {
	err := record.Save(path, record.File{Exchanges: []record.Exchange{{
		Request:  record.Request{Method: http.MethodPost, URL: "https://api.test/v1/chat", Body: []byte(`{"q":1}`)},
		Response: record.Response{Status: http.StatusOK, Body: []byte(`{"answer":"hi"}`)},
	}}})
	if err != nil {
		panic(err)
	}
}

func ExampleNew() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleFixture(path)

	rt := record.New(&exampleTB{}, path)
	fmt.Println(exampleSend(rt, "http://offline.invalid/v1/chat", nil, `{"q":1}`))
	// Output:
	// {"answer":"hi"}
}

func ExampleWithBodyMatch() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleFixture(path)

	rt := record.New(&exampleTB{}, path, record.WithBodyMatch())
	fmt.Println(exampleSend(rt, "http://offline.invalid/v1/chat", nil, `{"q":1}`))
	// Output:
	// {"answer":"hi"}
}

func ExampleWithUpstream() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleRecord(path, nil, `{"q":1}`)

	file, err := record.Load(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s\n", file.Exchanges[0].Response.Bytes())
	// Output:
	// {"ok":true}
}

func ExampleWithBodyRewrite() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleRecord(path, nil, `{"user":"alice"}`, record.WithBodyRewrite(func(body []byte) []byte {
		return []byte(strings.ReplaceAll(string(body), "alice", "user-1"))
	}))

	file, err := record.Load(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s\n", file.Exchanges[0].Request.Bytes())
	// Output:
	// {"user":"user-1"}
}

func ExampleWithRedactedHeaders() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleRecord(path, http.Header{"X-Tenant": {"tenant-1"}}, `{}`, record.WithRedactedHeaders("X-Tenant", "X-Tenant-Reply"))

	file, err := record.Load(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	ex := file.Exchanges[0]
	fmt.Println(ex.Request.Header.Get("X-Tenant"), ex.Response.Header.Get("X-Tenant-Reply"))
	// Output:
	// REDACTED REDACTED
}

func ExampleTransport_Requests() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")
	exampleFixture(path)

	rt := record.New(&exampleTB{}, path)
	exampleSend(rt, "http://offline.invalid/v1/chat", http.Header{"Authorization": {"Bearer secret"}}, `{"q":1}`)
	sent := rt.Requests()[0]
	fmt.Println(sent.Method, sent.URL, sent.Header.Get("Authorization"), string(sent.Bytes()))
	// Output:
	// POST http://offline.invalid/v1/chat REDACTED {"q":1}
}

func ExampleSave() {
	dir, cleanup := exampleDir()
	defer cleanup()
	path := filepath.Join(dir, "fixture.json")

	err := record.Save(path, record.File{Exchanges: []record.Exchange{{
		Request:  record.Request{Method: "POST", URL: "https://api.openai.com/v1/responses", Body: []byte(`{"model":"gpt-5-mini"}`)},
		Response: record.Response{Status: 200, BodyLines: []string{"data: hello", "", ""}},
	}}})
	if err != nil {
		fmt.Println(err)
		return
	}
	file, err := record.Load(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s", file.Exchanges[0].Response.Bytes())
	// Output:
	// data: hello
}

func ExampleLoad() {
	_, err := record.Load("testdata/missing.json")
	fmt.Println(err != nil)
	// Output:
	// true
}

func ExampleUpdating() {
	fmt.Println(record.Updating())
	// Output:
	// false
}

func ExampleRequest_Bytes() {
	r := record.Request{BodyLines: []string{"a", "b"}}
	fmt.Println(strings.ReplaceAll(string(r.Bytes()), "\n", "|"))
	// Output:
	// a|b
}

func ExampleResponse_Bytes() {
	r := record.Response{Body: []byte("{ \"a\": 1 }")}
	fmt.Println(string(r.Bytes()))
	// Output:
	// {"a":1}
}
