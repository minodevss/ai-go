// Package providertest is the contract suite every language model adapter runs against recorded fixtures.
package providertest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest/record"
)

// Adapter connects a provider to the suite. NewModel must build a model that sends all HTTP through client.
type Adapter struct {
	NewModel    func(t *testing.T, client *http.Client) ai.LanguageModel
	Wire        Wire
	CallIDValid func(id string) bool
}

// Wire reads the provider's JSON so the suite can compare what the provider sent with what it replays.
type Wire struct {
	Request        func(body []byte) ([]json.RawMessage, error)
	Response       func(body []byte) ([]json.RawMessage, error)
	StreamResponse func(body []byte) ([]json.RawMessage, error)
	CallIDs        func(body []byte) (calls, results []string, err error)
}

// Case names the fixtures Dir/<Name>.generate.json and Dir/<Name>.stream.json. Its request must use client-side tools only.
type Case struct {
	Name       string
	Request    ai.TextRequest
	WantKinds  []string
	WantIDs    []string
	WantUsage  *ai.Usage
	WantFinish ai.FinishReason
}

// SchemaCase uses the fixture Dir/<Name>.generate.json and expects a warning containing each WantWarnings string.
type SchemaCase struct {
	Name         string
	Schema       json.RawMessage
	WantWarnings []string
}

type Config struct {
	Dir    string
	Cases  []Case
	Schema *SchemaCase
	Replay string
}

const (
	foreignSignature = "SIG-FOREIGN-9f3a"
	foreignThought   = "FOREIGN-THOUGHT-7c1d"
	foreignResult    = "FOREIGN-SEARCH-RESULT-52ab"
	approvalMarker   = "APPROVAL-ID-4e7b"
)

func Run(t *testing.T, a Adapter, cfg Config) {
	t.Helper()
	for _, c := range cfg.Cases {
		t.Run(c.Name, func(t *testing.T) { runCase(t, a, cfg, c) })
	}
	if cfg.Schema != nil {
		t.Run("schema warnings", func(t *testing.T) { runSchema(t, a, cfg, *cfg.Schema) })
	}
	if cfg.Replay != "" {
		t.Run("approval parts are never sent", func(t *testing.T) { runApproval(t, a, cfg) })
		t.Run("cross-provider replay", func(t *testing.T) { runForeign(t, a, cfg) })
	}
}

func transportFor(t *testing.T, dir, name string) (*http.Client, *record.Transport) {
	t.Helper()
	rt := record.New(t, filepath.Join(dir, name+".json"))
	return &http.Client{Transport: rt}, rt
}

func runCase(t *testing.T, a Adapter, cfg Config, c Case) {
	t.Helper()
	genClient, _ := transportFor(t, cfg.Dir, c.Name+".generate")
	genModel := a.NewModel(t, genClient)
	gen, err := ai.GenerateText(t.Context(), genModel, withSingleStep(c.Request))
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	streamClient, _ := transportFor(t, cfg.Dir, c.Name+".stream")
	streamModel := a.NewModel(t, streamClient)
	stream := ai.StreamText(t.Context(), streamModel, withSingleStep(c.Request))
	var parts []ai.StreamPart
	for p := range stream.Parts() {
		parts = append(parts, p)
	}
	sres, err := stream.Result()
	if err != nil {
		t.Fatalf("StreamText().Result() error = %v", err)
	}

	t.Run("stream aggregate equals generate", func(t *testing.T) {
		if diff := cmp.Diff(gen.Content, sres.Content, jsonEqual); diff != "" {
			t.Errorf("content mismatch (-generate +stream):\n%s", diff)
		}
		if diff := cmp.Diff(gen.Usage, sres.Usage, jsonEqual); diff != "" {
			t.Errorf("usage mismatch (-generate +stream):\n%s", diff)
		}
		if gen.FinishReason != sres.FinishReason {
			t.Errorf("finish reason: generate = %q, stream = %q", gen.FinishReason, sres.FinishReason)
		}
		if diff := cmp.Diff(gen.Provider, sres.Provider, jsonEqual); diff != "" {
			t.Errorf("response provider data mismatch (-generate +stream):\n%s", diff)
		}
		if diff := cmp.Diff(gen.Messages, sres.Messages, jsonEqual); diff != "" {
			t.Errorf("messages mismatch (-generate +stream):\n%s", diff)
		}
	})
	t.Run("stream is well formed", func(t *testing.T) { checkStream(t, parts) })
	t.Run("part order", func(t *testing.T) {
		if c.WantKinds == nil {
			t.Skip("no WantKinds")
		}
		got := kinds(gen.Content)
		if diff := cmp.Diff(c.WantKinds, got); diff != "" {
			t.Errorf("generate part kinds mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(c.WantKinds, kinds(sres.Content)); diff != "" {
			t.Errorf("stream part kinds mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("provider ids and signatures preserved", func(t *testing.T) {
		raw, err := json.Marshal(gen.Messages)
		if err != nil {
			t.Fatalf("Marshal(messages) error = %v", err)
		}
		for _, id := range c.WantIDs {
			if !strings.Contains(string(raw), id) {
				t.Errorf("messages do not contain %q", id)
			}
		}
	})
	t.Run("usage arithmetic", func(t *testing.T) {
		checkUsage(t, gen.Usage)
		checkUsage(t, sres.Usage)
		if c.WantUsage != nil {
			want := *c.WantUsage
			got := gen.Usage
			got.Raw, want.Raw = nil, nil
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("usage mismatch (-want +got):\n%s", diff)
			}
		}
	})
	t.Run("finish reason", func(t *testing.T) {
		if c.WantFinish == "" {
			t.Skip("no WantFinish")
		}
		if gen.FinishReason != c.WantFinish {
			t.Errorf("finish reason = %q, want %q", gen.FinishReason, c.WantFinish)
		}
	})
	t.Run("wire identity", func(t *testing.T) {
		checkWire(t, a, cfg.Dir, c, c.Name+".generate", gen.Messages, a.Wire.Response)
		checkWire(t, a, cfg.Dir, c, c.Name+".stream", sres.Messages, a.Wire.StreamResponse)
	})
}

func withSingleStep(r ai.TextRequest) ai.TextRequest {
	r = r.Clone()
	r.MaxSteps = 1
	r.MaxRetries = new(0)
	return r
}

var jsonEqual = cmp.Transformer("json", func(raw json.RawMessage) any {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw)
	}
	return v
})

func kinds(parts ai.Parts) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = kindOf(p)
	}
	return out
}

func kindOf(p ai.Part) string {
	switch p.(type) {
	case ai.Text:
		return "text"
	case ai.Reasoning:
		return "reasoning"
	case ai.ToolCall:
		return "tool-call"
	case ai.ToolResult:
		return "tool-result"
	case ai.File:
		return "file"
	case ai.Custom:
		return "custom"
	default:
		return fmt.Sprintf("%T", p)
	}
}

func checkUsage(t *testing.T, u ai.Usage) {
	t.Helper()
	if u.Input < u.CacheRead+u.CacheWrite {
		t.Errorf("usage input %d is below cache read %d + cache write %d", u.Input, u.CacheRead, u.CacheWrite)
	}
	if u.Output < u.Reasoning {
		t.Errorf("usage output %d is below reasoning %d", u.Output, u.Reasoning)
	}
	if u.Input < 0 || u.Output < 0 || u.CacheRead < 0 || u.CacheWrite < 0 || u.Reasoning < 0 {
		t.Errorf("usage has a negative field: %+v", u)
	}
}

func checkStream(t *testing.T, parts []ai.StreamPart) {
	t.Helper()
	open := map[string]bool{}
	var last ai.StreamPart
	for i, p := range parts {
		last = p
		switch v := p.(type) {
		case ai.TextStart:
			open["text:"+v.ID] = true
		case ai.TextDelta:
			if !open["text:"+v.ID] {
				t.Errorf("part %d: text delta %q outside a text block", i, v.ID)
			}
		case ai.TextEnd:
			if !open["text:"+v.ID] {
				t.Errorf("part %d: text end %q without start", i, v.ID)
			}
			delete(open, "text:"+v.ID)
		case ai.ReasoningStart:
			open["reasoning:"+v.ID] = true
		case ai.ReasoningDelta:
			if !open["reasoning:"+v.ID] {
				t.Errorf("part %d: reasoning delta %q outside a reasoning block", i, v.ID)
			}
		case ai.ReasoningEnd:
			if !open["reasoning:"+v.ID] {
				t.Errorf("part %d: reasoning end %q without start", i, v.ID)
			}
			delete(open, "reasoning:"+v.ID)
		case ai.ToolInputStart:
			open["tool:"+v.ID] = true
		case ai.ToolInputDelta:
			if !open["tool:"+v.ID] {
				t.Errorf("part %d: tool input delta %q outside a tool input", i, v.ID)
			}
		case ai.ToolInputEnd:
			if !open["tool:"+v.ID] {
				t.Errorf("part %d: tool input end %q without start", i, v.ID)
			}
			delete(open, "tool:"+v.ID)
		default:
		}
	}
	for id := range open {
		t.Errorf("block %s was never closed", id)
	}
	if _, ok := last.(ai.StreamFinish); !ok {
		t.Errorf("last stream part = %T, want StreamFinish", last)
	}
}

func checkWire(t *testing.T, a Adapter, dir string, c Case, fixture string, msgs []ai.Message, responseItems func([]byte) ([]json.RawMessage, error)) {
	t.Helper()
	file, err := record.Load(filepath.Join(dir, fixture+".json"))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(file.Exchanges) == 0 {
		t.Fatalf("fixture %s has no exchanges", fixture)
	}
	want, err := responseItems(file.Exchanges[0].Response.Bytes())
	if err != nil {
		t.Fatalf("read response items: %v", err)
	}
	for name, history := range map[string][]ai.Message{"in memory": msgs, "after JSON round trip": roundTrip(t, msgs)} {
		history := slices.Concat(c.Request.Messages, history, []ai.Message{ai.User("continue")})
		got := sentItems(t, a, dir, fixture, c.Request, history)
		wire := compactAll(t, want)
		sent := compactAll(t, got)
		if name == "in memory" {
			checkSubsequence(t, name, wire, sent, func(a, b string) bool { return a == b })
			continue
		}
		checkSubsequence(t, name, wire, sent, jsonStringsEqual)
	}
}

func roundTrip(t *testing.T, msgs []ai.Message) []ai.Message {
	t.Helper()
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("Marshal(messages) error = %v", err)
	}
	var out []ai.Message
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("Unmarshal(messages) error = %v", err)
	}
	return out
}

func sentItems(t *testing.T, a Adapter, dir, fixture string, req ai.TextRequest, history []ai.Message) []json.RawMessage {
	t.Helper()
	client, rt := transportFor(t, dir, fixture)
	model := a.NewModel(t, client)
	call := &ai.Call{Instructions: req.Instructions, Messages: history, Tools: req.Tools, Provider: req.Provider}
	if strings.HasSuffix(fixture, ".stream") {
		resp, err := model.Stream(t.Context(), call)
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		for _, err := range resp.Parts {
			if err != nil {
				t.Fatalf("stream error = %v", err)
			}
		}
	} else if _, err := model.Generate(t.Context(), call); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	reqs := rt.Requests()
	if len(reqs) == 0 {
		t.Fatal("no request was sent")
	}
	items, err := a.Wire.Request(reqs[0].Bytes())
	if err != nil {
		t.Fatalf("read request items: %v", err)
	}
	return items
}

func compactAll(t *testing.T, items []json.RawMessage) []string {
	t.Helper()
	out := make([]string, len(items))
	for i, it := range items {
		var b bytes.Buffer
		if err := json.Compact(&b, it); err != nil {
			t.Fatalf("Compact(item %d) error = %v", i, err)
		}
		out[i] = b.String()
	}
	return out
}

func jsonStringsEqual(a, b string) bool {
	var x, y any
	if json.Unmarshal([]byte(a), &x) != nil || json.Unmarshal([]byte(b), &y) != nil {
		return false
	}
	return cmp.Equal(x, y)
}

func checkSubsequence(t *testing.T, mode string, want, got []string, equal func(a, b string) bool) {
	t.Helper()
	pos := 0
	for i, w := range want {
		found := false
		for ; pos < len(got); pos++ {
			if equal(w, got[pos]) {
				found = true
				pos++
				break
			}
		}
		if !found {
			t.Errorf("%s: response item %d was not replayed in order:\n%s", mode, i, w)
			return
		}
	}
}

func runSchema(t *testing.T, a Adapter, cfg Config, sc SchemaCase) {
	t.Helper()
	client, _ := transportFor(t, cfg.Dir, sc.Name+".generate")
	model := a.NewModel(t, client)
	resp, err := model.Generate(t.Context(), &ai.Call{
		Messages: []ai.Message{ai.User("hi")},
		Output:   &ai.OutputSpec{Name: "out", Schema: sc.Schema},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	var all []string
	for _, w := range resp.Warnings {
		all = append(all, w.Message)
	}
	joined := strings.Join(all, "\n")
	for _, want := range sc.WantWarnings {
		if !strings.Contains(joined, want) {
			t.Errorf("warnings do not mention %q:\n%s", want, joined)
		}
	}
}

func replayCall(history []ai.Message) *ai.Call {
	return &ai.Call{Messages: history}
}

func runApproval(t *testing.T, a Adapter, cfg Config) {
	t.Helper()
	history := []ai.Message{
		ai.User("delete it"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Text{Text: "Need permission"},
			ai.ToolApprovalRequest{ApprovalID: approvalMarker, CallID: "call_1"},
		}},
		{Role: ai.RoleUser, Content: ai.Parts{ai.ToolApprovalResponse{ApprovalID: approvalMarker, CallID: "call_1", Approved: true}}},
		ai.User("go on"),
	}
	body := sendHistory(t, a, cfg, history)
	if strings.Contains(string(body), approvalMarker) || strings.Contains(strings.ToLower(string(body)), "approval") {
		t.Errorf("request body contains approval data:\n%s", body)
	}
}

func runForeign(t *testing.T, a Adapter, cfg Config) {
	t.Helper()
	longID := strings.Repeat("x", 90) + " bad/id"
	history := []ai.Message{
		ai.User("question"),
		{Role: ai.RoleAssistant, Content: ai.Parts{
			ai.Reasoning{Text: foreignThought, Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"signature":"` + foreignSignature + `"}`)}},
			ai.Text{Text: "answer", Provider: ai.ProviderData{"anthropic": json.RawMessage(`{"x":1}`)}},
			ai.ToolCall{ID: longID, Name: "lookup", Input: json.RawMessage(`{"q":1}`)},
			ai.ToolCall{ID: "toolu_orphan", Name: "lookup2", Input: json.RawMessage(`{}`)},
			ai.ToolCall{ID: "srvtoolu_1", Name: "web_search", Input: json.RawMessage(`{"q":"x"}`), ProviderExecuted: true},
			ai.ToolResult{CallID: "srvtoolu_1", Name: "web_search", Output: ai.ToolOutput{Text: foreignResult}, ProviderExecuted: true},
		}},
		{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: longID, Name: "lookup", Output: ai.ToolOutput{Text: "ok"}}}},
		ai.User("next"),
	}
	body := sendHistory(t, a, cfg, history)
	text := string(body)
	for _, forbidden := range []string{foreignSignature, foreignThought, "srvtoolu_1"} {
		if strings.Contains(text, forbidden) {
			t.Errorf("request body contains foreign data %q", forbidden)
		}
	}
	if !strings.Contains(text, foreignResult) {
		t.Errorf("request body does not carry the foreign provider-executed result as text")
	}
	calls, results, err := a.Wire.CallIDs(body)
	if err != nil {
		t.Fatalf("read call ids: %v", err)
	}
	if len(calls) != 2 {
		t.Errorf("sent %d tool calls, want 2: %v", len(calls), calls)
	}
	for _, id := range slices.Concat(calls, results) {
		if a.CallIDValid != nil && !a.CallIDValid(id) {
			t.Errorf("call id %q is not valid for the provider", id)
		}
	}
	slices.Sort(calls)
	slices.Sort(results)
	if diff := cmp.Diff(calls, results); diff != "" {
		t.Errorf("every call needs exactly one result (-calls +results):\n%s", diff)
	}
}

func sendHistory(t *testing.T, a Adapter, cfg Config, history []ai.Message) []byte {
	t.Helper()
	client, rt := transportFor(t, cfg.Dir, cfg.Replay)
	model := a.NewModel(t, client)
	if _, err := model.Generate(t.Context(), replayCall(history)); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	reqs := rt.Requests()
	if len(reqs) == 0 {
		t.Fatal("no request was sent")
	}
	return reqs[0].Bytes()
}
