package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func TestGenerateText_continuesPausedTurn(t *testing.T) {
	var calls atomic.Int32
	paused := `{"id":"msg_p1","type":"message","role":"assistant","model":"m","content":[
		{"type":"text","text":"Searching."},
		{"type":"server_tool_use","id":"srvtoolu_p","name":"web_search","input":{"query":"q"}}],
		"stop_reason":"pause_turn","stop_sequence":null,
		"usage":{"input_tokens":10,"output_tokens":4},
		"container":{"id":"container_p","expires_at":"2099-01-01T00:00:00Z","skills":null}}`
	done := `{"id":"msg_p2","type":"message","role":"assistant","model":"m","content":[
		{"type":"web_search_tool_result","tool_use_id":"srvtoolu_p","content":[]},
		{"type":"text","text":"Found it."}],
		"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":6}}`
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			jsonReply(paused)(w, r)
			return
		}
		jsonReply(done)(w, r)
	})
	res, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{
		Messages: []ai.Message{ai.User("find")},
		Tools:    []ai.Tool{anthropic.WebSearch(anthropic.WebSearchOptions{}), anthropic.CodeExecution()},
	})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if len(res.Steps) != 2 || res.Steps[0].FinishReason != ai.FinishPause || res.FinishReason != ai.FinishStop || res.Text != "Found it." {
		t.Fatalf("steps = %d, first finish = %q, finish = %q, text = %q; want a paused step then a final stop", len(res.Steps), res.Steps[0].FinishReason, res.FinishReason, res.Text)
	}
	if res.Usage.Input != 30 || res.Usage.Output != 10 {
		t.Errorf("usage = %+v, want the sum of both steps", res.Usage)
	}
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"find"}]},
		{"role":"assistant","content":[
			{"type":"text","text":"Searching."},
			{"type":"server_tool_use","id":"srvtoolu_p","name":"web_search","input":{"query":"q"}}]}]`, body(t, srv)["messages"])
	if got := field(t, srv, "container"); got != `"container_p"` {
		t.Errorf("container of the continuation = %s, want container_p", got)
	}
	if len(res.Messages) != 2 {
		t.Fatalf("messages = %d, want 2 assistant messages", len(res.Messages))
	}
	data, ok := ai.ProviderValue[anthropic.ResponseData](res.Messages[0].Provider, "anthropic")
	if !ok || data.Container == nil || data.Container.ID != "container_p" {
		t.Errorf("first message provider data = %+v, want the container id", data)
	}
}

func TestGenerateText_toolLoopReplaysThinking(t *testing.T) {
	var calls atomic.Int32
	first := `{"id":"msg_t1","type":"message","role":"assistant","model":"m","content":[
		{"type":"thinking","thinking":"plan","signature":"SIG-A"},
		{"type":"tool_use","id":"toolu_w","name":"get_weather","input":{"city":"Seoul"}}],
		"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}`
	second := `{"id":"msg_t2","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"Sunny."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":2}}`
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			jsonReply(first)(w, r)
			return
		}
		jsonReply(second)(w, r)
	})
	weather := ai.NewTool("get_weather", "Weather", func(_ context.Context, in weatherInput) (string, error) { return "Sunny in " + in.City, nil })
	res, err := ai.GenerateText(t.Context(), srv.provider().Model("m"), ai.TextRequest{Messages: []ai.Message{ai.User("weather?")}, Tools: []ai.Tool{weather}})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if res.Text != "Sunny." {
		t.Errorf("text = %q, want Sunny.", res.Text)
	}
	raw, err := json.Marshal(res.Messages)
	if err != nil {
		t.Fatal(err)
	}
	var again []ai.Message
	if err := json.Unmarshal(raw, &again); err != nil {
		t.Fatal(err)
	}
	history := append([]ai.Message{ai.User("weather?")}, again...)
	history = append(history, ai.User("and tomorrow?"))
	generate(t, srv, &ai.Call{Messages: history, Tools: []ai.Tool{weather}})
	equalJSON(t, `[
		{"role":"user","content":[{"type":"text","text":"weather?"}]},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"plan","signature":"SIG-A"},
			{"type":"tool_use","id":"toolu_w","name":"get_weather","input":{"city":"Seoul"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_w","content":"\"Sunny in Seoul\""}]},
		{"role":"assistant","content":[{"type":"text","text":"Sunny."}]},
		{"role":"user","content":[{"type":"text","text":"and tomorrow?"}]}]`, body(t, srv)["messages"])
}

type countingLimiter struct{ n atomic.Int32 }

func (l *countingLimiter) Wait(context.Context) error {
	l.n.Add(1)
	return nil
}

func TestWithLimiter(t *testing.T) {
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") == "text/event-stream" {
			sseReply(sseOf(startEvent, `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`, `{"type":"message_stop"}`))(w, r)
			return
		}
		jsonReply(minimalMessage)(w, r)
	})
	l := &countingLimiter{}
	generate(t, srv, &ai.Call{Messages: []ai.Message{ai.User("hi")}}, anthropic.WithLimiter(l))
	if _, err := streamAll(t, srv.provider(anthropic.WithLimiter(l)).Model("m"), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	if l.n.Load() != 2 {
		t.Errorf("limiter waited %d times, want 2", l.n.Load())
	}
}

func TestProvider_languageModel(t *testing.T) {
	m, err := anthropic.New().LanguageModel("claude-haiku-5-5")
	if err != nil || m.Provider() != "anthropic" || m.ModelID() != "claude-haiku-5-5" {
		t.Errorf("LanguageModel() = %v, %v, want an anthropic claude-haiku-5-5 model", m, err)
	}
}
