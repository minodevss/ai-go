package ui_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/ui"
)

func rawMsg(s string) json.RawMessage { return json.RawMessage(s) }

func TestMessage_decodesEveryPartKind(t *testing.T) {
	const wire = `{"id":"m1","role":"assistant","metadata":{"k":1},"parts":[
		{"type":"step-start"},
		{"type":"text","text":"hi","state":"done","providerMetadata":{"openai":{"id":"x"}}},
		{"type":"reasoning","text":"hmm","state":"done"},
		{"type":"file","mediaType":"text/plain","filename":"a.txt","url":"data:text/plain;base64,aGk="},
		{"type":"source-url","sourceId":"s1","url":"https://e.com","title":"E"},
		{"type":"source-document","sourceId":"s2","mediaType":"application/pdf","title":"T","filename":"t.pdf"},
		{"type":"data-weather","id":"w","data":{"t":1}},
		{"type":"custom","kind":"openai.compaction"},
		{"type":"tool-search","toolCallId":"c1","state":"output-available","input":{"q":"x"},"output":{"hits":[1]}},
		{"type":"dynamic-tool","toolName":"mcp","toolCallId":"c2","state":"output-error","input":{},"errorText":"bad"},
		{"type":"tool-delete","toolCallId":"c3","state":"approval-responded","input":{"p":1},"approval":{"id":"a1","approved":false,"reason":"no"}},
		{"type":"reasoning-file","url":"u","mediaType":"image/png"}
	]}`
	var m ui.Message
	if err := json.Unmarshal([]byte(wire), &m); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if m.ID != "m1" || m.Role != ui.RoleAssistant || string(m.Metadata) != `{"k":1}` || len(m.Parts) != 12 {
		t.Fatalf("message = %+v", m)
	}
	if tp, ok := m.Parts[1].(ui.TextPart); !ok || tp.Text != "hi" || tp.State != ui.StateDone || string(tp.Provider["openai"]) != `{"id":"x"}` {
		t.Errorf("text part = %+v", m.Parts[1])
	}
	if tp, ok := m.Parts[8].(ui.ToolPart); !ok || tp.Name != "search" || tp.Dynamic || tp.State != ui.ToolOutputAvailable || string(tp.Output.JSON) != `{"hits":[1]}` {
		t.Errorf("static tool part = %+v", m.Parts[8])
	}
	if tp, ok := m.Parts[9].(ui.ToolPart); !ok || tp.Name != "mcp" || !tp.Dynamic || tp.ErrorText != "bad" {
		t.Errorf("dynamic tool part = %+v", m.Parts[9])
	}
	if tp, ok := m.Parts[10].(ui.ToolPart); !ok || tp.Approval == nil || tp.Approval.Approved == nil || *tp.Approval.Approved || tp.Approval.Reason != "no" {
		t.Errorf("approval tool part = %+v", m.Parts[10])
	}
	if rp, ok := m.Parts[11].(ui.RawPart); !ok || rp.Type != "reasoning-file" {
		t.Errorf("unknown part = %+v", m.Parts[11])
	}
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	var again ui.Message
	if err := json.Unmarshal(out, &again); err != nil {
		t.Fatalf("second Unmarshal() error = %v", err)
	}
	if diff := cmp.Diff(m, again); diff != "" {
		t.Errorf("wire round trip mismatch (-want +got):\n%s", diff)
	}
}

func TestMessage_marshalsToolStatesLikeTheSDK(t *testing.T) {
	tests := []struct {
		name string
		part ui.ToolPart
		want string
	}{
		{
			name: "input available",
			part: ui.ToolPart{Name: "search", CallID: "c1", State: ui.ToolInputAvailable, Input: rawMsg(`{"q":"x"}`)},
			want: `{"type":"tool-search","toolCallId":"c1","state":"input-available","input":{"q":"x"}}`,
		},
		{
			name: "output available text",
			part: ui.ToolPart{Name: "search", CallID: "c1", State: ui.ToolOutputAvailable, Input: rawMsg(`{}`), Output: ai.ToolOutput{Text: "ok"}},
			want: `{"type":"tool-search","toolCallId":"c1","state":"output-available","input":{},"output":"ok"}`,
		},
		{
			name: "output error",
			part: ui.ToolPart{Name: "search", CallID: "c1", State: ui.ToolOutputError, Input: rawMsg(`{}`), ErrorText: "bad"},
			want: `{"type":"tool-search","toolCallId":"c1","state":"output-error","input":{},"errorText":"bad"}`,
		},
		{
			name: "dynamic approval requested",
			part: ui.ToolPart{Name: "mcp", Dynamic: true, CallID: "c2", State: ui.ToolApprovalRequested, Input: rawMsg(`{}`), Approval: &ui.Approval{ID: "a", RequestReason: "r"}},
			want: `{"type":"dynamic-tool","toolName":"mcp","toolCallId":"c2","state":"approval-requested","input":{},"approval":{"id":"a","requestReason":"r"}}`,
		},
		{
			name: "streaming input has no input key",
			part: ui.ToolPart{Name: "search", CallID: "c1", State: ui.ToolInputStreaming, RawInput: `{"q`},
			want: `{"type":"tool-search","toolCallId":"c1","state":"input-streaming","rawInput":"{\"q"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.part)
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("Marshal() = %s, want %s", got, tt.want)
			}
		})
	}
}

func roundTripHistories() map[string][]ai.Message {
	call := ai.ToolCall{ID: "c1", Name: "search", Input: rawMsg(`{"q":"x"}`)}
	return map[string][]ai.Message{
		"plain text": {ai.User("hi"), ai.Assistant("hello")},
		"provider data everywhere": {
			ai.User("hi"),
			{Role: ai.RoleAssistant, Provider: ai.ProviderData{"anthropic": rawMsg(`{"container":"c"}`)}, Content: ai.Parts{
				ai.Reasoning{Text: "think", Provider: ai.ProviderData{"anthropic": rawMsg(`{"signature":"sig"}`)}},
				ai.Text{Text: "answer", Provider: ai.ProviderData{"openai": rawMsg(`{"id":"msg_1"}`)}},
				ai.Text{Provider: ai.ProviderData{"openai": rawMsg(`{"id":"empty"}`)}},
			}},
		},
		"tool call and result": {
			ai.User("search"),
			{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "looking"}, call}},
			{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "c1", Name: "search", Output: ai.ToolOutput{JSON: rawMsg(`{"hits":[1,2]}`)}}}},
			ai.Assistant("found two"),
		},
		"text and error results": {
			ai.User("go"),
			{Role: ai.RoleAssistant, Content: ai.Parts{call, ai.ToolCall{ID: "c2", Name: "search", Input: rawMsg(`{}`)}}},
			{Role: ai.RoleTool, Content: ai.Parts{
				ai.ToolResult{CallID: "c1", Name: "search", Output: ai.ToolOutput{Text: "plain"}},
				ai.ToolResult{CallID: "c2", Name: "search", Output: ai.ToolOutput{Text: "tool failed"}, IsError: true},
			}},
		},
		"approval pending": {
			ai.User("delete"),
			{Role: ai.RoleAssistant, Content: ai.Parts{
				ai.ToolCall{ID: "c1", Name: "delete", Input: rawMsg(`{"p":1}`)},
				ai.ToolApprovalRequest{ApprovalID: "a1", CallID: "c1", Reason: "destructive", InputHash: [32]byte{1, 2, 3}},
			}},
		},
		"approval answered and executed": {
			ai.User("delete"),
			{Role: ai.RoleAssistant, Content: ai.Parts{
				ai.ToolCall{ID: "c1", Name: "delete", Input: rawMsg(`{"p":1}`)},
				ai.ToolApprovalRequest{ApprovalID: "a1", CallID: "c1", InputHash: [32]byte{9}},
			}},
			{Role: ai.RoleTool, Content: ai.Parts{
				ai.ToolApprovalResponse{ApprovalID: "a1", CallID: "c1", Approved: true},
				ai.ToolResult{CallID: "c1", Name: "delete", Output: ai.ToolOutput{JSON: rawMsg(`{"ok":true}`)}},
			}},
			ai.Assistant("deleted"),
		},
		"approval denied": {
			ai.User("delete"),
			{Role: ai.RoleAssistant, Content: ai.Parts{
				ai.ToolCall{ID: "c1", Name: "delete", Input: rawMsg(`{"p":1}`)},
				ai.ToolApprovalRequest{ApprovalID: "a1", CallID: "c1"},
			}},
			{Role: ai.RoleTool, Content: ai.Parts{
				ai.ToolApprovalResponse{ApprovalID: "a1", CallID: "c1", Approved: false, Reason: "no"},
				ai.ToolResult{CallID: "c1", Name: "delete", Output: ai.ToolOutput{Text: "tool call was denied: no"}, IsError: true},
			}},
		},
		"provider executed": {
			ai.User("search the web"),
			{Role: ai.RoleAssistant, Content: ai.Parts{
				ai.ToolCall{ID: "s1", Name: "web_search", Input: rawMsg(`{"q":"x"}`), ProviderExecuted: true, Provider: ai.ProviderData{"anthropic": rawMsg(`{"id":"srv_1"}`)}},
				ai.ToolResult{CallID: "s1", Name: "web_search", Output: ai.ToolOutput{JSON: rawMsg(`[{"url":"u"}]`)}, ProviderExecuted: true, Provider: ai.ProviderData{"anthropic": rawMsg(`{"enc":"e"}`)}},
				ai.Text{Text: "result", Citations: []ai.Citation{{URL: "https://e.com", Title: "E", CitedText: "c", Start: 1, End: 4}}},
			}},
		},
		"files and custom": {
			{Role: ai.RoleUser, Content: ai.Parts{ai.Text{Text: "see"}, ai.File{MediaType: "image/png", Filename: "a.png", Data: []byte{1, 2, 3}}, ai.File{MediaType: "image/png", URL: "https://e.com/a.png"}}},
			{Role: ai.RoleAssistant, Content: ai.Parts{ai.File{MediaType: "image/png", Data: []byte{4}, Citations: true}, ai.Custom{Kind: "openai.compaction", Data: rawMsg(`{"x":1}`), Provider: ai.ProviderData{"openai": rawMsg(`{"enc":"e"}`)}}}},
		},
		"multi step with results between assistant messages": {
			ai.User("two steps"),
			{Role: ai.RoleAssistant, Provider: ai.ProviderData{"openai": rawMsg(`{"resp":"r1"}`)}, Content: ai.Parts{call}},
			{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "c1", Name: "search", Output: ai.ToolOutput{Text: "r"}}}},
			{Role: ai.RoleAssistant, Provider: ai.ProviderData{"openai": rawMsg(`{"resp":"r2"}`)}, Content: ai.Parts{ai.Text{Text: "done"}}},
			ai.User("again"),
			ai.Assistant("ok"),
		},
	}
}

func TestFromMessages_roundTripsThroughToAI(t *testing.T) {
	for name, history := range roundTripHistories() {
		t.Run(name, func(t *testing.T) {
			got, err := ui.ToAI(ui.FromMessages(history))
			if err != nil {
				t.Fatalf("ToAI() error = %v", err)
			}
			if diff := cmp.Diff(history, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("ToAI(FromMessages(h)) mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFromMessages_groupsAssistantAndToolMessagesIntoOneUIMessage(t *testing.T) {
	msgs := roundTripHistories()["multi step with results between assistant messages"]
	got := ui.FromMessages(msgs)
	if len(got) != 4 {
		t.Fatalf("FromMessages() = %d messages, want user, assistant, user, assistant", len(got))
	}
	roles := []ui.Role{got[0].Role, got[1].Role, got[2].Role, got[3].Role}
	if diff := cmp.Diff([]ui.Role{ui.RoleUser, ui.RoleAssistant, ui.RoleUser, ui.RoleAssistant}, roles); diff != "" {
		t.Errorf("roles mismatch (-want +got):\n%s", diff)
	}
	steps := 0
	for _, p := range got[1].Parts {
		if _, ok := p.(ui.StepStartPart); ok {
			steps++
		}
	}
	if steps != 2 {
		t.Errorf("assistant message has %d step-start parts, want 2", steps)
	}
	tools := toolParts(got[1])
	if len(tools) != 1 || tools[0].State != ui.ToolOutputAvailable {
		t.Errorf("tool parts = %+v, want one finished tool part", tools)
	}
}

func TestMessage_messageLevelProviderDataIsNotOnTheWire(t *testing.T) {
	msgs := roundTripHistories()["provider data everywhere"]
	wire, err := json.Marshal(ui.FromMessages(msgs))
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	var back []ui.Message
	if err := json.Unmarshal(wire, &back); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	got, err := ui.ToAI(back)
	if err != nil {
		t.Fatalf("ToAI() error = %v", err)
	}
	if got[1].Provider != nil {
		t.Errorf("message-level provider data = %v, want none after a wire round trip", got[1].Provider)
	}
}

func TestMessage_WithoutProvider(t *testing.T) {
	m := ui.FromMessages(roundTripHistories()["provider data everywhere"])[1]
	wire, err := json.Marshal(m.WithoutProvider())
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	for _, leaked := range []string{"providerMetadata", "signature", "container"} {
		if strings.Contains(string(wire), leaked) {
			t.Errorf("WithoutProvider() wire = %s, still contains %q", wire, leaked)
		}
	}
	if step, ok := m.Parts[0].(ui.StepStartPart); len(m.Parts) == 0 || !ok || step.Provider == nil {
		t.Error("WithoutProvider() modified the receiver")
	}
}
