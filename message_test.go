package ai_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
)

func richMessage() ai.Message {
	docIndex := 2
	return ai.Message{
		Role:     ai.RoleAssistant,
		Provider: ai.ProviderData{}.WithRaw("anthropic", rawJSON(`{"container":"c1"}`)),
		Content: ai.Parts{
			ai.Text{Text: "hello", Citations: []ai.Citation{{Kind: "page", URL: "https://x.test", Title: "X", CitedText: "q", DocIndex: &docIndex, Start: 1, End: 4}}},
			ai.Reasoning{Text: "hmm", Provider: ai.ProviderData{}.WithRaw("anthropic", rawJSON(`{"signature":"sig=="}`))},
			ai.File{MediaType: "image/png", Filename: "a.png", Data: []byte{1, 2, 3}, Citations: true},
			ai.File{URL: "https://x.test/f.pdf", MediaType: "application/pdf"},
			ai.ToolCall{ID: "c1", Name: "search", Input: json.RawMessage(`{"query":"go"}`), ProviderExecuted: true},
			ai.ToolResult{CallID: "c1", Name: "search", Output: ai.ToolOutput{JSON: json.RawMessage(`{"ok":true}`)}},
			ai.ToolResult{CallID: "c2", Name: "t", Output: ai.ToolOutput{Text: "boom"}, IsError: true},
			ai.ToolResult{CallID: "c3", Name: "t", Output: ai.ToolOutput{Parts: ai.Parts{ai.Text{Text: "inner"}, ai.File{MediaType: "image/png", Data: []byte{9}}}}},
			ai.ToolApprovalRequest{ApprovalID: "a1", CallID: "c1", Reason: "why", InputHash: [32]byte{1, 2, 3, 255}},
			ai.ToolApprovalResponse{ApprovalID: "a1", CallID: "c1", Approved: true, Reason: "ok"},
			ai.Custom{Kind: "web-search-meta", Data: json.RawMessage(`{"n":1}`)},
		},
	}
}

func TestMessage_jsonRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		msg  ai.Message
	}{
		{name: "every part type", msg: richMessage()},
		{name: "user text", msg: ai.User("hi")},
		{name: "empty content", msg: ai.Message{Role: ai.RoleUser}},
		{name: "tool role", msg: ai.Message{Role: ai.RoleTool, Content: ai.Parts{ai.ToolResult{CallID: "c", Name: "n", Output: ai.ToolOutput{Text: "x"}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := mustMarshal(t, tt.msg)
			var got ai.Message
			mustUnmarshal(t, raw, &got)
			if diff := cmp.Diff(tt.msg, got); diff != "" {
				t.Errorf("round trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMessage_typeDiscriminator(t *testing.T) {
	tests := []struct {
		part ai.Part
		want string
	}{
		{part: ai.Text{Text: "a"}, want: "text"},
		{part: ai.Reasoning{Text: "a"}, want: "reasoning"},
		{part: ai.File{}, want: "file"},
		{part: ai.ToolCall{ID: "i", Name: "n"}, want: "tool-call"},
		{part: ai.ToolResult{}, want: "tool-result"},
		{part: ai.ToolApprovalRequest{}, want: "tool-approval-request"},
		{part: ai.ToolApprovalResponse{}, want: "tool-approval-response"},
		{part: ai.Custom{Kind: "k"}, want: "custom"},
	}
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			var got struct {
				Type string `json:"type"`
			}
			mustUnmarshal(t, mustMarshal(t, tt.part), &got)
			if got.Type != tt.want {
				t.Errorf("type = %q, want %q", got.Type, tt.want)
			}
		})
	}
}

func TestParts_pointerNormalization(t *testing.T) {
	call := &ai.ToolCall{ID: "c", Name: "n"}
	var nilText *ai.Text
	tests := []struct {
		name string
		got  ai.Parts
		want ai.Parts
	}{
		{name: "UserParts derefs pointers", got: ai.UserParts(&ai.Text{Text: "a"}, call, ai.Reasoning{Text: "r"}).Content, want: ai.Parts{ai.Text{Text: "a"}, ai.ToolCall{ID: "c", Name: "n"}, ai.Reasoning{Text: "r"}}},
		{name: "UserParts drops nil pointers", got: ai.UserParts(nilText, ai.Text{Text: "b"}).Content, want: ai.Parts{ai.Text{Text: "b"}}},
		{name: "UserParts empty", got: ai.UserParts().Content, want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, tt.got); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParts_pointerMarshalsLikeValue(t *testing.T) {
	byValue := mustMarshal(t, ai.Parts{ai.Text{Text: "a"}, ai.ToolCall{ID: "1", Name: "n"}})
	byPointer := mustMarshal(t, ai.Parts{&ai.Text{Text: "a"}, &ai.ToolCall{ID: "1", Name: "n"}})
	if !bytes.Equal(byValue, byPointer) {
		t.Errorf("pointer parts marshal to %s, values to %s", byPointer, byValue)
	}
}

func TestParts_pointerInHistoryReachesModelAsValue(t *testing.T) {
	msgs := []ai.Message{{Role: ai.RoleUser, Content: ai.Parts{&ai.Text{Text: "ptr"}}}}
	m := newCapture()
	mustGenerate(t, m, ai.TextRequest{Messages: msgs})
	got := m.calls[0].Messages[0].Content[0]
	if _, ok := got.(ai.Text); !ok {
		t.Errorf("provider saw %T, want ai.Text", got)
	}
}

func TestParts_unknownFieldsArePreserved(t *testing.T) {
	in := `{"role":"assistant","content":[` +
		`{"type":"text","text":"hi","futureField":{"a":[1,2]},"provider":{"openai":{"id":"msg_1"}}},` +
		`{"type":"tool-call","id":"c","name":"n","input":{"x":1},"extra":"y"},` +
		`{"type":"hologram","payload":{"k":"v"}}` +
		`]}`
	var msg ai.Message
	mustUnmarshal(t, []byte(in), &msg)

	text, ok := msg.Content[0].(ai.Text)
	if !ok {
		t.Fatalf("Content[0] = %T, want ai.Text", msg.Content[0])
	}
	if id, ok := ai.ProviderValue[map[string]string](text.Provider, "openai"); !ok || id["id"] != "msg_1" {
		t.Errorf("provider openai value lost: %v", text.Provider)
	}
	unknown, ok := ai.ProviderValue[map[string]any](text.Provider, "unknown")
	if !ok || !cmp.Equal(unknown, map[string]any{"futureField": map[string]any{"a": []any{1.0, 2.0}}}) {
		t.Errorf("unknown fields = %v, want futureField preserved", unknown)
	}
	call, ok := msg.Content[1].(ai.ToolCall)
	if !ok {
		t.Fatalf("Content[1] = %T, want ai.ToolCall", msg.Content[1])
	}
	if extra, _ := ai.ProviderValue[map[string]string](call.Provider, "unknown"); extra["extra"] != "y" {
		t.Errorf("tool call unknown fields = %v", call.Provider)
	}
	custom, ok := msg.Content[2].(ai.Custom)
	if !ok || custom.Kind != "hologram" || !bytes.Contains(custom.Data, []byte(`"payload"`)) {
		t.Errorf("Content[2] = %+v, want Custom{Kind: hologram} carrying the payload", msg.Content[2])
	}

	again := mustMarshal(t, msg)
	var second ai.Message
	mustUnmarshal(t, again, &second)
	if diff := cmp.Diff(msg, second); diff != "" {
		t.Errorf("second round trip differs (-first +second):\n%s", diff)
	}
}

func TestParts_decodeErrors(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{name: "content is not an array", in: `{"role":"user","content":"hi"}`},
		{name: "part is not an object", in: `{"role":"user","content":["x"]}`},
		{name: "type is not a string", in: `{"role":"user","content":[{"type":5}]}`},
		{name: "bad field type", in: `{"role":"user","content":[{"type":"text","text":5}]}`},
		{name: "bad input hash", in: `{"role":"user","content":[{"type":"tool-approval-request","inputHash":"zz"}]}`},
		{name: "short input hash", in: `{"role":"user","content":[{"type":"tool-approval-request","inputHash":"abcd"}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg ai.Message
			if err := json.Unmarshal([]byte(tt.in), &msg); err == nil {
				t.Errorf("Unmarshal(%s) error = nil, want an error", tt.in)
			}
		})
	}
}

func TestParts_nullContent(t *testing.T) {
	var msg ai.Message
	mustUnmarshal(t, []byte(`{"role":"user","content":null}`), &msg)
	if msg.Content != nil {
		t.Errorf("Content = %v, want nil", msg.Content)
	}
}

func FuzzMessageJSON(f *testing.F) {
	f.Add(mustMarshalFuzz(richMessage()))
	f.Add([]byte(`{"role":"user","content":[{"type":"text","text":"x","zzz":1}]}`))
	f.Add([]byte(`{"role":"assistant","content":[{"type":"mystery","a":[]}]}`))
	f.Add([]byte(`{"role":"tool","content":[{"type":"tool-approval-request","inputHash":"00"}]}`))
	f.Add([]byte(`{"content":[null]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var first ai.Message
		if json.Unmarshal(data, &first) != nil {
			return
		}
		enc1, err := json.Marshal(first)
		if err != nil {
			t.Fatalf("Marshal(decoded) error = %v", err)
		}
		var second ai.Message
		if err := json.Unmarshal(enc1, &second); err != nil {
			t.Fatalf("Unmarshal(own output %s) error = %v", enc1, err)
		}
		enc2, err := json.Marshal(second)
		if err != nil {
			t.Fatalf("Marshal(second) error = %v", err)
		}
		if !bytes.Equal(enc1, enc2) {
			t.Errorf("encoding is not stable:\n%s\n%s", enc1, enc2)
		}
	})
}

func mustMarshalFuzz(m ai.Message) []byte {
	b, err := json.Marshal(m)
	if err != nil {
		panic(err)
	}
	return b
}

func TestProviderData(t *testing.T) {
	base := ai.ProviderData{}.WithRaw("a", rawJSON(`1`))
	derived := base.WithRaw("b", rawJSON(`"two"`))

	if _, ok := base["b"]; ok {
		t.Error("WithRaw mutated the receiver")
	}
	if got, ok := ai.ProviderValue[int](derived, "a"); !ok || got != 1 {
		t.Errorf("ProviderValue[int](a) = %d, %v; want 1, true", got, ok)
	}
	if got, ok := ai.ProviderValue[string](derived, "b"); !ok || got != "two" {
		t.Errorf("ProviderValue[string](b) = %q, %v; want two, true", got, ok)
	}
	if _, ok := ai.ProviderValue[string](derived, "a"); ok {
		t.Error("ProviderValue[string](a) ok = true for a number")
	}
	if _, ok := ai.ProviderValue[int](derived, "missing"); ok {
		t.Error("ProviderValue(missing) ok = true")
	}
	var nilData ai.ProviderData
	if got := nilData.WithRaw("k", rawJSON(`true`)); len(got) != 1 {
		t.Errorf("WithRaw on nil = %v", got)
	}
}

func TestProviderData_unencodableValueDoesNotPanic(t *testing.T) {
	base := ai.ProviderData{}.WithRaw("a", rawJSON(`1`))
	got, err := base.WithValue("k", make(chan int))
	if err == nil || len(got) != 1 {
		t.Errorf("WithValue(chan) = %v, %v; want unchanged data and an error", got, err)
	}
	ok, err := base.WithValue("k", 2)
	if v, found := ai.ProviderValue[int](ok, "k"); err != nil || !found || v != 2 {
		t.Errorf("WithValue(2) = %v, %v", ok, err)
	}
	raw := base.WithRaw("r", json.RawMessage(`{"x":1}`))
	if v, found := ai.ProviderValue[map[string]int](raw, "r"); !found || v["x"] != 1 {
		t.Errorf("WithRaw = %v", raw)
	}
}

type testOptions struct {
	name string
	Tier string `json:"tier"`
}

func (o testOptions) ProviderName() string { return o.name }

func TestProviders(t *testing.T) {
	got := ai.Providers(testOptions{name: "openai", Tier: "flex"}, testOptions{name: "anthropic", Tier: "x"})
	if v, ok := ai.ProviderValue[testOptions](got, "openai"); !ok || v.Tier != "flex" {
		t.Errorf("openai options = %+v, %v", v, ok)
	}
	if len(got) != 2 {
		t.Errorf("len = %d, want 2", len(got))
	}
	if ai.Providers() != nil {
		t.Error("Providers() with no options should be nil")
	}
}

func TestHelpers(t *testing.T) {
	tests := []struct {
		name string
		got  any
		want any
	}{
		{name: "User", got: ai.User("hi"), want: ai.Message{Role: ai.RoleUser, Content: ai.Parts{ai.Text{Text: "hi"}}}},
		{name: "Assistant", got: ai.Assistant("yo"), want: ai.Message{Role: ai.RoleAssistant, Content: ai.Parts{ai.Text{Text: "yo"}}}},
		{name: "Texts", got: ai.Texts("a", "b"), want: []ai.Text{{Text: "a"}, {Text: "b"}}},
		{name: "Texts none", got: ai.Texts(), want: []ai.Text{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, tt.got); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTextRequest_cloneIsIndependent(t *testing.T) {
	req := ai.TextRequest{
		Instructions:  ai.Texts("a"),
		Messages:      []ai.Message{ai.User("hi")},
		Temperature:   new(0.1),
		Seed:          new(5),
		MaxRetries:    new(1),
		StopSequences: []string{"x"},
		Headers:       map[string][]string{"K": {"v"}},
		Provider:      ai.ProviderData{}.WithRaw("p", rawJSON(`1`)),
	}
	clone := req.Clone()
	clone.Instructions[0].Text = "changed"
	clone.Messages[0].Content[0] = ai.Text{Text: "changed"}
	*clone.Temperature = 0.9
	*clone.Seed = 9
	clone.StopSequences[0] = "changed"
	clone.Headers["K"][0] = "changed"
	clone.Provider["p"] = json.RawMessage("2")

	if req.Instructions[0].Text != "a" || !cmp.Equal(req.Messages[0], ai.User("hi")) || *req.Temperature != 0.1 || *req.Seed != 5 || req.StopSequences[0] != "x" || req.Headers["K"][0] != "v" || string(req.Provider["p"]) != "1" {
		t.Errorf("Clone shares state with the original: %+v", req)
	}
}

func TestTextRequest_withMessages(t *testing.T) {
	base := ai.TextRequest{Messages: []ai.Message{ai.User("old")}, Instructions: ai.Texts("keep")}
	msgs := []ai.Message{ai.User("new")}
	got := base.WithMessages(msgs)
	msgs[0] = ai.User("tampered")

	if !cmp.Equal(got.Messages[0], ai.User("new")) || !cmp.Equal(base.Messages[0], ai.User("old")) {
		t.Errorf("WithMessages shares state: got %+v base %+v", got.Messages, base.Messages)
	}
	if got.Instructions[0].Text != "keep" {
		t.Error("WithMessages dropped other fields")
	}
}
