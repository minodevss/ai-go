package claudewire_test

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/internal/claudewire"
)

type result struct{ ID string }

func settle(ts *claudewire.Turns[result]) (dropped int) {
	ts.Settle(func(r result) string { return r.ID }, func(id string) result { return result{ID: "missing-" + id} }, func(n int) { dropped += n })
	return dropped
}

func TestTurns_forMergesSameRole(t *testing.T) {
	var ts claudewire.Turns[result]
	ts.For(claudewire.User).Blocks = append(ts.For(claudewire.User).Blocks, "a")
	ts.For(claudewire.User).Blocks = append(ts.For(claudewire.User).Blocks, "b")
	ts.For(claudewire.Assistant)
	if len(ts) != 2 || len(ts[0].Blocks) != 2 {
		t.Errorf("turns = %+v, want 2 turns with the first holding 2 blocks", ts)
	}
}

func TestTurns_settle(t *testing.T) {
	tests := []struct {
		name        string
		turns       claudewire.Turns[result]
		want        []claudewire.Message
		wantDropped int
	}{
		{
			name:  "drops empty turns and merges neighbors",
			turns: claudewire.Turns[result]{{Role: claudewire.User, Blocks: []any{"a"}}, {Role: claudewire.Assistant}, {Role: claudewire.User, Blocks: []any{"b"}}},
			want:  []claudewire.Message{{Role: claudewire.User, Content: []any{"a", "b"}}},
		},
		{
			name: "answers an unanswered call in a closing user turn",
			turns: claudewire.Turns[result]{
				{Role: claudewire.User, Blocks: []any{"q"}},
				{Role: claudewire.Assistant, Blocks: []any{"call"}, Calls: []string{"c1"}},
			},
			want: []claudewire.Message{
				{Role: claudewire.User, Content: []any{"q"}},
				{Role: claudewire.Assistant, Content: []any{"call"}},
				{Role: claudewire.User, Content: []any{result{ID: "missing-c1"}}},
			},
		},
		{
			name: "drops results without a call and duplicates",
			turns: claudewire.Turns[result]{
				{Role: claudewire.Assistant, Blocks: []any{"call"}, Calls: []string{"c1"}},
				{Role: claudewire.User, Results: []result{{ID: "c1"}, {ID: "c1"}, {ID: "x"}}},
			},
			want: []claudewire.Message{
				{Role: claudewire.Assistant, Content: []any{"call"}},
				{Role: claudewire.User, Content: []any{result{ID: "c1"}}},
			},
			wantDropped: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDropped := settle(&tt.turns)
			if diff := cmp.Diff(tt.want, tt.turns.Messages()); diff != "" {
				t.Errorf("Messages() mismatch (-want +got):\n%s", diff)
			}
			if gotDropped != tt.wantDropped {
				t.Errorf("dropped = %d, want %d", gotDropped, tt.wantDropped)
			}
		})
	}
}

func TestCallID(t *testing.T) {
	strict := regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
	loose := regexp.MustCompile(`^[A-Za-z0-9_.:-]+$`)
	tests := []struct {
		name  string
		id    string
		valid *regexp.Regexp
		keep  bool
	}{
		{name: "valid kept", id: "call_1", valid: strict, keep: true},
		{name: "dot rejected by strict", id: "a.b", valid: strict},
		{name: "dot kept by loose", id: "a.b", valid: loose, keep: true},
		{name: "empty hashed", id: "", valid: strict},
		{name: "too long hashed", id: strings.Repeat("a", 65), valid: strict},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := claudewire.CallID(tt.id, tt.valid)
			if tt.keep != (got == tt.id) {
				t.Errorf("CallID(%q) = %q, keep = %v", tt.id, got, tt.keep)
			}
			if !strict.MatchString(got) && !tt.keep || len(got) > 64 {
				t.Errorf("CallID(%q) = %q is not a valid id", tt.id, got)
			}
			if again := claudewire.CallID(tt.id, tt.valid); again != got {
				t.Errorf("CallID(%q) is not deterministic: %q then %q", tt.id, got, again)
			}
		})
	}
}

func TestInputs(t *testing.T) {
	tests := []struct {
		name string
		fn   func(json.RawMessage) json.RawMessage
		in   string
		want string
	}{
		{name: "object input keeps object", fn: claudewire.ObjectInput, in: `{"a":1}`, want: `{"a":1}`},
		{name: "object input replaces array", fn: claudewire.ObjectInput, in: `[1]`, want: `{}`},
		{name: "object input replaces null", fn: claudewire.ObjectInput, in: `null`, want: `{}`},
		{name: "object input replaces empty", fn: claudewire.ObjectInput, in: ``, want: `{}`},
		{name: "tool input keeps value", fn: claudewire.ToolInput, in: `[1]`, want: `[1]`},
		{name: "tool input replaces null", fn: claudewire.ToolInput, in: `null`, want: `{}`},
		{name: "tool input replaces empty", fn: claudewire.ToolInput, in: ``, want: `{}`},
		{name: "nullless drops null", fn: claudewire.Nullless, in: `null`, want: ``},
		{name: "nullless keeps value", fn: claudewire.Nullless, in: `{"a":1}`, want: `{"a":1}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(tt.fn(json.RawMessage(tt.in))); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}
