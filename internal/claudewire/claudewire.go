// Package claudewire holds the turn and tool-call bookkeeping that the Anthropic Messages and Bedrock Converse wire formats share.
package claudewire

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"slices"
)

const (
	User      = "user"
	Assistant = "assistant"

	maxCallID    = 64
	callIDPrefix = "toolu_"
)

type Message struct {
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

type Turn[R any] struct {
	Role    string
	Blocks  []any
	Results []R
	Calls   []string
}

type Turns[R any] []*Turn[R]

func (ts *Turns[R]) For(role string) *Turn[R] {
	if n := len(*ts); n > 0 && (*ts)[n-1].Role == role {
		return (*ts)[n-1]
	}
	t := &Turn[R]{Role: role}
	*ts = append(*ts, t)
	return t
}

func (ts *Turns[R]) Settle(id func(R) string, missing func(id string) R, dropped func(n int)) {
	ts.dropEmpty()
	if n := len(*ts); n > 0 && (*ts)[n-1].Role == Assistant && len((*ts)[n-1].Calls) > 0 {
		*ts = append(*ts, &Turn[R]{Role: User})
	}
	for i, t := range *ts {
		if t.Role != User {
			continue
		}
		var calls []string
		if i > 0 {
			calls = (*ts)[i-1].Calls
		}
		t.complete(calls, id, missing, dropped)
	}
}

func (ts Turns[R]) Messages() []Message {
	out := make([]Message, 0, len(ts))
	for _, t := range ts {
		content := make([]any, 0, len(t.Results)+len(t.Blocks))
		for _, r := range t.Results {
			content = append(content, r)
		}
		content = append(content, t.Blocks...)
		if len(content) > 0 {
			out = append(out, Message{Role: t.Role, Content: content})
		}
	}
	return out
}

func (ts *Turns[R]) dropEmpty() {
	*ts = slices.DeleteFunc(*ts, func(t *Turn[R]) bool { return len(t.Blocks) == 0 && len(t.Results) == 0 })
	merged := (*ts)[:0]
	for _, t := range *ts {
		if n := len(merged); n > 0 && merged[n-1].Role == t.Role {
			last := merged[n-1]
			last.Blocks = append(last.Blocks, t.Blocks...)
			last.Results = append(last.Results, t.Results...)
			last.Calls = append(last.Calls, t.Calls...)
			continue
		}
		merged = append(merged, t)
	}
	*ts = merged
}

func (t *Turn[R]) complete(calls []string, id func(R) string, missing func(string) R, dropped func(int)) {
	have := map[string]bool{}
	kept := make([]R, 0, len(t.Results))
	for _, r := range t.Results {
		rid := id(r)
		if slices.Contains(calls, rid) && !have[rid] {
			have[rid] = true
			kept = append(kept, r)
		}
	}
	if n := len(t.Results) - len(kept); n > 0 {
		dropped(n)
	}
	var absent []R
	for _, cid := range calls {
		if !have[cid] {
			have[cid] = true
			absent = append(absent, missing(cid))
		}
	}
	t.Results = slices.Concat(absent, kept)
}

func CallID(id string, valid *regexp.Regexp) string {
	if id != "" && len(id) <= maxCallID && valid.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return callIDPrefix + hex.EncodeToString(sum[:])[:maxCallID-len(callIDPrefix)-8]
}

func ObjectInput(in json.RawMessage) json.RawMessage {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(in, &obj); err != nil || obj == nil {
		return json.RawMessage("{}")
	}
	return in
}

func ToolInput(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage("{}")
	}
	return raw
}

func Nullless(raw json.RawMessage) json.RawMessage {
	if string(raw) == "null" {
		return nil
	}
	return raw
}
