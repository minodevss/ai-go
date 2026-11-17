package sse_test

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/internal/sse"
)

func readAll(t *testing.T, input string) []sse.Event {
	t.Helper()
	r := sse.NewReader(strings.NewReader(input))
	var events []sse.Event
	for {
		ev, err := r.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("Next() error = %v", err)
		}
		events = append(events, ev)
	}
}

func TestReader(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []sse.Event
	}{
		{name: "single", input: "data: hello\n\n", want: []sse.Event{{Data: "hello"}}},
		{name: "named", input: "event: response.created\ndata: {}\n\n", want: []sse.Event{{Name: "response.created", Data: "{}"}}},
		{name: "multiline data", input: "data: a\ndata: b\n\n", want: []sse.Event{{Data: "a\nb"}}},
		{name: "crlf", input: "data: a\r\n\r\ndata: b\r\n\r\n", want: []sse.Event{{Data: "a"}, {Data: "b"}}},
		{name: "lone cr", input: "data: a\r\rdata: b\r\r", want: []sse.Event{{Data: "a"}, {Data: "b"}}},
		{name: "comment", input: ": ping\ndata: x\n\n", want: []sse.Event{{Data: "x"}}},
		{name: "no space after colon", input: "data:x\n\n", want: []sse.Event{{Data: "x"}}},
		{name: "empty data dispatches", input: "data:\n\n", want: []sse.Event{{Data: ""}}},
		{name: "name without data is dropped", input: "event: x\n\ndata: y\n\n", want: []sse.Event{{Data: "y"}}},
		{name: "unterminated event is dropped", input: "data: a\n\ndata: b", want: []sse.Event{{Data: "a"}}},
		{name: "bom", input: "\xef\xbb\xbfdata: a\n\n", want: []sse.Event{{Data: "a"}}},
		{name: "id sticks", input: "id: 7\ndata: a\n\ndata: b\n\n", want: []sse.Event{{Data: "a", ID: "7"}, {Data: "b", ID: "7"}}},
		{name: "done sentinel", input: "data: [DONE]\n\n", want: []sse.Event{{Data: "[DONE]"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := readAll(t, tt.input)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReader_tooLarge(t *testing.T) {
	r := sse.NewReader(strings.NewReader("data: " + strings.Repeat("a", 33<<20) + "\n\n"))
	if _, err := r.Next(); !errors.Is(err, sse.ErrTooLarge) {
		t.Errorf("Next() error = %v, want %v", err, sse.ErrTooLarge)
	}
}

type endless struct {
	line []byte
	left int
}

func (e *endless) Read(p []byte) (int, error) {
	if e.left <= 0 {
		return 0, io.EOF
	}
	n := copy(p, bytes.Repeat(e.line, 1+len(p)/len(e.line)))
	n = min(n, e.left)
	e.left -= n
	return n, nil
}

func TestReader_tooLargeAcrossLines(t *testing.T) {
	line := []byte("data: " + strings.Repeat("a", 1000) + "\n")
	r := sse.NewReader(&endless{line: line, left: 600 << 20})
	if _, err := r.Next(); !errors.Is(err, sse.ErrTooLarge) {
		t.Errorf("Next() error = %v, want %v", err, sse.ErrTooLarge)
	}
}

func TestReader_tooLargeComments(t *testing.T) {
	r := sse.NewReader(&endless{line: []byte(": " + strings.Repeat("c", 1000) + "\n"), left: 600 << 20})
	if _, err := r.Next(); !errors.Is(err, sse.ErrTooLarge) {
		t.Errorf("Next() error = %v, want %v", err, sse.ErrTooLarge)
	}
}

func TestReader_manyEventsAreNotCapped(t *testing.T) {
	event := "data: " + strings.Repeat("a", 1<<20) + "\n\n"
	r := sse.NewReader(strings.NewReader(strings.Repeat(event, 40)))
	for i := range 40 {
		if _, err := r.Next(); err != nil {
			t.Fatalf("Next() event %d error = %v", i, err)
		}
	}
}

func encode(ev sse.Event) string {
	var b strings.Builder
	if ev.Name != "" {
		b.WriteString("event: " + ev.Name + "\n")
	}
	for line := range strings.SplitSeq(ev.Data, "\n") {
		b.WriteString("data: " + line + "\n")
	}
	b.WriteString("\n")
	return b.String()
}

func FuzzReader(f *testing.F) {
	for _, seed := range []string{
		"data: a\n\n",
		"event: x\ndata: 1\ndata: 2\n\n",
		": c\r\ndata:\r\n\r\n",
		"data: a\r\rdata: b\r\r",
		"\xef\xbb\xbfid: 1\ndata: z\n\n",
		"event\ndata\n\n",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		r := sse.NewReader(strings.NewReader(input))
		var events []sse.Event
		for {
			ev, err := r.Next()
			if err != nil {
				break
			}
			if strings.ContainsAny(ev.Data, "\r") || strings.ContainsAny(ev.Name, "\r\n") {
				t.Fatalf("event contains line break: %#v", ev)
			}
			events = append(events, ev)
		}
		var b strings.Builder
		for _, ev := range events {
			b.WriteString(encode(ev))
		}
		again := sse.NewReader(strings.NewReader(b.String()))
		for i, want := range events {
			got, err := again.Next()
			if err != nil {
				t.Fatalf("re-read event %d: %v", i, err)
			}
			if got.Name != want.Name || got.Data != want.Data {
				t.Fatalf("re-read event %d = %#v, want %#v", i, got, want)
			}
		}
	})
}
