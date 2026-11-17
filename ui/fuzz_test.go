package ui_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/minodevss/ai-go/ui"
)

func FuzzReadRequest(f *testing.F) {
	for _, seed := range []string{
		exampleBody,
		`{"messages":[{"id":"a","role":"assistant","parts":[{"type":"tool-x","toolCallId":"c","state":"approval-responded","input":{"a":[1,2]},"approval":{"id":"i","approved":true}},{"type":"data-y","data":null},{"type":"file","url":"data:text/plain;base64,aGk=","mediaType":"text/plain"}]}]}`,
		`{"messages":[{"id":"a","role":"user","parts":[{"type":"text","text":"\ud800"}]}]}`,
		`{"messages":[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]}`,
		`{`,
		``,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body string) {
		r := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/chat", strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		req, err := ui.ReadRequest(r, ui.Limits{MaxBodyBytes: 1 << 16})
		if err != nil {
			if _, ok := errors.AsType[*ui.RequestError](err); !ok {
				t.Fatalf("ReadRequest(%q) error = %v, want a *RequestError", body, err)
			}
			return
		}
		if _, err := json.Marshal(req.Messages); err != nil {
			t.Fatalf("Marshal(messages) error = %v for %q", err, body)
		}
		if _, err := ui.ToMessages(req, nil); err != nil {
			if _, ok := errors.AsType[*ui.RequestError](err); !ok {
				t.Fatalf("ToMessages() error = %v, want a *RequestError", err)
			}
		}
	})
}
