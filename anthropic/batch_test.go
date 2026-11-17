package anthropic_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/anthropic"
)

func batchServer(t *testing.T, statusFixture string) *fakeServer {
	t.Helper()
	return newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/messages/batches":
			jsonReply(fixture(t, "batch/created.json"))(w, r)
		case strings.HasSuffix(r.URL.Path, "/cancel"):
			jsonReply(strings.Replace(fixture(t, "batch/created.json"), `"in_progress"`, `"canceling"`, 1))(w, r)
		case strings.HasSuffix(r.URL.Path, "/results"):
			w.Header().Set("Content-Type", "application/binary")
			_, _ = io.WriteString(w, fixture(t, "batch/results.jsonl"))
		default:
			jsonReply(fixture(t, statusFixture))(w, r)
		}
	})
}

func batchItems(ids ...string) []ai.BatchItem {
	items := make([]ai.BatchItem, len(ids))
	for i, id := range ids {
		items[i] = ai.BatchItem{ID: id, Request: ai.TextRequest{
			Instructions:    []ai.Text{anthropic.CachedTextFor("Shared context.", anthropic.TTL1Hour)},
			Messages:        []ai.Message{ai.User("Reply with " + id)},
			MaxOutputTokens: 64,
		}}
	}
	return items
}

func TestBatch_startSendsInlineRequestsWithCustomIDs(t *testing.T) {
	srv := batchServer(t, "batch/created.json")
	m := srv.provider().Model("claude-opus-5-5")
	job, err := ai.StartBatch(t.Context(), m, batchItems("my-first-request", "my-second-request"))
	if err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if job.ID() != "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d" {
		t.Errorf("ID() = %q, want the batch id", job.ID())
	}
	last := srv.last(t)
	if last.Path != "/v1/messages/batches" {
		t.Errorf("path = %s, want /v1/messages/batches", last.Path)
	}
	equalJSON(t, `{"requests":[
		{"custom_id":"my-first-request","params":{"model":"claude-opus-5-5","max_tokens":64,
			"system":[{"type":"text","text":"Shared context.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
			"messages":[{"role":"user","content":[{"type":"text","text":"Reply with my-first-request"}]}]}},
		{"custom_id":"my-second-request","params":{"model":"claude-opus-5-5","max_tokens":64,
			"system":[{"type":"text","text":"Shared context.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
			"messages":[{"role":"user","content":[{"type":"text","text":"Reply with my-second-request"}]}]}}]}`, last.Body)
}

func TestBatch_resultsCoverEveryResultTypeInProviderOrder(t *testing.T) {
	srv := batchServer(t, "batch/ended.json")
	job := ai.ResumeBatch(srv.provider().Model("claude-opus-5-5"), "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")
	got := map[string]ai.BatchResult{}
	var order []string
	for r := range job.Results(t.Context()) {
		got[r.ID] = r
		order = append(order, r.ID)
	}
	if err := job.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if diff := cmp.Diff([]string{"my-second-request", "my-first-request", "third-request", "fourth-request"}, order); diff != "" {
		t.Errorf("order mismatch (-want +got):\n%s", diff)
	}
	ok := got["my-first-request"]
	if ok.Err != nil || ok.Result == nil || ok.Result.Text != "Hello! How can I help you today?" || ok.Result.Usage.Input != 11 || ok.Result.Usage.Output != 9 {
		t.Errorf("succeeded result = %+v, want text and usage", ok)
	}
	apiErr, isAPI := errors.AsType[*ai.APIError](got["my-second-request"].Err)
	if !isAPI || apiErr.Code != "invalid_request_error" || apiErr.StatusCode != http.StatusBadRequest || apiErr.RequestID != "req_011CQ" {
		t.Errorf("errored result error = %v, want a 400 invalid_request_error", got["my-second-request"].Err)
	}
	if !errors.Is(got["third-request"].Err, anthropic.ErrBatchCanceled) {
		t.Errorf("canceled result error = %v, want ErrBatchCanceled", got["third-request"].Err)
	}
	if !errors.Is(got["fourth-request"].Err, anthropic.ErrBatchExpired) {
		t.Errorf("expired result error = %v, want ErrBatchExpired", got["fourth-request"].Err)
	}
}

func TestBatch_statusMapsStatesAndCounts(t *testing.T) {
	tests := []struct {
		name string
		body string
		want ai.BatchState
	}{
		{name: "in progress", body: fixture(t, "batch/created.json"), want: ai.BatchRunning},
		{name: "canceling", body: strings.Replace(fixture(t, "batch/created.json"), `"in_progress"`, `"canceling"`, 1), want: ai.BatchCanceling},
		{name: "ended with expired requests", body: fixture(t, "batch/ended.json"), want: ai.BatchExpired},
		{name: "ended clean", body: strings.Replace(fixture(t, "batch/ended.json"), `"expired": 1`, `"expired": 0`, 1), want: ai.BatchCompleted},
		{name: "ended after cancel", body: strings.Replace(fixture(t, "batch/ended.json"), `"cancel_initiated_at": null`, `"cancel_initiated_at": "2024-09-24T18:40:00Z"`, 1), want: ai.BatchCanceled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.body))
			job := ai.ResumeBatch(srv.provider().Model("claude-opus-5-5"), "msgbatch_1")
			st, err := job.Status(t.Context())
			if err != nil {
				t.Fatalf("Status() error = %v", err)
			}
			if st.State != tt.want {
				t.Errorf("Status().State = %q, want %q", st.State, tt.want)
			}
			if st.ExpiresAt.IsZero() {
				t.Error("Status().ExpiresAt is zero, want the 24h expiry")
			}
		})
	}
	srv := newFakeServer(t, jsonReply(fixture(t, "batch/ended.json")))
	st, err := ai.ResumeBatch(srv.provider().Model("claude-opus-5-5"), "msgbatch_1").Status(t.Context())
	if err != nil || st.Total != 4 || st.Completed != 1 || st.Failed != 1 {
		t.Errorf("Status() = %+v, %v, want total 4, completed 1, failed 1", st, err)
	}
}

func TestBatch_resultsBeforeTheBatchEndedFail(t *testing.T) {
	srv := batchServer(t, "batch/created.json")
	job := ai.ResumeBatch(srv.provider().Model("claude-opus-5-5"), "msgbatch_1")
	for range job.Results(t.Context()) {
		t.Fatal("Results() yielded a result before the batch ended")
	}
	if err := job.Err(); err == nil || !strings.Contains(err.Error(), "no results yet") {
		t.Errorf("Err() = %v, want a no-results error", err)
	}
}

func TestBatch_cancel(t *testing.T) {
	srv := batchServer(t, "batch/created.json")
	job := ai.ResumeBatch(srv.provider().Model("claude-opus-5-5"), "msgbatch_1")
	if err := job.Cancel(t.Context()); err != nil {
		t.Fatalf("Cancel() error = %v", err)
	}
	if last := srv.last(t); last.Path != "/v1/messages/batches/msgbatch_1/cancel" {
		t.Errorf("path = %s, want the cancel endpoint", last.Path)
	}
}

func TestBatch_rejectsInvalidInput(t *testing.T) {
	tooMany := make([]ai.BatchCall, 100001)
	for i := range tooMany {
		tooMany[i] = ai.BatchCall{ID: "a", Call: &ai.Call{Messages: []ai.Message{ai.User("x")}}}
	}
	tests := []struct {
		name  string
		calls []ai.BatchCall
		want  string
	}{
		{name: "id with a space", calls: []ai.BatchCall{{ID: "bad id", Call: &ai.Call{Messages: []ai.Message{ai.User("x")}}}}, want: "must match"},
		{name: "id of 65 characters", calls: []ai.BatchCall{{ID: strings.Repeat("a", 65), Call: &ai.Call{Messages: []ai.Message{ai.User("x")}}}}, want: "must match"},
		{name: "more than 100000 requests", calls: tooMany, want: "at most 100000"},
		{name: "forced tool output", calls: []ai.BatchCall{{ID: "a", Call: &ai.Call{Messages: []ai.Message{ai.User("x")}, Output: &ai.OutputSpec{Name: "o"}}}}, want: "forced tool call"},
		{name: "too many cache breakpoints", calls: []ai.BatchCall{{ID: "a", Call: &ai.Call{Messages: []ai.Message{ai.UserParts(manyCached(5)...)}}}}, want: "cache breakpoints"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(fixture(t, "batch/created.json")))
			_, err := srv.provider().Model("claude-opus-5-5").StartBatch(t.Context(), tt.calls)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("StartBatch() error = %v, want %q", err, tt.want)
			}
			if srv.count() != 0 {
				t.Errorf("StartBatch() sent %d requests, want none", srv.count())
			}
		})
	}
}

func TestBatch_betasAreUnionedAndOtherHeadersMustMatch(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "batch/created.json")))
	m := srv.provider(anthropic.WithBeta("a-beta")).Model("claude-opus-5-5")
	calls := []ai.BatchCall{
		{ID: "a", Call: &ai.Call{Messages: []ai.Message{ai.User("x")}, Provider: ai.Providers(anthropic.Options{Betas: []string{"b-beta"}})}},
		{ID: "b", Call: &ai.Call{Messages: []ai.Message{ai.User("y")}, Provider: ai.Providers(anthropic.Options{Betas: []string{"c-beta"}})}},
	}
	if _, err := m.StartBatch(t.Context(), calls); err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if got := srv.last(t).Header.Get("Anthropic-Beta"); got != "a-beta,b-beta,c-beta" {
		t.Errorf("anthropic-beta = %q, want a-beta,b-beta,c-beta", got)
	}
	calls[1].Call.Headers = http.Header{"X-Trace": {"1"}}
	if _, err := m.StartBatch(t.Context(), calls); err == nil || !strings.Contains(err.Error(), "headers other than") {
		t.Errorf("StartBatch() error = %v, want a header mismatch", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestBatch_waitPollsUntilTheBatchEnds(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var polls atomic.Int32
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			body := fixture(t, "batch/created.json")
			if polls.Add(1) >= 3 {
				body = fixture(t, "batch/ended.json")
			}
			return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(body))}, nil
		})}
		p := anthropic.New(anthropic.WithAPIKey("k"), anthropic.WithBaseURL("http://anthropic.test"), anthropic.WithHTTPClient(client))
		job := ai.ResumeBatch(p.Model("claude-opus-5-5"), "msgbatch_1")
		start := time.Now()
		if err := job.Wait(t.Context(), time.Minute); err != nil {
			t.Fatalf("Wait() error = %v", err)
		}
		if got := time.Since(start); got != 2*time.Minute {
			t.Errorf("Wait() took %s, want 2m0s", got)
		}
		if polls.Load() != 3 {
			t.Errorf("polls = %d, want 3", polls.Load())
		}
	})
}

func manyCached(n int) []ai.Part {
	parts := make([]ai.Part, n)
	for i := range parts {
		parts[i] = anthropic.Cache(ai.Text{Text: "chunk " + string(rune('a'+i))})
	}
	return parts
}
