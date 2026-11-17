package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func batchServer(t *testing.T, status string) *fakeServer {
	t.Helper()
	var srv *fakeServer
	srv = newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1beta/models/gemini-3-flash:batchGenerateContent":
			jsonReply(batchFixture(t, "create.json"))(w, r)
		case r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files":
			w.Header().Set("X-Goog-Upload-Url", srv.URL+"/upload-session")
		case r.URL.Path == "/upload-session":
			jsonReply(batchFixture(t, "upload.json"))(w, r)
		case r.URL.Path == "/v1beta/batches/123456:cancel":
			jsonReply(`{}`)(w, r)
		case r.URL.Path == "/v1beta/batches/123456":
			jsonReply(batchFixture(t, status))(w, r)
		case r.URL.Path == "/download/v1beta/files/batch-results-1:download":
			rawReply("application/octet-stream", batchFixture(t, "results.jsonl")+"\n")(w, r)
		case r.Method == http.MethodDelete:
			jsonReply(`{}`)(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	return srv
}

func batchItems(ids ...string) []ai.BatchItem {
	items := make([]ai.BatchItem, len(ids))
	for i, id := range ids {
		items[i] = ai.BatchItem{ID: id, Request: ai.TextRequest{
			Instructions:    ai.Texts("Reply briefly."),
			Messages:        []ai.Message{ai.User("Reply with " + id + ".")},
			MaxOutputTokens: 64,
		}}
	}
	return items
}

func TestBatch_startSendsInlineRequestsWithKeys(t *testing.T) {
	srv := batchServer(t, "running.json")
	m := srv.provider().Model("gemini-3-flash")
	job, err := ai.StartBatch(t.Context(), m, batchItems("request-1", "request-2"))
	if err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if job.ID() != "batches/123456" {
		t.Errorf("job.ID() = %q, want batches/123456", job.ID())
	}
	var body struct {
		Batch struct {
			DisplayName string `json:"displayName"`
			InputConfig struct {
				Requests struct {
					Requests []struct {
						Metadata map[string]string `json:"metadata"`
						Request  json.RawMessage   `json:"request"`
					} `json:"requests"`
				} `json:"requests"`
			} `json:"inputConfig"`
		} `json:"batch"`
	}
	if err := json.Unmarshal(srv.last(t).Body, &body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	var keys []string
	for _, r := range body.Batch.InputConfig.Requests.Requests {
		keys = append(keys, r.Metadata["key"])
	}
	if diff := cmp.Diff([]string{"request-1", "request-2"}, keys); diff != "" {
		t.Errorf("request keys mismatch (-want +got):\n%s", diff)
	}
	if body.Batch.DisplayName == "" {
		t.Error("batch has no displayName, which the API requires")
	}
	if got := srv.last(t).Header.Get("x-goog-api-key"); got != "test-key" {
		t.Errorf("x-goog-api-key = %q, want test-key", got)
	}
}

func TestBatch_largeInputGoesThroughTheFilesAPI(t *testing.T) {
	srv := batchServer(t, "running.json")
	m := srv.provider().Model("gemini-3-flash")
	items := batchItems("big")
	items[0].Request.Messages = []ai.Message{ai.User(strings.Repeat("x", 21<<20))}
	if _, err := ai.StartBatch(t.Context(), m, items); err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	var paths []string
	srv.mu.Lock()
	for _, r := range srv.requests {
		paths = append(paths, r.Path)
	}
	last := srv.requests[len(srv.requests)-1]
	upload := srv.requests[1]
	srv.mu.Unlock()
	want := []string{"/upload/v1beta/files", "/upload-session", "/v1beta/models/gemini-3-flash:batchGenerateContent"}
	if diff := cmp.Diff(want, paths); diff != "" {
		t.Errorf("request paths mismatch (-want +got):\n%s", diff)
	}
	if !strings.HasPrefix(string(upload.Body), `{"key":"big","request":`) || !strings.HasSuffix(string(upload.Body), "}\n") {
		t.Errorf("uploaded JSONL = %.60q..., want one keyed request line", upload.Body)
	}
	equalJSON(t, `{"batch":{"displayName":"ai-go-batch","inputConfig":{"fileName":"files/abc-123"}}}`, last.Body)
}

func TestBatch_statusMapsStates(t *testing.T) {
	tests := []struct {
		name  string
		state string
		want  ai.BatchState
	}{
		{name: "pending", state: "BATCH_STATE_PENDING", want: ai.BatchValidating},
		{name: "running", state: "JOB_STATE_RUNNING", want: ai.BatchRunning},
		{name: "succeeded", state: "JOB_STATE_SUCCEEDED", want: ai.BatchCompleted},
		{name: "failed", state: "BATCH_STATE_FAILED", want: ai.BatchFailed},
		{name: "canceled", state: "JOB_STATE_CANCELLED", want: ai.BatchCanceled}, //nolint:misspell // Gemini spells the batch state this way
		{name: "expired", state: "BATCH_STATE_EXPIRED", want: ai.BatchExpired},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(`{"name":"batches/123456","metadata":{"state":"`+tt.state+`"}}`))
			job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
			st, err := job.Status(t.Context())
			if err != nil {
				t.Fatalf("Status() error = %v", err)
			}
			if st.State != tt.want {
				t.Errorf("Status().State = %q, want %q", st.State, tt.want)
			}
		})
	}
}

func TestBatch_statusReportsCountsAndExpiry(t *testing.T) {
	srv := batchServer(t, "running.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	st, err := job.Status(t.Context())
	if err != nil {
		t.Fatalf("Status() error = %v", err)
	}
	want := ai.BatchStatus{State: ai.BatchRunning, Total: 2, Completed: 1, ExpiresAt: time.Date(2026, 11, 19, 10, 0, 0, 0, time.UTC)}
	if diff := cmp.Diff(want, st); diff != "" {
		t.Errorf("Status() mismatch (-want +got):\n%s", diff)
	}
}

func TestBatch_failedOperationCarriesItsMessage(t *testing.T) {
	srv := batchServer(t, "failed.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	err := job.Wait(t.Context(), time.Millisecond)
	var jobErr *ai.JobError
	if !errors.As(err, &jobErr) || !strings.Contains(err.Error(), "internal error") {
		t.Errorf("Wait() error = %v, want a JobError carrying the batch error", err)
	}
}

func TestBatch_resultsFromInlineResponses(t *testing.T) {
	srv := batchServer(t, "succeeded_inline.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	got := map[string]ai.BatchResult{}
	for r := range job.Results(t.Context()) {
		got[r.ID] = r
	}
	if err := job.Err(); err != nil {
		t.Fatalf("Results() error = %v", err)
	}
	if r := got["request-1"]; r.Result == nil || r.Result.Text != "Photosynthesis turns light into sugar." || r.Result.Usage.Input != 8 {
		t.Errorf("result request-1 = %+v, want the response text and usage", r)
	}
	var apiErr *ai.APIError
	if r := got["request-2"]; !errors.As(r.Err, &apiErr) || apiErr.Code != "rpc_3" || apiErr.Message != "Request contains an invalid argument." {
		t.Errorf("result request-2 error = %v, want the item's status", r.Err)
	}
}

func TestBatch_resultsFromResponsesFileInAnyOrder(t *testing.T) {
	srv := batchServer(t, "succeeded_file.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	var ids []string
	got := map[string]ai.BatchResult{}
	for r := range job.Results(t.Context()) {
		ids = append(ids, r.ID)
		got[r.ID] = r
	}
	if err := job.Err(); err != nil {
		t.Fatalf("Results() error = %v", err)
	}
	if diff := cmp.Diff([]string{"request-2", "request-1"}, ids); diff != "" {
		t.Errorf("result order mismatch (-want +got):\n%s", diff)
	}
	if r := got["request-2"]; r.Result == nil || r.Result.Text != "Dough, tomatoes and mozzarella." {
		t.Errorf("result request-2 = %+v, want the response text", r)
	}
	if r := got["request-1"]; r.Err == nil || !strings.Contains(r.Err.Error(), "unavailable") {
		t.Errorf("result request-1 error = %v, want the item's status", r.Err)
	}
	last := srv.last(t)
	if last.Query != "alt=media" {
		t.Errorf("download query = %q, want alt=media", last.Query)
	}
}

func TestBatch_resultsBeforeTheBatchEndedFail(t *testing.T) {
	srv := batchServer(t, "running.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	for range job.Results(t.Context()) {
		t.Error("Results() yielded a result for a running batch")
	}
	if job.Err() == nil {
		t.Error("Err() = nil, want a no-results-yet error")
	}
}

func TestBatch_cancel(t *testing.T) {
	srv := batchServer(t, "running.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/123456")
	if err := job.Cancel(t.Context()); err != nil {
		t.Fatalf("Cancel() error = %v", err)
	}
	if got := srv.last(t); got.Path != "/v1beta/batches/123456:cancel" {
		t.Errorf("cancel path = %q, want /v1beta/batches/123456:cancel", got.Path)
	}
}

func TestBatch_invalidIDIsRejected(t *testing.T) {
	srv := batchServer(t, "running.json")
	job := ai.ResumeBatch(srv.provider().Model("gemini-3-flash"), "batches/../files/x")
	if _, err := job.Status(t.Context()); err == nil {
		t.Error("Status() error = nil, want an invalid id error")
	}
	if srv.count() != 0 {
		t.Errorf("server received %d requests, want 0", srv.count())
	}
}

func TestBatch_vertexIsUnsupported(t *testing.T) {
	p := google.New(google.WithVertex("p", "us-central1"), google.WithTokenSource(func(context.Context) (string, error) { return "token", nil }))
	_, err := ai.StartBatch(t.Context(), p.Model("gemini-3-flash"), batchItems("a"))
	if !errors.Is(err, ai.ErrUnsupported) {
		t.Errorf("StartBatch() error = %v, want ErrUnsupported", err)
	}
}

func TestBatch_waitPollsUntilSucceeded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		polls := 0
		client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			polls++
			if polls < 3 {
				return stubResponse(`{"name":"batches/123456","metadata":{"state":"JOB_STATE_RUNNING"}}`), nil
			}
			return stubResponse(`{"name":"batches/123456","done":true,"metadata":{"state":"JOB_STATE_SUCCEEDED"}}`), nil
		})}
		p := google.New(google.WithAPIKey("k"), google.WithBaseURL("https://example.test/v1beta"), google.WithHTTPClient(client))
		start := time.Now()
		if err := ai.ResumeBatch(p.Model("gemini-3-flash"), "batches/123456").Wait(t.Context(), time.Minute); err != nil {
			t.Fatalf("Wait() error = %v", err)
		}
		if polls != 3 || time.Since(start) != 2*time.Minute {
			t.Errorf("Wait() polled %d times over %s, want 3 polls over 2m0s", polls, time.Since(start))
		}
	})
}
