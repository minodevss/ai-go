package openai_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func batchServer(t *testing.T, status string) *fakeServer {
	t.Helper()
	return newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/files":
			jsonReply(fixture(t, "file_object.json"))(w, r)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/batches":
			jsonReply(`{"id":"batch_abc123","object":"batch","endpoint":"/v1/responses","status":"validating","input_file_id":"file-abc123","completion_window":"24h","created_at":1714508499}`)(w, r)
		case r.URL.Path == "/v1/batches/batch_abc123/cancel":
			jsonReply(`{"id":"batch_abc123","object":"batch","status":"cancelling"}`)(w, r) //nolint:misspell // OpenAI spells the batch status this way
		case r.URL.Path == "/v1/batches/batch_abc123":
			jsonReply(strings.Replace(fixture(t, "batch_completed.json"), `"status":"completed"`, `"status":"`+status+`"`, 1))(w, r)
		case r.URL.Path == "/v1/files/file-out/content":
			rawReply("application/octet-stream", fixture(t, "batch_output.jsonl"))(w, r)
		case r.URL.Path == "/v1/files/file-err/content":
			rawReply("application/octet-stream", fixture(t, "batch_error.jsonl"))(w, r)
		case r.Method == http.MethodDelete:
			jsonReply(`{"id":"file-abc123","object":"file","deleted":true}`)(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
}

func batchRequests(ids ...string) []ai.BatchItem {
	items := make([]ai.BatchItem, len(ids))
	for i, id := range ids {
		items[i] = ai.BatchItem{ID: id, Request: ai.TextRequest{
			Instructions:    ai.Texts("Reply briefly."),
			Messages:        []ai.Message{ai.User("Reply with the word " + id + ".")},
			MaxOutputTokens: 64,
		}}
	}
	return items
}

func TestBatch_startUploadsJSONLAndCreatesTheBatch(t *testing.T) {
	srv := batchServer(t, "completed")
	m := srv.provider().Model("gpt-5-mini")
	job, err := ai.StartBatch(t.Context(), m, batchRequests("request-1", "request-2"))
	if err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if job.ID() != "batch_abc123" {
		t.Errorf("ID() = %q, want batch_abc123", job.ID())
	}
	srv.mu.Lock()
	defer srv.mu.Unlock()
	if len(srv.requests) != 2 || srv.requests[0].Path != "/v1/files" || srv.requests[1].Path != "/v1/batches" {
		t.Fatalf("requests = %+v, want an upload then a batch creation", srv.requests)
	}
	parts := parseForm(t, srv.requests[0])
	if fields := formFields(parts); fields["purpose"] != "batch" {
		t.Errorf("upload fields = %v, want purpose=batch", fields)
	}
	files := formFiles(parts, "file")
	if len(files) != 1 {
		t.Fatalf("file parts = %+v, want one", files)
	}
	lines := strings.Split(strings.TrimSuffix(string(files[0].Data), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("jsonl has %d lines, want 2:\n%s", len(lines), files[0].Data)
	}
	for i, id := range []string{"request-1", "request-2"} {
		var line struct {
			CustomID string          `json:"custom_id"`
			Method   string          `json:"method"`
			URL      string          `json:"url"`
			Body     json.RawMessage `json:"body"`
		}
		if err := json.Unmarshal([]byte(lines[i]), &line); err != nil {
			t.Fatalf("line %d is not JSON: %v", i, err)
		}
		if line.CustomID != id || line.Method != http.MethodPost || line.URL != "/v1/responses" {
			t.Errorf("line %d = %+v, want custom_id %s, POST /v1/responses", i, line, id)
		}
		equalJSON(t, `{"model":"gpt-5-mini","instructions":"Reply briefly.","input":[{"role":"user","content":[{"type":"input_text","text":"Reply with the word `+id+`."}]}],"max_output_tokens":64}`, line.Body)
	}
	equalJSON(t, `{"input_file_id":"file-abc123","endpoint":"/v1/responses","completion_window":"24h"}`, srv.requests[1].Body)
}

func TestBatch_resultsCombineOutputAndErrorFiles(t *testing.T) {
	srv := batchServer(t, "completed")
	m := srv.provider().Model("gpt-5-mini")
	job := ai.ResumeBatch(m, "batch_abc123")
	results := map[string]ai.BatchResult{}
	for r := range job.Results(t.Context()) {
		results[r.ID] = r
	}
	if err := job.Err(); err != nil {
		t.Fatalf("Err() = %v", err)
	}
	if len(results) != 4 {
		t.Fatalf("results = %+v, want 4 (two successes, one 400, one expired)", results)
	}
	for id, want := range map[string]string{"request-1": "blue", "request-2": "green"} {
		r := results[id]
		if r.Err != nil || r.Result == nil || r.Result.Text != want || r.Result.Usage.Input != 12 || r.Result.FinishReason != ai.FinishStop {
			t.Errorf("result %s = %+v, want text %q", id, r, want)
		}
	}
	apiErr, ok := errors.AsType[*ai.APIError](results["request-4"].Err)
	if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Code != "invalid_value" || apiErr.RequestID != "req_4" {
		t.Errorf("request-4 error = %v, want a 400 invalid_value APIError with the request id", results["request-4"].Err)
	}
	expired, ok := errors.AsType[*ai.APIError](results["request-3"].Err)
	if !ok || expired.Code != "batch_expired" {
		t.Errorf("request-3 error = %v, want batch_expired", results["request-3"].Err)
	}
}

func TestBatch_statusMapsEveryState(t *testing.T) {
	tests := []struct {
		status string
		want   ai.BatchState
	}{
		{"validating", ai.BatchValidating},
		{"in_progress", ai.BatchRunning},
		{"finalizing", ai.BatchFinalizing},
		{"cancelling", ai.BatchCanceling}, //nolint:misspell // OpenAI spells the batch status this way
		{"completed", ai.BatchCompleted},
		{"failed", ai.BatchFailed},
		{"expired", ai.BatchExpired},
		{"cancelled", ai.BatchCanceled}, //nolint:misspell // OpenAI spells the batch status this way
	}
	for _, tt := range tests {
		t.Run(tt.status, func(t *testing.T) {
			srv := batchServer(t, tt.status)
			st, err := ai.ResumeBatch(srv.provider().Model("m"), "batch_abc123").Status(t.Context())
			if err != nil {
				t.Fatalf("Status() error = %v", err)
			}
			want := ai.BatchStatus{State: tt.want, Total: 4, Completed: 2, Failed: 2, ExpiresAt: time.Unix(1714536634, 0)}
			if diff := cmp.Diff(want, st); diff != "" {
				t.Errorf("Status() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBatch_failedBatchSurfacesTheValidationMessage(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"batch_x","object":"batch","status":"failed","errors":{"object":"list","data":[{"code":"invalid_json_line","message":"This line is not parseable as valid JSON.","line":2}]},"request_counts":{"total":0,"completed":0,"failed":0}}`))
	job := ai.ResumeBatch(srv.provider().Model("m"), "batch_x")
	err := job.Wait(t.Context(), time.Millisecond)
	jobErr, ok := errors.AsType[*ai.JobError](err)
	if !ok || !strings.Contains(err.Error(), "not parseable") || jobErr.ID != "batch_x" {
		t.Errorf("Wait() error = %v, want the validation message in a JobError", err)
	}
}

func TestBatch_resultsBeforeCompletionFail(t *testing.T) {
	srv := newFakeServer(t, jsonReply(`{"id":"batch_x","object":"batch","status":"in_progress","output_file_id":null,"error_file_id":null}`))
	job := ai.ResumeBatch(srv.provider().Model("m"), "batch_x")
	for range job.Results(t.Context()) {
		t.Error("Results() yielded before the batch finished")
	}
	if err := job.Err(); err == nil || !strings.Contains(err.Error(), "no results yet") {
		t.Errorf("Err() = %v, want a not-ready error", err)
	}
}

func TestBatch_cancel(t *testing.T) {
	srv := batchServer(t, "cancelling") //nolint:misspell // OpenAI spells the batch status this way
	if err := ai.ResumeBatch(srv.provider().Model("m"), "batch_abc123").Cancel(t.Context()); err != nil {
		t.Fatalf("Cancel() error = %v", err)
	}
	if got := srv.last(t); got.Path != "/v1/batches/batch_abc123/cancel" {
		t.Errorf("path = %s, want the cancel endpoint", got.Path)
	}
}

func TestBatch_removesTheInputFileWhenCreationFails(t *testing.T) {
	var deleted []string
	srv := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/files" && r.Method == http.MethodPost:
			jsonReply(fixture(t, "file_object.json"))(w, r)
		case r.Method == http.MethodDelete:
			deleted = append(deleted, r.URL.Path)
			jsonReply(`{"id":"file-abc123","object":"file","deleted":true}`)(w, r)
		default:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"model not available for batch","code":"model_not_found"}}`))
		}
	})
	_, err := ai.StartBatch(t.Context(), srv.provider().Model("m"), batchRequests("a"))
	if apiErr, ok := errors.AsType[*ai.APIError](err); !ok || apiErr.Code != "model_not_found" {
		t.Errorf("StartBatch() error = %v, want the creation error", err)
	}
	if diff := cmp.Diff([]string{"/v1/files/file-abc123"}, deleted); diff != "" {
		t.Errorf("deleted files mismatch (-want +got):\n%s", diff)
	}
}

func TestBatch_limits(t *testing.T) {
	srv := batchServer(t, "completed")
	m := srv.provider().Model("m")
	_, err := m.StartBatch(t.Context(), make([]ai.BatchCall, 50001))
	if err == nil || !strings.Contains(err.Error(), "50000") || srv.count() != 0 {
		t.Errorf("StartBatch() error = %v with %d requests, want a local 50000 limit", err, srv.count())
	}
	big := &ai.Call{Messages: []ai.Message{ai.User(strings.Repeat("x", 32<<20))}}
	calls := make([]ai.BatchCall, 7)
	for i := range calls {
		calls[i] = ai.BatchCall{ID: "c" + itoa(i), Call: big}
	}
	_, err = m.StartBatch(t.Context(), calls)
	if err == nil || !strings.Contains(err.Error(), "200 MB") || srv.count() != 0 {
		t.Errorf("StartBatch() error = %v with %d requests, want a local 200 MB limit", err, srv.count())
	}
}

func TestBatch_rejectedForFallbackAndWrappedModels(t *testing.T) {
	srv := batchServer(t, "completed")
	m := srv.provider().Model("m")
	for name, model := range map[string]ai.LanguageModel{
		"fallback": ai.Fallback(m, m),
		"wrapped":  ai.WrapModel(m, ai.Middleware{}),
	} {
		if _, err := ai.StartBatch(t.Context(), model, batchRequests("a")); !errors.Is(err, ai.ErrUnsupported) {
			t.Errorf("%s: StartBatch() error = %v, want ErrUnsupported", name, err)
		}
	}
	if srv.count() != 0 {
		t.Errorf("%d requests were sent for rejected models", srv.count())
	}
}

func TestBatch_limiterAppliesToEveryCall(t *testing.T) {
	srv := batchServer(t, "completed")
	lim := &countingLimiter{}
	m := srv.provider(openai.WithLimiter(lim)).Model("m")
	job, err := ai.StartBatch(t.Context(), m, batchRequests("a"))
	if err != nil {
		t.Fatalf("StartBatch() error = %v", err)
	}
	if _, err := job.Status(t.Context()); err != nil {
		t.Fatalf("Status() error = %v", err)
	}
	if lim.n != 3 {
		t.Errorf("limiter waited %d times, want 3 (upload, create, status)", lim.n)
	}
}
