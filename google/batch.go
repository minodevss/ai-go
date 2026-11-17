package google

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxInlineBatchBytes = 20 << 20
	maxBatchFileBytes   = 2 << 30
	batchExpiry         = 48 * time.Hour
	batchesPrefix       = "batches/"
	batchDisplayName    = "ai-go-batch"
	jsonlType           = "application/jsonl"
)

var validBatchID = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)

type batchKey struct {
	Key string `json:"key"`
}

type inlineRequest struct {
	Request *generateRequest `json:"request"`
	Meta    batchKey         `json:"metadata"`
}

type fileRequest struct {
	Key     string           `json:"key"`
	Request *generateRequest `json:"request"`
}

type batchInput struct {
	FileName string          `json:"fileName,omitzero"`
	Requests *inlineEnvelope `json:"requests,omitzero"`
}

type inlineEnvelope struct {
	Requests []inlineRequest `json:"requests"`
}

type batchCreate struct {
	Batch struct {
		DisplayName string     `json:"displayName"`
		InputConfig batchInput `json:"inputConfig"`
	} `json:"batch"`
}

type rpcStatus struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type batchOutput struct {
	ResponsesFile string `json:"responsesFile"`
	FileName      string `json:"fileName"`
	Inlined       *struct {
		Responses []inlinedResponse `json:"inlinedResponses"`
	} `json:"inlinedResponses"`
}

type batchStats struct {
	RequestCount string `json:"requestCount"`
	SuccessCount string `json:"successfulRequestCount"`
	FailedCount  string `json:"failedRequestCount"`
	PendingCount string `json:"pendingRequestCount"`
}

type batchResource struct {
	State      string       `json:"state"`
	CreateTime string       `json:"createTime"`
	Stats      *batchStats  `json:"batchStats"`
	Output     *batchOutput `json:"output"`
	Dest       *batchOutput `json:"dest"`
}

type batchOperation struct {
	Name     string         `json:"name"`
	Done     bool           `json:"done"`
	Metadata *batchResource `json:"metadata"`
	Response *batchOutput   `json:"response"`
	Error    *rpcStatus     `json:"error"`
	batchResource
}

type inlinedResponse struct {
	Response *responseBody   `json:"response"`
	Error    json.RawMessage `json:"error"`
	Metadata batchKey        `json:"metadata"`
}

type resultLine struct {
	Key      string          `json:"key"`
	Response *responseBody   `json:"response"`
	Error    json.RawMessage `json:"error"`
	Metadata batchKey        `json:"metadata"`
}

// StartBatch sends requests inline under 20 MB and uploads a JSONL file above that. Vertex AI is not supported.
func (m *LanguageModel) StartBatch(ctx context.Context, calls []ai.BatchCall) (string, error) {
	if m.backend.vertex {
		return "", fmt.Errorf("%w: Gemini batches are not part of Vertex AI", ai.ErrUnsupported)
	}
	requests := make([]fileRequest, len(calls))
	for i, bc := range calls {
		req, _, err := m.request(bc.Call)
		if err != nil {
			return "", fmt.Errorf("google: batch item %q: %w", bc.ID, err)
		}
		requests[i] = fileRequest{Key: bc.ID, Request: req}
	}
	var create batchCreate
	create.Batch.DisplayName = batchDisplayName
	inline := make([]inlineRequest, len(requests))
	size := 0
	for i, r := range requests {
		inline[i] = inlineRequest{Request: r.Request, Meta: batchKey{Key: r.Key}}
		raw, err := transport.EncodeJSON(inline[i])
		if err != nil {
			return "", fmt.Errorf("google: batch item %q: %w", r.Key, err)
		}
		size += len(raw)
	}
	var uploaded *ai.File
	if size < maxInlineBatchBytes {
		create.Batch.InputConfig.Requests = &inlineEnvelope{Requests: inline}
	} else {
		file, err := m.uploadBatchInput(ctx, requests)
		if err != nil {
			return "", err
		}
		uploaded = file
		create.Batch.InputConfig.FileName = fileName(*file)
	}
	var op batchOperation
	_, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":batchGenerateContent", Body: create}, &op)
	if err != nil {
		err = fail(err)
		if uploaded != nil {
			if delErr := (filesAPI{m.backend}).remove(context.WithoutCancel(ctx), fileName(*uploaded)); delErr != nil {
				err = errors.Join(err, fmt.Errorf("google: remove unused batch input: %w", delErr))
			}
		}
		return "", err
	}
	return op.Name, nil
}

func (m *LanguageModel) uploadBatchInput(ctx context.Context, requests []fileRequest) (*ai.File, error) {
	var buf bytes.Buffer
	for _, r := range requests {
		line, err := transport.EncodeJSON(r)
		if err != nil {
			return nil, fmt.Errorf("google: batch item %q: %w", r.Key, err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
		if buf.Len() > maxBatchFileBytes {
			return nil, fmt.Errorf("google: batch input exceeds %d GB", maxBatchFileBytes>>30)
		}
	}
	return filesAPI{m.backend}.upload(ctx, &ai.FileUploadCall{Name: batchDisplayName, MediaType: jsonlType, Data: &buf})
}

func (m *LanguageModel) BatchStatus(ctx context.Context, id string) (*ai.BatchStatus, error) {
	op, err := m.batch(ctx, id)
	if err != nil {
		return nil, err
	}
	view := op.view()
	st := &ai.BatchStatus{State: batchState(view.State, op.Done)}
	if view.Stats != nil {
		st.Total = atoi(view.Stats.RequestCount)
		st.Completed = atoi(view.Stats.SuccessCount)
		st.Failed = atoi(view.Stats.FailedCount)
	}
	if created, err := time.Parse(time.RFC3339Nano, view.CreateTime); err == nil {
		st.ExpiresAt = created.Add(batchExpiry)
	}
	if op.Error != nil {
		st.Message = op.Error.Message
	}
	return st, nil
}

func (o batchOperation) view() batchResource {
	view := o.batchResource
	if o.Metadata != nil {
		view = *o.Metadata
	}
	return view
}

func (o batchOperation) output() *batchOutput {
	switch {
	case o.Response != nil:
		return o.Response
	case o.view().Output != nil:
		return o.view().Output
	default:
		return o.view().Dest
	}
}

func (m *LanguageModel) batch(ctx context.Context, id string) (batchOperation, error) {
	var op batchOperation
	if m.backend.vertex {
		return op, fmt.Errorf("%w: Gemini batches are not part of Vertex AI", ai.ErrUnsupported)
	}
	name, err := batchResourceName(id)
	if err != nil {
		return op, err
	}
	_, err = m.backend.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/" + name}, &op)
	return op, fail(err)
}

// BatchResults reads inline responses or downloads the responses file; a response without a key is identified by its position.
func (m *LanguageModel) BatchResults(ctx context.Context, id string) iter.Seq2[ai.BatchEntry, error] {
	return func(yield func(ai.BatchEntry, error) bool) {
		op, err := m.batch(ctx, id)
		if err != nil {
			yield(ai.BatchEntry{}, err)
			return
		}
		state := batchState(op.view().State, op.Done)
		if !state.Done() {
			yield(ai.BatchEntry{}, fmt.Errorf("google: batch %s has no results yet (state %s)", id, state))
			return
		}
		out := op.output()
		switch {
		case out == nil:
		case out.Inlined != nil:
			for i, r := range out.Inlined.Responses {
				if !yield(m.entryOf(resultLine{Key: cmp.Or(r.Metadata.Key, strconv.Itoa(i)), Response: r.Response, Error: r.Error}), nil) {
					return
				}
			}
		case cmp.Or(out.ResponsesFile, out.FileName) != "":
			m.resultFile(ctx, cmp.Or(out.ResponsesFile, out.FileName), yield)
		}
	}
}

func (m *LanguageModel) resultFile(ctx context.Context, name string, yield func(ai.BatchEntry, error) bool) {
	resource, err := fileResource(name)
	if err != nil {
		yield(ai.BatchEntry{}, err)
		return
	}
	target := filesAPI{m.backend}.root() + "/download" + versionSuffix + "/" + resource + ":download?alt=media"
	resp, err := m.backend.send(ctx, http.MethodGet, target, nil, nil, nil, true)
	if err != nil {
		yield(ai.BatchEntry{}, err)
		return
	}
	defer resp.Body.Close()
	for line, err := range resultLines(resp.Body) {
		if err != nil {
			yield(ai.BatchEntry{}, err)
			return
		}
		var rl resultLine
		if err := json.Unmarshal(line, &rl); err != nil {
			yield(ai.BatchEntry{}, fmt.Errorf("google: decode batch result: %w", err))
			return
		}
		rl.Key = cmp.Or(rl.Key, rl.Metadata.Key)
		if !yield(m.entryOf(rl), nil) {
			return
		}
	}
}

func resultLines(r io.Reader) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		br := bufio.NewReader(r)
		for {
			line, err := br.ReadBytes('\n')
			if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 && !yield(trimmed, nil) {
				return
			}
			if errors.Is(err, io.EOF) {
				return
			}
			if err != nil {
				yield(nil, fmt.Errorf("google: read batch results: %w", err))
				return
			}
		}
	}
}

func (m *LanguageModel) entryOf(rl resultLine) ai.BatchEntry {
	entry := ai.BatchEntry{ID: rl.Key}
	switch {
	case rl.Response != nil:
		resp, err := m.convert(*rl.Response, nil, nil)
		entry.Response, entry.Err = resp, err
	case len(rl.Error) > 0 && string(rl.Error) != "null":
		entry.Err = m.backend.batchError(rl.Error)
	default:
		entry.Err = errors.New("google: batch result has neither response nor error")
	}
	return entry
}

func (b *backend) batchError(raw json.RawMessage) error {
	var status rpcStatus
	if err := json.Unmarshal(raw, &status); err != nil {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			text = string(raw)
		}
		return &ai.APIError{Provider: b.name, Message: transport.Truncate(text)}
	}
	return &ai.APIError{Provider: b.name, Code: statusCode(status.Code), Message: transport.Truncate(status.Message), Body: string(raw)}
}

func statusCode(code int) string {
	return "rpc_" + strconv.Itoa(code)
}

func (m *LanguageModel) CancelBatch(ctx context.Context, id string) error {
	if m.backend.vertex {
		return fmt.Errorf("%w: Gemini batches are not part of Vertex AI", ai.ErrUnsupported)
	}
	name, err := batchResourceName(id)
	if err != nil {
		return err
	}
	var empty json.RawMessage
	_, err = m.backend.client.JSON(ctx, transport.Request{Path: "/" + name + ":cancel"}, &empty)
	return fail(err)
}

func batchResourceName(id string) (string, error) {
	bare := strings.TrimPrefix(id, batchesPrefix)
	if !validBatchID.MatchString(bare) {
		return "", fmt.Errorf("google: invalid batch id %q", id)
	}
	return batchesPrefix + url.PathEscape(bare), nil
}

func batchState(state string, done bool) ai.BatchState {
	name := strings.TrimPrefix(strings.TrimPrefix(state, "JOB_STATE_"), "BATCH_STATE_")
	switch name {
	case "PENDING":
		return ai.BatchValidating
	case "RUNNING":
		return ai.BatchRunning
	case "SUCCEEDED":
		return ai.BatchCompleted
	case "FAILED":
		return ai.BatchFailed
	case "CANCELLED": //nolint:misspell // Gemini spells the batch state this way
		return ai.BatchCanceled
	case "EXPIRED":
		return ai.BatchExpired
	default:
		if done {
			return ai.BatchCompleted
		}
		return ai.BatchRunning
	}
}

func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}

var _ ai.Batches = (*LanguageModel)(nil)
