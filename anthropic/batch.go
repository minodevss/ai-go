package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxBatchRequests   = 100000
	maxBatchInputBytes = 256 << 20
	batchPath          = "/v1/messages/batches"
)

var (
	ErrBatchCanceled = errors.New("anthropic: batch request was canceled before it was sent to the model")
	ErrBatchExpired  = errors.New("anthropic: batch request expired before it was sent to the model")

	batchCustomID = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)
)

type batchRequest struct {
	CustomID string           `json:"custom_id"`
	Params   *messagesRequest `json:"params"`
}

type batchCreate struct {
	Requests []json.RawMessage `json:"requests"`
}

type batchBody struct {
	ID               string `json:"id"`
	ProcessingStatus string `json:"processing_status"`
	RequestCounts    struct {
		Processing int `json:"processing"`
		Succeeded  int `json:"succeeded"`
		Errored    int `json:"errored"`
		Canceled   int `json:"canceled"`
		Expired    int `json:"expired"`
	} `json:"request_counts"`
	ExpiresAt         string  `json:"expires_at"`
	CancelInitiatedAt *string `json:"cancel_initiated_at"`
	ResultsURL        *string `json:"results_url"`
}

type batchResultLine struct {
	CustomID string `json:"custom_id"`
	Result   struct {
		Type    string          `json:"type"`
		Message json.RawMessage `json:"message"`
		Error   struct {
			RequestID string `json:"request_id"`
			Error     struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		} `json:"error"`
	} `json:"result"`
}

// StartBatch creates a Message Batch with one request per call. A batch holds at most 100,000 requests and 256 MB, every ID must match ^[a-zA-Z0-9_-]{1,64}$, and structured output that needs a forced tool call is rejected because results are read without the request. Request warnings are not reported, and headers other than anthropic-beta must be equal across calls.
func (m *LanguageModel) StartBatch(ctx context.Context, calls []ai.BatchCall) (string, error) {
	if len(calls) > maxBatchRequests {
		return "", fmt.Errorf("anthropic: a batch holds at most %d requests, got %d", maxBatchRequests, len(calls))
	}
	requests := make([]json.RawMessage, len(calls))
	header := http.Header{}
	var betas []string
	size := 0
	for i, bc := range calls {
		if !batchCustomID.MatchString(bc.ID) {
			return "", fmt.Errorf("anthropic: batch item ID %q must match %s", bc.ID, batchCustomID)
		}
		p, err := m.prepare(bc.Call, false)
		if err != nil {
			return "", fmt.Errorf("anthropic: batch item %q: %w", bc.ID, err)
		}
		if p.forced != "" {
			return "", fmt.Errorf("anthropic: batch item %q: structured output through a forced tool call is not supported in batches; use a schema with output_config.format", bc.ID)
		}
		other := p.header.Clone()
		other.Del(betaHeader)
		if i == 0 {
			header = other
		} else if !maps.EqualFunc(header, other, slices.Equal) {
			return "", fmt.Errorf("anthropic: batch item %q: headers other than %s differ from the first item", bc.ID, betaHeader)
		}
		betas = append(betas, p.header.Values(betaHeader)...)
		raw, err := transport.EncodeJSON(batchRequest{CustomID: bc.ID, Params: p.req})
		if err != nil {
			return "", fmt.Errorf("anthropic: batch item %q: %w", bc.ID, err)
		}
		if size += len(raw); size > maxBatchInputBytes {
			return "", fmt.Errorf("anthropic: batch input exceeds %d MB", maxBatchInputBytes>>20)
		}
		requests[i] = raw
	}
	if names := splitBetas(betas); len(names) > 0 {
		header.Set(betaHeader, strings.Join(names, ","))
	}
	var body batchBody
	if _, err := m.client.JSON(ctx, transport.Request{Path: batchPath, Body: batchCreate{Requests: requests}, Header: header}, &body); err != nil {
		return "", err
	}
	return body.ID, nil
}

func (m *LanguageModel) BatchStatus(ctx context.Context, id string) (*ai.BatchStatus, error) {
	b, err := m.batch(ctx, id)
	if err != nil {
		return nil, err
	}
	c := b.RequestCounts
	st := &ai.BatchStatus{
		State:     b.state(),
		Total:     c.Processing + c.Succeeded + c.Errored + c.Canceled + c.Expired,
		Completed: c.Succeeded,
		Failed:    c.Errored,
	}
	if at, err := time.Parse(time.RFC3339, b.ExpiresAt); err == nil {
		st.ExpiresAt = at
	}
	return st, nil
}

func (m *LanguageModel) batch(ctx context.Context, id string) (batchBody, error) {
	var b batchBody
	if err := checkID(id); err != nil {
		return b, err
	}
	_, err := m.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: batchPath + "/" + url.PathEscape(id), Header: m.batchHeader()}, &b)
	return b, err
}

func (m *LanguageModel) batchHeader() http.Header {
	if len(m.betas) == 0 {
		return nil
	}
	return http.Header{betaHeader: {strings.Join(m.betas, ",")}}
}

func (b batchBody) state() ai.BatchState {
	switch b.ProcessingStatus {
	case "canceling":
		return ai.BatchCanceling
	case "ended":
		switch {
		case b.CancelInitiatedAt != nil:
			return ai.BatchCanceled
		case b.RequestCounts.Expired > 0:
			return ai.BatchExpired
		default:
			return ai.BatchCompleted
		}
	default:
		return ai.BatchRunning
	}
}

// BatchResults streams the results file in provider order, which can differ from request order. Results exist only after the batch ended and for 29 days after its creation.
func (m *LanguageModel) BatchResults(ctx context.Context, id string) iter.Seq2[ai.BatchEntry, error] {
	return func(yield func(ai.BatchEntry, error) bool) {
		b, err := m.batch(ctx, id)
		if err != nil {
			yield(ai.BatchEntry{}, err)
			return
		}
		switch {
		case b.ProcessingStatus != "ended":
			yield(ai.BatchEntry{}, fmt.Errorf("anthropic: batch %s has no results yet (status %s)", id, b.ProcessingStatus))
			return
		case b.ResultsURL == nil || *b.ResultsURL == "":
			yield(ai.BatchEntry{}, fmt.Errorf("anthropic: batch %s has no results file; results are kept for 29 days after creation", id))
			return
		}
		resp, err := m.client.Do(ctx, transport.Request{Method: http.MethodGet, Path: batchPath + "/" + url.PathEscape(id) + "/results", Header: m.batchHeader()})
		if err != nil {
			yield(ai.BatchEntry{}, err)
			return
		}
		defer resp.Body.Close()
		dec := json.NewDecoder(resp.Body)
		for {
			var line batchResultLine
			err := dec.Decode(&line)
			if errors.Is(err, io.EOF) {
				return
			}
			if err != nil {
				yield(ai.BatchEntry{}, fmt.Errorf("anthropic: decode batch result: %w", err))
				return
			}
			if !yield(entryOf(line), nil) {
				return
			}
		}
	}
}

func entryOf(line batchResultLine) ai.BatchEntry {
	entry := ai.BatchEntry{ID: line.CustomID}
	switch line.Result.Type {
	case "succeeded":
		entry.Response, entry.Err = batchResponse(line.Result.Message)
	case "errored":
		e := line.Result.Error
		entry.Err = &ai.APIError{Provider: providerName, StatusCode: statusForType(e.Error.Type), Code: e.Error.Type, Message: e.Error.Message, RequestID: e.RequestID}
	case "canceled":
		entry.Err = ErrBatchCanceled
	case "expired":
		entry.Err = ErrBatchExpired
	default:
		entry.Err = fmt.Errorf("anthropic: unknown batch result type %q", line.Result.Type)
	}
	return entry
}

func batchResponse(raw json.RawMessage) (*ai.Response, error) {
	var body messageBody
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("anthropic: decode batch message: %w", err)
	}
	conv, err := convertContent(body.Content, "")
	if err != nil {
		return nil, err
	}
	finish, rawFinish := finishOf(body.StopReason)
	if finish == ai.FinishToolCalls && !slices.ContainsFunc(conv, isClientCall) {
		finish = ai.FinishStop
	}
	return &ai.Response{Content: conv, Usage: usageOf(body.Usage), FinishReason: finish, FinishRaw: rawFinish, Provider: body.metadata()}, nil
}

func (m *LanguageModel) CancelBatch(ctx context.Context, id string) error {
	if err := checkID(id); err != nil {
		return err
	}
	var b batchBody
	_, err := m.client.JSON(ctx, transport.Request{Path: batchPath + "/" + url.PathEscape(id) + "/cancel", Header: m.batchHeader()}, &b)
	return err
}

func checkID(id string) error {
	if id == "" || id == "." || id == ".." {
		return fmt.Errorf("anthropic: invalid id %q", id)
	}
	return nil
}

var _ ai.Batches = (*LanguageModel)(nil)
