package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxBatchRequests   = 50000
	maxBatchInputBytes = 200 << 20
	batchWindow        = "24h"
	batchPurpose       = "batch"
	responsesEndpoint  = "/v1/responses"
)

type batchLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

type batchCreate struct {
	InputFileID      string `json:"input_file_id"`
	Endpoint         string `json:"endpoint"`
	CompletionWindow string `json:"completion_window"`
}

type batchBody struct {
	ID            string `json:"id"`
	Status        string `json:"status"`
	OutputFileID  string `json:"output_file_id"`
	ErrorFileID   string `json:"error_file_id"`
	ExpiresAt     int64  `json:"expires_at"`
	RequestCounts struct {
		Total     int `json:"total"`
		Completed int `json:"completed"`
		Failed    int `json:"failed"`
	} `json:"request_counts"`
	Errors *struct {
		Data []struct {
			Message string `json:"message"`
		} `json:"data"`
	} `json:"errors"`
}

type batchResultLine struct {
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		RequestID  string          `json:"request_id"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// StartBatch creates a Batch API job with the 24h completion window. A batch holds at most 50,000 requests and 200 MB of input, all for this model.
func (m *LanguageModel) StartBatch(ctx context.Context, calls []ai.BatchCall) (string, error) {
	input, err := m.batchInput(calls)
	if err != nil {
		return "", err
	}
	files := filesAPI{m.client}
	file, err := files.uploadBytes(ctx, &ai.FileUploadCall{Name: "batch.jsonl", MediaType: "application/jsonl", Purpose: batchPurpose}, input)
	if err != nil {
		return "", err
	}
	fileID, ok := FileIDOf(*file)
	if !ok {
		return "", errors.New("openai: uploaded batch input has no file id")
	}
	var batch batchBody
	_, err = m.client.JSON(ctx, transport.Request{Path: "/batches", Body: batchCreate{InputFileID: fileID, Endpoint: responsesEndpoint, CompletionWindow: batchWindow}}, &batch)
	if err != nil {
		if delErr := files.remove(context.WithoutCancel(ctx), fileID); delErr != nil {
			err = errors.Join(err, fmt.Errorf("openai: remove unused batch input: %w", delErr))
		}
		return "", err
	}
	return batch.ID, nil
}

func (m *LanguageModel) batchInput(calls []ai.BatchCall) ([]byte, error) {
	if len(calls) > maxBatchRequests {
		return nil, fmt.Errorf("openai: a batch holds at most %d requests, got %d", maxBatchRequests, len(calls))
	}
	var buf bytes.Buffer
	for _, bc := range calls {
		req, _, err := m.request(bc.Call, false)
		if err != nil {
			return nil, fmt.Errorf("openai: batch item %q: %w", bc.ID, err)
		}
		body, err := transport.EncodeJSON(req)
		if err != nil {
			return nil, fmt.Errorf("openai: batch item %q: %w", bc.ID, err)
		}
		line, err := transport.EncodeJSON(batchLine{CustomID: bc.ID, Method: http.MethodPost, URL: responsesEndpoint, Body: body})
		if err != nil {
			return nil, fmt.Errorf("openai: batch item %q: %w", bc.ID, err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
		if buf.Len() > maxBatchInputBytes {
			return nil, fmt.Errorf("openai: batch input exceeds %d MB", maxBatchInputBytes>>20)
		}
	}
	return buf.Bytes(), nil
}

func (m *LanguageModel) BatchStatus(ctx context.Context, id string) (*ai.BatchStatus, error) {
	b, err := m.batch(ctx, id)
	if err != nil {
		return nil, err
	}
	st := &ai.BatchStatus{
		State:     batchState(b.Status),
		Total:     b.RequestCounts.Total,
		Completed: b.RequestCounts.Completed,
		Failed:    b.RequestCounts.Failed,
	}
	if b.ExpiresAt > 0 {
		st.ExpiresAt = time.Unix(b.ExpiresAt, 0)
	}
	if b.Errors != nil && len(b.Errors.Data) > 0 {
		st.Message = b.Errors.Data[0].Message
	}
	return st, nil
}

func (m *LanguageModel) batch(ctx context.Context, id string) (batchBody, error) {
	var b batchBody
	if err := checkID(id); err != nil {
		return b, err
	}
	_, err := m.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/batches/" + escapeID(id)}, &b)
	return b, err
}

// BatchResults reads the output file, then the error file. Entries come in provider order and carry the request's id.
func (m *LanguageModel) BatchResults(ctx context.Context, id string) iter.Seq2[ai.BatchEntry, error] {
	return func(yield func(ai.BatchEntry, error) bool) {
		b, err := m.batch(ctx, id)
		if err != nil {
			yield(ai.BatchEntry{}, err)
			return
		}
		if b.OutputFileID == "" && b.ErrorFileID == "" && !batchState(b.Status).Done() {
			yield(ai.BatchEntry{}, fmt.Errorf("openai: batch %s has no results yet (status %s)", id, b.Status))
			return
		}
		for _, fileID := range []string{b.OutputFileID, b.ErrorFileID} {
			if fileID == "" {
				continue
			}
			if !m.resultFile(ctx, fileID, yield) {
				return
			}
		}
	}
}

func (m *LanguageModel) resultFile(ctx context.Context, fileID string, yield func(ai.BatchEntry, error) bool) bool {
	if err := checkID(fileID); err != nil {
		yield(ai.BatchEntry{}, err)
		return false
	}
	resp, err := m.client.Do(ctx, transport.Request{Method: http.MethodGet, Path: "/files/" + escapeID(fileID) + "/content", Header: http.Header{"Accept": {"application/octet-stream"}}})
	if err != nil {
		yield(ai.BatchEntry{}, err)
		return false
	}
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	for {
		var line batchResultLine
		err := dec.Decode(&line)
		if errors.Is(err, io.EOF) {
			return true
		}
		if err != nil {
			yield(ai.BatchEntry{}, fmt.Errorf("openai: decode batch result: %w", err))
			return false
		}
		if !yield(entryOf(line), nil) {
			return false
		}
	}
}

func entryOf(line batchResultLine) ai.BatchEntry {
	entry := ai.BatchEntry{ID: line.CustomID}
	switch {
	case line.Response != nil && line.Response.StatusCode == http.StatusOK:
		var body responseBody
		if err := json.Unmarshal(line.Response.Body, &body); err != nil {
			entry.Err = fmt.Errorf("openai: decode batch response: %w", err)
			return entry
		}
		resp, err := body.response(nil, nil)
		entry.Response, entry.Err = resp, err
	case line.Response != nil:
		code, message := transport.ParseError(line.Response.Body)
		entry.Err = &ai.APIError{Provider: providerName, StatusCode: line.Response.StatusCode, Code: code, Message: message, RequestID: line.Response.RequestID, Body: string(line.Response.Body)}
	case line.Error != nil:
		entry.Err = &ai.APIError{Provider: providerName, Code: line.Error.Code, Message: line.Error.Message}
	default:
		entry.Err = errors.New("openai: batch result line has neither response nor error")
	}
	return entry
}

func (m *LanguageModel) CancelBatch(ctx context.Context, id string) error {
	if err := checkID(id); err != nil {
		return err
	}
	var b batchBody
	_, err := m.client.JSON(ctx, transport.Request{Path: "/batches/" + escapeID(id) + "/cancel"}, &b)
	return err
}

func batchState(status string) ai.BatchState {
	switch strings.ToLower(status) {
	case "validating":
		return ai.BatchValidating
	case "in_progress":
		return ai.BatchRunning
	case "finalizing":
		return ai.BatchFinalizing
	case "cancelling": //nolint:misspell // OpenAI spells the batch status this way
		return ai.BatchCanceling
	case "completed":
		return ai.BatchCompleted
	case "failed":
		return ai.BatchFailed
	case "expired":
		return ai.BatchExpired
	case "cancelled": //nolint:misspell // OpenAI spells the batch status this way
		return ai.BatchCanceled
	default:
		return ai.BatchRunning
	}
}

var _ ai.Batches = (*LanguageModel)(nil)
