package ai

import (
	"context"
	"io"
	"iter"
	"net/http"
	"time"
)

// SpeechStreamModel is the capability of speech models that synthesize while text is still arriving.
type SpeechStreamModel interface {
	Provider() string
	ModelID() string
	StreamSpeech(ctx context.Context, c *SpeechStreamCall) (*SpeechStreamResponse, error)
}

type SpeechStreamCall struct {
	Text         iter.Seq[string]
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	Headers      http.Header
	Provider     ProviderData
}

// SpeechStreamResponse.Audio ends with the first error it yields; MediaType describes every chunk.
type SpeechStreamResponse struct {
	Audio     iter.Seq2[[]byte, error]
	MediaType string
	Header    http.Header
}

// TranscriptionStreamModel is the capability of transcription models that return text while audio is still being read.
type TranscriptionStreamModel interface {
	Provider() string
	ModelID() string
	StreamTranscribe(ctx context.Context, c *TranscribeCall) (*TranscribeStreamResponse, error)
}

type TranscribeStreamResponse struct {
	Segments iter.Seq2[Segment, error]
	Warnings []Warning
	Header   http.Header
}

type BatchState string

const (
	BatchValidating BatchState = "validating"
	BatchRunning    BatchState = "running"
	BatchFinalizing BatchState = "finalizing"
	BatchCanceling  BatchState = "canceling"
	BatchCompleted  BatchState = "completed"
	BatchFailed     BatchState = "failed"
	BatchExpired    BatchState = "expired"
	BatchCanceled   BatchState = "canceled"
)

// Done reports whether the batch can no longer change state.
func (s BatchState) Done() bool {
	switch s {
	case BatchCompleted, BatchFailed, BatchExpired, BatchCanceled:
		return true
	default:
		return false
	}
}

type BatchStatus struct {
	State     BatchState
	Total     int
	Completed int
	Failed    int
	ExpiresAt time.Time
	Message   string
}

// BatchCall is one request of a batch. ID is unique within the batch and is how results are matched.
type BatchCall struct {
	ID   string
	Call *Call
}

// BatchEntry is one result line of a batch; exactly one of Response and Err is set.
type BatchEntry struct {
	ID       string
	Response *Response
	Err      error
}

// Batches is the capability of language models whose provider runs requests asynchronously at a discount. A provider implements it on a concrete model, never on wrappers.
type Batches interface {
	StartBatch(ctx context.Context, calls []BatchCall) (string, error)
	BatchStatus(ctx context.Context, id string) (*BatchStatus, error)
	BatchResults(ctx context.Context, id string) iter.Seq2[BatchEntry, error]
	CancelBatch(ctx context.Context, id string) error
}

type FileUploadCall struct {
	Name         string
	MediaType    string
	Data         io.Reader
	Purpose      string
	ExpiresAfter time.Duration
	Headers      http.Header
	Provider     ProviderData
}

type FileListCall struct {
	Limit    int
	After    string
	Purpose  string
	Headers  http.Header
	Provider ProviderData
}

type FileList struct {
	Files []File
	Next  string
}

// Files is the capability of providers that store files. Returned Files carry the provider's file id in Provider and never carry Data.
type Files interface {
	UploadFile(ctx context.Context, c *FileUploadCall) (*File, error)
	GetFile(ctx context.Context, id string) (*File, error)
	ListFiles(ctx context.Context, c *FileListCall) (*FileList, error)
	DeleteFile(ctx context.Context, id string) error
}
