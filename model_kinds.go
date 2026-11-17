package ai

import (
	"context"
	"io"
	"net/http"
	"time"
)

type InputType string

const (
	InputDocument InputType = "document"
	InputQuery    InputType = "query"
)

// EmbeddingModel reports MaxBatch as the largest number of inputs per Embed call; 0 means unlimited.
type EmbeddingModel interface {
	Provider() string
	ModelID() string
	MaxBatch() int
	Embed(ctx context.Context, c *EmbedCall) (*EmbedResponse, error)
}

type EmbedCall struct {
	Inputs     []Parts
	InputType  InputType
	Dimensions int
	Headers    http.Header
	Provider   ProviderData
}

type EmbedResponse struct {
	Embeddings [][]float32
	Usage      Usage
	Warnings   []Warning
	Header     http.Header
}

type RerankModel interface {
	Provider() string
	ModelID() string
	Rerank(ctx context.Context, c *RerankCall) (*RerankResponse, error)
}

type RerankCall struct {
	Query     string
	Documents []string
	TopN      int
	Headers   http.Header
	Provider  ProviderData
}

type Ranked struct {
	Index int
	Score float64
}

type RerankResponse struct {
	Ranking  []Ranked
	Usage    Usage
	Warnings []Warning
	Header   http.Header
}

type ImageModel interface {
	Provider() string
	ModelID() string
	GenerateImage(ctx context.Context, c *ImageCall) (*ImageResponse, error)
}

type ImageCall struct {
	Prompt      string
	N           int
	Size        string
	AspectRatio string
	Seed        *int
	Files       []File
	Mask        *File
	Headers     http.Header
	Provider    ProviderData
}

type ImageResponse struct {
	Images   []File
	Usage    Usage
	Warnings []Warning
	Header   http.Header
}

// VideoModel generates videos as asynchronous jobs: StartVideo submits one and PollVideo reports its progress until it is done.
type VideoModel interface {
	Provider() string
	ModelID() string
	StartVideo(ctx context.Context, c *VideoCall) (*VideoJob, error)
	PollVideo(ctx context.Context, job *VideoJob) (*VideoPoll, error)
}

type VideoCall struct {
	Prompt      string
	Duration    time.Duration
	AspectRatio string
	Seed        *int
	Files       []File
	Headers     http.Header
	Provider    ProviderData
}

type VideoJob struct {
	ID       string
	Provider ProviderData
	Header   http.Header
}

// VideoPoll reports a job that is still running while Done and Failure are both unset. Interval, when set, overrides the core's poll interval for the next poll.
type VideoPoll struct {
	Done     bool
	Progress int
	Videos   []File
	Failure  error
	Interval time.Duration
	Warnings []Warning
	Header   http.Header
}

type SpeechModel interface {
	Provider() string
	ModelID() string
	GenerateSpeech(ctx context.Context, c *SpeechCall) (*SpeechResponse, error)
}

type SpeechCall struct {
	Text         string
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	Headers      http.Header
	Provider     ProviderData
}

type SpeechResponse struct {
	Audio    File
	Warnings []Warning
	Header   http.Header
}

type TranscriptionModel interface {
	Provider() string
	ModelID() string
	Transcribe(ctx context.Context, c *TranscribeCall) (*TranscribeResponse, error)
}

type TranscribeCall struct {
	Audio     io.Reader
	MediaType string
	Language  string
	Headers   http.Header
	Provider  ProviderData
}

// Segment.Partial marks streaming text that a later segment of the same speech replaces.
type Segment struct {
	Text    string
	Start   time.Duration
	End     time.Duration
	Speaker string
	Partial bool
}

type TranscribeResponse struct {
	Text     string
	Segments []Segment
	Language string
	Warnings []Warning
	Header   http.Header
}
