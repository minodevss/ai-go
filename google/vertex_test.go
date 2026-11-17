package google_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func vertexProvider(f *fakeServer, ts google.TokenSource) *google.Provider {
	return google.New(
		google.WithVertex("my-proj", "us-central1"),
		google.WithBaseURL(f.URL+"/v1/projects/my-proj/locations/us-central1"),
		google.WithAPIKey("ignored"),
		google.WithTokenSource(ts),
	)
}

func TestVertex_generateUsesBearerAndPublisherPath(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	var asked atomic.Int32
	p := vertexProvider(f, func(context.Context) (string, error) {
		asked.Add(1)
		return "ya29.token", nil
	})
	res, err := ai.GenerateText(t.Context(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)})
	if err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	got := f.last(t)
	if got.Path != "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-3-flash:generateContent" {
		t.Errorf("path = %q", got.Path)
	}
	if got.Header.Get("Authorization") != "Bearer ya29.token" || got.Header.Get("x-goog-api-key") != "" {
		t.Errorf("auth headers = %q, key %q", got.Header.Get("Authorization"), got.Header.Get("x-goog-api-key"))
	}
	if res.Text != "hi" || asked.Load() != 1 {
		t.Errorf("text = %q, token requests = %d", res.Text, asked.Load())
	}
	if m := p.Model("x"); m.Provider() != "google-vertex" {
		t.Errorf("Provider() = %q", m.Provider())
	}
}

func TestVertex_streamUsesBearer(t *testing.T) {
	f := newFakeServer(t, sseReply(`{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`))
	p := vertexProvider(f, func(context.Context) (string, error) { return "ya29.stream", nil })
	res, err := ai.StreamText(t.Context(), p.Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}).Result()
	if err != nil || res.Text != "hi" {
		t.Fatalf("Result() = %+v, %v", res, err)
	}
	got := f.last(t)
	if got.Header.Get("Authorization") != "Bearer ya29.stream" || got.Query != "alt=sse" || !strings.HasSuffix(got.Path, ":streamGenerateContent") {
		t.Errorf("request = %s?%s with %q", got.Path, got.Query, got.Header.Get("Authorization"))
	}
}

func TestVertex_tokenErrorIsReported(t *testing.T) {
	f := newFakeServer(t, jsonReply(textReply))
	boom := errors.New("no credentials")
	p := vertexProvider(f, func(context.Context) (string, error) { return "", boom })
	_, err := p.Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
	if !errors.Is(err, boom) || f.count() != 0 {
		t.Errorf("Generate() error = %v after %d requests, want the token error and no request", err, f.count())
	}
}

func TestVertex_embedUsesPredict(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"predictions":[{"embeddings":{"values":[0.5,0.25],"statistics":{"token_count":3,"truncated":false}}},{"embeddings":{"values":[1,2],"statistics":{"token_count":4}}}]}`))
	p := vertexProvider(f, func(context.Context) (string, error) { return "tok", nil })
	model := p.EmbeddingModel("text-embedding-005")
	if model.MaxBatch() != 5 || p.EmbeddingModel("gemini-embedding-001").MaxBatch() != 1 {
		t.Errorf("MaxBatch() = %d and %d, want 5 and 1", model.MaxBatch(), p.EmbeddingModel("gemini-embedding-001").MaxBatch())
	}
	res, err := ai.Embed(t.Context(), model, ai.EmbedRequest{Texts: []string{"a", "b"}, InputType: ai.InputDocument, Dimensions: 2})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	got := f.last(t)
	if got.Path != "/v1/projects/my-proj/locations/us-central1/publishers/google/models/text-embedding-005:predict" {
		t.Errorf("path = %q", got.Path)
	}
	equalJSON(t, `{"instances":[{"content":"a","task_type":"RETRIEVAL_DOCUMENT"},{"content":"b","task_type":"RETRIEVAL_DOCUMENT"}],"parameters":{"outputDimensionality":2}}`, got.Body)
	if len(res.Embeddings) != 2 || res.Embeddings[0][1] != 0.25 || res.Usage.Input != 7 {
		t.Errorf("result = %+v", res)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestVertex_environmentDefaults(t *testing.T) {
	tests := []struct {
		name     string
		project  string
		location string
		envLoc   string
		want     string
	}{
		{name: "environment", envLoc: "asia-northeast3", want: "https://asia-northeast3-aiplatform.googleapis.com/v1/projects/env-proj/locations/asia-northeast3/publishers/google/models/m:generateContent"},
		{name: "explicit wins", project: "p", location: "global", envLoc: "asia-northeast3", want: "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/m:generateContent"},
		{name: "location defaults to global", want: "https://aiplatform.googleapis.com/v1/projects/env-proj/locations/global/publishers/google/models/m:generateContent"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
			t.Setenv("GOOGLE_CLOUD_LOCATION", tt.envLoc)
			var got string
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				got = r.URL.String()
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(textReply))}, nil
			})}
			p := google.New(google.WithVertex(tt.project, tt.location), google.WithHTTPClient(client), google.WithTokenSource(func(context.Context) (string, error) { return "tok", nil }))
			if _, err := p.Model("m").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}}); err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("URL = %q, want %q", got, tt.want)
			}
		})
	}
}
