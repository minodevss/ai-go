package google_test

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func streamKinds(parts []ai.StreamPart) []string {
	var out []string
	for _, p := range parts {
		switch v := p.(type) {
		case ai.TextStart:
			out = append(out, "text-start")
		case ai.TextDelta:
			out = append(out, "text:"+v.Text)
		case ai.TextEnd:
			out = append(out, "text-end")
		case ai.ReasoningStart:
			out = append(out, "reasoning-start")
		case ai.ReasoningDelta:
			out = append(out, "reasoning:"+v.Text)
		case ai.ReasoningEnd:
			out = append(out, "reasoning-end")
		case ai.ToolInputStart:
			out = append(out, "input-start:"+v.Name)
		case ai.ToolInputDelta:
			out = append(out, "input:"+v.Text)
		case ai.ToolInputEnd:
			out = append(out, "input-end")
		case ai.ToolCall:
			out = append(out, "call:"+v.Name)
		case ai.Citation:
			out = append(out, "cite:"+v.CitedText)
		case ai.File:
			out = append(out, "file")
		default:
		}
	}
	return out
}

func runStream(t *testing.T, tools []ai.Tool, chunks ...string) ([]ai.StreamPart, *ai.TextResult, error) {
	t.Helper()
	f := newFakeServer(t, sseReply(chunks...))
	s := ai.StreamText(t.Context(), f.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, Tools: tools, MaxSteps: 1, MaxRetries: new(0)})
	var parts []ai.StreamPart
	for p := range s.Parts() {
		parts = append(parts, p)
	}
	res, err := s.Result()
	return parts, res, err
}

func chunkOf(parts string, tail string) string {
	return `{"candidates":[{"content":{"role":"model","parts":[` + parts + `]}` + tail + `}],"responseId":"r1"}`
}

func TestLanguageModel_streamPartBoundaries(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   []string
	}{
		{
			name: "thought then text then signed tail",
			chunks: []string{
				chunkOf(`{"text":"th","thought":true}`, ""),
				chunkOf(`{"text":"ink","thought":true}`, ""),
				chunkOf(`{"text":"He"}`, ""),
				chunkOf(`{"text":"llo"}`, ""),
				chunkOf(`{"text":"","thoughtSignature":"S1"}`, `,"finishReason":"STOP"`),
			},
			want: []string{"reasoning-start", "reasoning:th", "reasoning:ink", "reasoning-end", "text-start", "text:He", "text:llo", "text-end", "text-start", "text-end"},
		},
		{
			name: "signed chunk is its own part",
			chunks: []string{
				chunkOf(`{"text":"a"}`, ""),
				chunkOf(`{"text":"b","thoughtSignature":"S1"}`, ""),
				chunkOf(`{"text":"c"}`, `,"finishReason":"STOP"`),
			},
			want: []string{"text-start", "text:a", "text-end", "text-start", "text:b", "text-end", "text-start", "text:c", "text-end"},
		},
		{
			name: "empty unsigned chunks are ignored",
			chunks: []string{
				chunkOf(`{"text":""}`, ""),
				chunkOf(`{"text":"a"}`, `,"finishReason":"STOP"`),
			},
			want: []string{"text-start", "text:a", "text-end"},
		},
		{
			name: "function call closes text and streams its input",
			chunks: []string{
				chunkOf(`{"text":"calling"}`, ""),
				chunkOf(`{"functionCall":{"id":"c1","name":"f","args":{"x":1}},"thoughtSignature":"S1"}`, `,"finishReason":"STOP"`),
			},
			want: []string{"text-start", "text:calling", "text-end", "input-start:f", `input:{"x":1}`, "input-end", "call:f"},
		},
		{
			name: "files are whole parts",
			chunks: []string{
				chunkOf(`{"inlineData":{"mimeType":"image/png","data":"AQI="}}`, `,"finishReason":"STOP"`),
			},
			want: []string{"file"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts, _, err := runStream(t, nil, tt.chunks...)
			if err != nil {
				t.Fatalf("Result() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, streamKinds(parts)); diff != "" {
				t.Errorf("stream parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLanguageModel_streamSignaturesLandOnTheirParts(t *testing.T) {
	_, res, err := runStream(t, nil,
		chunkOf(`{"text":"Hi "}`, ""),
		chunkOf(`{"text":"there","thoughtSignature":"S-ONE"}`, ""),
		chunkOf(`{"text":"","thoughtSignature":"S-TWO"}`, `,"finishReason":"STOP"`),
	)
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	var got []string
	for _, p := range res.Content {
		text, ok := p.(ai.Text)
		if !ok {
			t.Fatalf("part = %T, want ai.Text", p)
		}
		data, _ := ai.ProviderValue[map[string]any](text.Provider, "google")
		raw, _ := data["part"].(map[string]any)
		got = append(got, fmt.Sprintf("%q/%v", raw["text"], raw["thoughtSignature"]))
	}
	want := []string{`"Hi "/<nil>`, `"there"/S-ONE`, `""/S-TWO`}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("part placement mismatch (-want +got):\n%s", diff)
	}
}

func TestLanguageModel_streamCitationsFollowTheirText(t *testing.T) {
	meta := `,"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://a.example","title":"A"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":3,"text":"two"},"groundingChunkIndices":[0]}]}`
	meta2 := `,"finishReason":"STOP","groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://b.example","title":"B"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":3,"text":"one"},"groundingChunkIndices":[1]}]}`
	parts, res, err := runStream(t, []ai.Tool{google.Search()},
		chunkOf(`{"text":"one","thoughtSignature":"S1"}`, ""),
		chunkOf(`{"text":"two","thoughtSignature":"S2"}`, meta),
		chunkOf(``, meta2),
	)
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	want := []string{"text-start", "text:one", "text-start", "text:two", "text-end", "cite:one", "text-end", "cite:two"}
	if diff := cmp.Diff(want, streamKinds(parts)); diff != "" {
		t.Errorf("stream parts mismatch (-want +got):\n%s", diff)
	}
	first, _ := res.Content[0].(ai.Text)
	second, _ := res.Content[1].(ai.Text)
	if len(first.Citations) != 1 || first.Citations[0].URL != "https://b.example" || len(second.Citations) != 1 || second.Citations[0].URL != "https://a.example" {
		t.Errorf("citations = %+v / %+v", first.Citations, second.Citations)
	}
	data, _ := ai.ProviderValue[google.ResponseData](res.Provider, "google")
	if strings.Count(string(data.GroundingMetadata), "uri") != 2 {
		t.Errorf("merged grounding metadata = %s, want both chunks", data.GroundingMetadata)
	}
}

func TestLanguageModel_streamUsageAndFinish(t *testing.T) {
	_, res, err := runStream(t, nil,
		`{"candidates":[{"content":{"parts":[{"text":"a"}]}}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":1}}`,
		`{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,"thoughtsTokenCount":6}}`,
	)
	if err != nil {
		t.Fatalf("Result() error = %v", err)
	}
	if res.FinishReason != ai.FinishLength || res.Usage.Input != 9 || res.Usage.Output != 10 || res.Usage.Reasoning != 6 {
		t.Errorf("finish = %q, usage = %+v", res.FinishReason, res.Usage)
	}
}

func TestLanguageModel_streamErrors(t *testing.T) {
	t.Run("error event", func(t *testing.T) {
		_, res, err := runStream(t, nil,
			chunkOf(`{"text":"a"}`, ""),
			`{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3s"}]}}`,
		)
		apiErr, ok := errors.AsType[*ai.APIError](err)
		if !ok || apiErr.Code != "UNAVAILABLE" || apiErr.StatusCode != http.StatusServiceUnavailable || apiErr.RetryAfter.Seconds() != 3 {
			t.Fatalf("Result() error = %v, want an UNAVAILABLE APIError with a 3s delay", err)
		}
		if res == nil || len(res.Messages) != 0 {
			t.Errorf("result = %+v, want no completed steps", res)
		}
	})
	t.Run("truncated stream", func(t *testing.T) {
		_, _, err := runStream(t, nil, chunkOf(`{"text":"a"}`, ""))
		if err == nil || !strings.Contains(err.Error(), "stream ended before the response finished") {
			t.Errorf("Result() error = %v, want a truncation error", err)
		}
	})
	t.Run("bad event", func(t *testing.T) {
		_, _, err := runStream(t, nil, `{not json`)
		if err == nil || !strings.Contains(err.Error(), "decode stream event") {
			t.Errorf("Result() error = %v, want a decode error", err)
		}
	})
	t.Run("blocked prompt", func(t *testing.T) {
		_, res, err := runStream(t, nil, `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":4}}`)
		if err != nil || res.FinishReason != ai.FinishContentFilter || res.Usage.Input != 4 {
			t.Errorf("Result() = %+v, %v; want a content-filter finish", res, err)
		}
	})
	t.Run("http error before the stream", func(t *testing.T) {
		f := newFakeServer(t, jsonErrorReply(429, `{"error":{"code":429,"message":"slow","status":"RESOURCE_EXHAUSTED"}}`))
		_, err := ai.StreamText(t.Context(), f.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}, MaxRetries: new(0)}).Result()
		if !errors.Is(err, ai.ErrRateLimit) {
			t.Errorf("Result() error = %v, want a rate limit error", err)
		}
	})
}

func TestLanguageModel_streamStopsEarly(t *testing.T) {
	f := newFakeServer(t, sseReply(chunkOf(`{"text":"a"}`, ""), chunkOf(`{"text":"b"}`, `,"finishReason":"STOP"`)))
	s := ai.StreamText(t.Context(), f.provider().Model("gemini-3-flash"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	for p := range s.Parts() {
		if _, ok := p.(ai.TextDelta); ok {
			break
		}
	}
	if _, err := s.Result(); !errors.Is(err, errCanceled) {
		t.Errorf("Result() error = %v, want context canceled", err)
	}
}
