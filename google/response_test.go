package google_test

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func replyWith(finish string, usage string) string {
	body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"` + finish + `"}]`
	if usage != "" {
		body += `,"usageMetadata":` + usage
	}
	return body + `}`
}

func TestLanguageModel_finishReasons(t *testing.T) {
	tests := []struct {
		name  string
		reply string
		want  ai.FinishReason
		raw   string
	}{
		{name: "stop", reply: replyWith("STOP", ""), want: ai.FinishStop, raw: "STOP"},
		{name: "stop with client call", reply: `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}`, want: ai.FinishToolCalls, raw: "STOP"},
		{name: "stop with provider-executed call only", reply: `{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"1"}},{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"1"}}]},"finishReason":"STOP"}]}`, want: ai.FinishStop, raw: "STOP"},
		{name: "max tokens", reply: replyWith("MAX_TOKENS", ""), want: ai.FinishLength, raw: "MAX_TOKENS"},
		{name: "continuation", reply: replyWith("CONTINUATION", ""), want: ai.FinishLength, raw: "CONTINUATION"},
		{name: "escalation", reply: replyWith("ESCALATION", ""), want: ai.FinishContentFilter, raw: "ESCALATION"},
		{name: "pup limited disabled", reply: replyWith("PUP_LIMITED_DISABLED", ""), want: ai.FinishContentFilter, raw: "PUP_LIMITED_DISABLED"},
		{name: "image other", reply: replyWith("IMAGE_OTHER", ""), want: ai.FinishError, raw: "IMAGE_OTHER"},
		{name: "no image", reply: replyWith("NO_IMAGE", ""), want: ai.FinishError, raw: "NO_IMAGE"},
		{name: "safety", reply: replyWith("SAFETY", ""), want: ai.FinishContentFilter, raw: "SAFETY"},
		{name: "recitation", reply: replyWith("RECITATION", ""), want: ai.FinishContentFilter, raw: "RECITATION"},
		{name: "prohibited", reply: replyWith("PROHIBITED_CONTENT", ""), want: ai.FinishContentFilter, raw: "PROHIBITED_CONTENT"},
		{name: "malformed call", reply: replyWith("MALFORMED_FUNCTION_CALL", ""), want: ai.FinishError, raw: "MALFORMED_FUNCTION_CALL"},
		{name: "missing signature", reply: replyWith("MISSING_THOUGHT_SIGNATURE", ""), want: ai.FinishError, raw: "MISSING_THOUGHT_SIGNATURE"},
		{name: "other", reply: replyWith("OTHER", ""), want: ai.FinishOther, raw: "OTHER"},
		{name: "vertex prefix", reply: replyWith("FINISH_REASON_MAX_TOKENS", ""), want: ai.FinishLength, raw: "FINISH_REASON_MAX_TOKENS"},
		{name: "prompt blocked", reply: `{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}`, want: ai.FinishContentFilter, raw: "PROHIBITED_CONTENT"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(tt.reply))
			resp := sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")})
			if resp.FinishReason != tt.want || resp.FinishRaw != tt.raw {
				t.Errorf("finish = %q (%q), want %q (%q)", resp.FinishReason, resp.FinishRaw, tt.want, tt.raw)
			}
		})
	}
}

func TestLanguageModel_usage(t *testing.T) {
	tests := []struct {
		name  string
		usage string
		want  ai.Usage
	}{
		{name: "thoughts are added to output", usage: `{"promptTokenCount":100,"candidatesTokenCount":10,"thoughtsTokenCount":40,"totalTokenCount":150}`, want: ai.Usage{Input: 100, Output: 50, Reasoning: 40}},
		{name: "cached tokens are part of the prompt", usage: `{"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":5}`, want: ai.Usage{Input: 100, Output: 5, CacheRead: 80}},
		{name: "tool use prompt counts as input", usage: `{"promptTokenCount":100,"toolUsePromptTokenCount":25,"candidatesTokenCount":5}`, want: ai.Usage{Input: 125, Output: 5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, jsonReply(replyWith("STOP", tt.usage)))
			got := sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")}).Usage
			if string(got.Raw) != tt.usage {
				t.Errorf("Usage.Raw = %s, want the response usageMetadata", got.Raw)
			}
			got.Raw = nil
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("usage mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

const groundedReply = `{"candidates":[{"content":{"role":"model","parts":[
	{"text":"서울은 수도입니다. "},{"text":"Seoul has 9M people. Busan has 3M."}]},
	"finishReason":"STOP",
	"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://a.example","title":"A"}},{"retrievedContext":{"uri":"https://b.example","title":"B"}}],
	"groundingSupports":[
		{"segment":{"startIndex":%d,"endIndex":%d,"text":"Seoul has 9M people."},"groundingChunkIndices":[0,1],"confidenceScores":[0.9,0.8]},
		{"segment":{"startIndex":0,"endIndex":5,"text":"Busan has 3M."},"groundingChunkIndices":[1]},
		{"segment":{"startIndex":900,"endIndex":950,"text":"missing"},"groundingChunkIndices":[0]},
		{"segment":{"startIndex":0,"endIndex":5,"text":"Seoul"},"groundingChunkIndices":[7]}]}}]}`

func TestLanguageModel_citations(t *testing.T) {
	lead := "서울은 수도입니다. "
	start := len(lead)
	reply := fmt.Sprintf(groundedReply, start, start+len("Seoul has 9M people."))
	f := newFakeServer(t, jsonReply(reply))
	resp := sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")})
	first, _ := resp.Content[0].(ai.Text)
	second, _ := resp.Content[1].(ai.Text)
	if len(first.Citations) != 0 {
		t.Errorf("first text has citations %v, want none", first.Citations)
	}
	var got []string
	for _, c := range second.Citations {
		got = append(got, c.URL+"|"+c.Title+"|"+c.CitedText+"|"+strconv.Itoa(c.Start)+"-"+strconv.Itoa(c.End))
	}
	want := []string{
		"https://a.example|A|Seoul has 9M people.|0-20",
		"https://b.example|B|Seoul has 9M people.|0-20",
		"https://b.example|B|Busan has 3M.|21-34",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("citations mismatch (-want +got):\n%s", diff)
	}
	data, ok := ai.ProviderValue[google.ResponseData](resp.Provider, "google")
	if !ok || !strings.Contains(string(data.GroundingMetadata), "groundingSupports") {
		t.Errorf("response data = %+v, want the grounding metadata", data)
	}
}

func TestLanguageModel_providerExecutedPairs(t *testing.T) {
	f := newFakeServer(t, jsonReply(`{"candidates":[{"content":{"parts":[
		{"executableCode":{"language":"PYTHON","code":"print(1)"}},
		{"codeExecutionResult":{"outcome":"OUTCOME_FAILED","output":"boom"}},
		{"toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["x"]},"id":"s1"}},
		{"toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"ok":true},"id":"s1"}},
		{"videoMetadata":{"x":1}}]},"finishReason":"STOP"}],"responseId":"r"}`))
	resp := sendHistory(t, f, "gemini-3-flash", []ai.Message{ai.User("hi")})
	call, _ := resp.Content[0].(ai.ToolCall)
	res, _ := resp.Content[1].(ai.ToolResult)
	if !call.ProviderExecuted || call.Name != "code_execution" || res.CallID != call.ID || !res.IsError || !res.ProviderExecuted {
		t.Errorf("code execution parts = %+v, %+v", call, res)
	}
	search, _ := resp.Content[2].(ai.ToolCall)
	found, _ := resp.Content[3].(ai.ToolResult)
	if search.ID != "s1" || search.Name != "google_search_web" || found.CallID != "s1" {
		t.Errorf("server tool parts = %+v, %+v", search, found)
	}
	if custom, ok := resp.Content[4].(ai.Custom); !ok || custom.Kind != "google.part" {
		t.Errorf("unknown part = %#v, want Custom google.part", resp.Content[4])
	}
}

func TestLanguageModel_errors(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		body      string
		header    string
		wantCode  string
		wantAfter string
		wantIs    error
		wantRetry bool
		wantNotIs error
	}{
		{
			name:     "rate limit with RetryInfo",
			status:   429,
			body:     `{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34s"}]}}`,
			wantCode: "RESOURCE_EXHAUSTED", wantAfter: "34s", wantIs: ai.ErrRateLimit, wantRetry: true, wantNotIs: ai.ErrQuota,
		},
		{
			name:   "header beats RetryInfo",
			status: 429, header: "2",
			body:     `{"error":{"code":429,"message":"slow","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34s"}]}}`,
			wantCode: "RESOURCE_EXHAUSTED", wantAfter: "2s", wantIs: ai.ErrRateLimit, wantRetry: true,
		},
		{
			name:     "overloaded",
			status:   503,
			body:     `{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}`,
			wantCode: "UNAVAILABLE", wantRetry: true,
		},
		{
			name:     "prepay depleted is quota and not retried",
			status:   402,
			body:     `{"error":{"code":402,"message":"Your prepay credit balance is depleted.","status":"RESOURCE_EXHAUSTED"}}`,
			wantCode: "RESOURCE_EXHAUSTED_QUOTA", wantIs: ai.ErrQuota,
		},
		{
			name:     "bad key",
			status:   400,
			body:     `{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}`,
			wantCode: "INVALID_ARGUMENT",
		},
		{
			name:     "forbidden",
			status:   403,
			body:     `{"error":{"code":403,"message":"denied","status":"PERMISSION_DENIED"}}`,
			wantCode: "PERMISSION_DENIED", wantIs: ai.ErrAuth,
		},
		{
			name:     "context too long",
			status:   400,
			body:     `{"error":{"code":400,"message":"The input token count (1100000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}`,
			wantCode: "context_length_exceeded", wantIs: ai.ErrContextTooLong,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
				if tt.header != "" {
					w.Header().Set("Retry-After", tt.header)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := f.provider().Model("gemini-3-flash").Generate(t.Context(), &ai.Call{Messages: []ai.Message{ai.User("hi")}})
			apiErr, ok := errors.AsType[*ai.APIError](err)
			if !ok {
				t.Fatalf("Generate() error = %v, want *ai.APIError", err)
			}
			if apiErr.Code != tt.wantCode || apiErr.StatusCode != tt.status || apiErr.Provider != "google" {
				t.Errorf("error = %+v, want code %q status %d", apiErr, tt.wantCode, tt.status)
			}
			wantAfter, perr := time.ParseDuration(cmpOr(tt.wantAfter, "0s"))
			if perr != nil || apiErr.RetryAfter != wantAfter {
				t.Errorf("RetryAfter = %v, want %v", apiErr.RetryAfter, wantAfter)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("errors.Is(%v, %v) = false", err, tt.wantIs)
			}
			if tt.wantNotIs != nil && errors.Is(err, tt.wantNotIs) {
				t.Errorf("errors.Is(%v, %v) = true", err, tt.wantNotIs)
			}
			if apiErr.Retryable() != tt.wantRetry {
				t.Errorf("Retryable() = %v, want %v", apiErr.Retryable(), tt.wantRetry)
			}
		})
	}
}

func cmpOr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
