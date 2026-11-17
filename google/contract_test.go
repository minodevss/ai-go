package google_test

import (
	"encoding/json"
	"net/http"
	"regexp"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
	"github.com/minodevss/ai-go/providertest"
)

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

var callIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

const fixtureBaseURL = "https://generativelanguage.googleapis.com/v1beta"

func newModel(t *testing.T, client *http.Client) ai.LanguageModel {
	t.Helper()
	return google.New(google.WithAPIKey("test-key"), google.WithBaseURL(fixtureBaseURL), google.WithHTTPClient(client)).Model("gemini-3-flash")
}

func TestLanguageModel_contract(t *testing.T) {
	weather := ai.NewClientTool[weatherInput]("get_weather", "Weather by city")
	user := []ai.Message{ai.User("hello")}
	providertest.Run(t, providertest.Adapter{
		NewModel:    newModel,
		CallIDValid: callIDPattern.MatchString,
		Wire:        providertest.Wire{Request: requestItems, Response: responseItems, StreamResponse: streamItems, CallIDs: callIDs},
	}, providertest.Config{
		Dir:    "testdata/contract",
		Replay: "replay",
		Schema: &providertest.SchemaCase{
			Name:         "schema",
			Schema:       json.RawMessage(`{"type":"object","properties":{"a":{"type":"string","minLength":1},"b":{"oneOf":[{"type":"integer"},{"type":"null"}]},"c":{"type":"array","items":{"type":"string"},"uniqueItems":true}},"required":["a"]}`),
			WantWarnings: []string{`"minLength"`, `"uniqueItems"`, `"oneOf"`},
		},
		Cases: []providertest.Case{
			{
				Name:       "reasoning_tool",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"reasoning", "tool-call"},
				WantIDs:    []string{"fc_aaa1", "CiQB", "resp_r1"},
				WantUsage:  &ai.Usage{Input: 120, Output: 85, CacheRead: 64, Reasoning: 70},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "parallel_calls",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{"tool-call", "tool-call"},
				WantIDs:    []string{"CjEB", "Busan"},
				WantUsage:  &ai.Usage{Input: 50, Output: 50, Reasoning: 30},
				WantFinish: ai.FinishToolCalls,
			},
			{
				Name:       "text_trailing_signature",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text", "text"},
				WantIDs:    []string{"CiMB", "Hello there"},
				WantUsage:  &ai.Usage{Input: 10, Output: 4},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "grounding",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{google.Search(), google.URLContext()}},
				WantKinds:  []string{"reasoning", "text", "text"},
				WantIDs:    []string{"CiUB", "resp_g1", "urlRetrievalStatus", "groundingSupports"},
				WantUsage:  &ai.Usage{Input: 130, Output: 30, Reasoning: 10},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "code_execution",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{google.CodeExecution()}},
				WantKinds:  []string{"text", "tool-call", "tool-result", "file", "text"},
				WantIDs:    []string{"CiEB", "OUTCOME_OK", "iVBORw0KGgo"},
				WantUsage:  &ai.Usage{Input: 40, Output: 30},
				WantFinish: ai.FinishStop,
			},
			{
				Name:       "safety",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantIDs:    []string{"HARM_CATEGORY_DANGEROUS_CONTENT"},
				WantFinish: ai.FinishContentFilter,
			},
			{
				Name:       "blocked",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{},
				WantUsage:  &ai.Usage{Input: 5},
				WantFinish: ai.FinishContentFilter,
			},
			{
				Name:       "max_tokens",
				Request:    ai.TextRequest{Messages: user},
				WantKinds:  []string{"text"},
				WantUsage:  &ai.Usage{Input: 10, Output: 16, Reasoning: 13},
				WantFinish: ai.FinishLength,
			},
			{
				Name:       "malformed",
				Request:    ai.TextRequest{Messages: user, Tools: []ai.Tool{weather}},
				WantKinds:  []string{},
				WantFinish: ai.FinishError,
			},
		},
	})
}
