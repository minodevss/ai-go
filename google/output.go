package google

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
)

const (
	codeExecutionName = "code_execution"
	outcomeOK         = "OUTCOME_OK"
)

type responseBody struct {
	Candidates     []candidate     `json:"candidates"`
	PromptFeedback json.RawMessage `json:"promptFeedback"`
	UsageMetadata  json.RawMessage `json:"usageMetadata"`
	ModelVersion   string          `json:"modelVersion"`
	ResponseID     string          `json:"responseId"`
	Error          *errorEnvelope  `json:"error"`
}

type candidate struct {
	Content struct {
		Parts []json.RawMessage `json:"parts"`
	} `json:"content"`
	FinishReason       string          `json:"finishReason"`
	FinishMessage      string          `json:"finishMessage"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata"`
	URLContextMetadata json.RawMessage `json:"urlContextMetadata"`
	CitationMetadata   json.RawMessage `json:"citationMetadata"`
	SafetyRatings      json.RawMessage `json:"safetyRatings"`
}

type partProbe struct {
	Text             *string       `json:"text"`
	Thought          bool          `json:"thought"`
	ThoughtSignature string        `json:"thoughtSignature"`
	FunctionCall     *callIn       `json:"functionCall"`
	ExecutableCode   *codeIn       `json:"executableCode"`
	CodeResult       *codeResultIn `json:"codeExecutionResult"`
	ToolCall         *serverToolIn `json:"toolCall"`
	ToolResponse     *serverToolIn `json:"toolResponse"`
	InlineData       *blobIn       `json:"inlineData"`
	FileData         *fileRefIn    `json:"fileData"`
}

type callIn struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type codeIn struct {
	ID       string `json:"id"`
	Language string `json:"language"`
	Code     string `json:"code"`
}

type codeResultIn struct {
	ID      string `json:"id"`
	Outcome string `json:"outcome"`
	Output  string `json:"output"`
}

type serverToolIn struct {
	ID       string          `json:"id"`
	ToolType string          `json:"toolType"`
	Args     json.RawMessage `json:"args"`
	Response json.RawMessage `json:"response"`
}

type blobIn struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileRefIn struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type converter struct {
	b          *backend
	responseID string
	calls      int
	openCode   []string
}

func (c *converter) part(raw json.RawMessage) (ai.Part, error) {
	var p partProbe
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("google: decode part: %w", err)
	}
	switch {
	case p.FunctionCall != nil:
		return c.functionCall(raw, p.FunctionCall), nil
	case p.ExecutableCode != nil:
		return c.executableCode(raw, p.ExecutableCode), nil
	case p.CodeResult != nil:
		return c.codeResult(raw, p.CodeResult), nil
	case p.ToolCall != nil:
		return c.serverToolCall(raw, p.ToolCall), nil
	case p.ToolResponse != nil:
		return c.serverToolResult(raw, p.ToolResponse), nil
	case p.InlineData != nil:
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return nil, fmt.Errorf("google: decode inline data: %w", err)
		}
		return ai.File{MediaType: p.InlineData.MimeType, Data: data, Provider: c.b.encode(partData{Part: raw})}, nil
	case p.FileData != nil:
		return ai.File{MediaType: p.FileData.MimeType, URL: p.FileData.FileURI, Provider: c.b.encode(partData{Part: raw})}, nil
	case p.Text != nil && p.Thought:
		return ai.Reasoning{Text: *p.Text, Provider: c.b.encode(partData{Part: raw})}, nil
	case p.Text != nil:
		return ai.Text{Text: *p.Text, Provider: c.b.encode(partData{Part: raw})}, nil
	default:
		return ai.Custom{Kind: providerName + ".part", Data: raw, Provider: c.b.encode(partData{Part: raw})}, nil
	}
}

func (c *converter) functionCall(raw json.RawMessage, in *callIn) ai.ToolCall {
	args := argsOf(in.Args)
	id, generated := in.ID, false
	if id == "" {
		id, generated = generatedID("call_", c.responseID, in.Name, args, c.calls), true
	}
	c.calls++
	return ai.ToolCall{ID: id, Name: in.Name, Input: args, Provider: c.b.encode(partData{Part: raw, GeneratedID: generated})}
}

func (c *converter) executableCode(raw json.RawMessage, in *codeIn) ai.ToolCall {
	input := pdata.JSON(providerName, map[string]string{"language": in.Language, "code": in.Code})
	id := in.ID
	if id == "" {
		id = generatedID("code_", c.responseID, codeExecutionName, input, c.calls)
	}
	c.calls++
	c.openCode = append(c.openCode, id)
	return ai.ToolCall{ID: id, Name: codeExecutionName, Input: input, ProviderExecuted: true, Provider: c.b.encode(partData{Part: raw})}
}

func (c *converter) codeResult(raw json.RawMessage, in *codeResultIn) ai.ToolResult {
	id := in.ID
	if id == "" && len(c.openCode) > 0 {
		id = c.openCode[0]
	}
	if len(c.openCode) > 0 {
		c.openCode = c.openCode[1:]
	}
	return ai.ToolResult{
		CallID:           id,
		Name:             codeExecutionName,
		Output:           ai.ToolOutput{JSON: pdata.JSON(providerName, map[string]string{"outcome": in.Outcome, "output": in.Output})},
		IsError:          in.Outcome != "" && in.Outcome != outcomeOK,
		ProviderExecuted: true,
		Provider:         c.b.encode(partData{Part: raw}),
	}
}

func (c *converter) serverToolCall(raw json.RawMessage, in *serverToolIn) ai.ToolCall {
	args := argsOf(in.Args)
	id := in.ID
	if id == "" {
		id = generatedID("tool_", c.responseID, in.ToolType, args, c.calls)
	}
	c.calls++
	c.openCode = append(c.openCode, id)
	return ai.ToolCall{ID: id, Name: strings.ToLower(in.ToolType), Input: args, ProviderExecuted: true, Provider: c.b.encode(partData{Part: raw})}
}

func (c *converter) serverToolResult(raw json.RawMessage, in *serverToolIn) ai.ToolResult {
	id := in.ID
	if id == "" && len(c.openCode) > 0 {
		id = c.openCode[0]
	}
	if len(c.openCode) > 0 {
		c.openCode = c.openCode[1:]
	}
	out := in.Response
	if len(out) == 0 {
		out = json.RawMessage("{}")
	}
	return ai.ToolResult{CallID: id, Name: strings.ToLower(in.ToolType), Output: ai.ToolOutput{JSON: out}, ProviderExecuted: true, Provider: c.b.encode(partData{Part: raw})}
}

func (r responseBody) first() candidate {
	if len(r.Candidates) == 0 {
		return candidate{}
	}
	return r.Candidates[0]
}

func (r responseBody) failure(b *backend) error {
	if r.Error == nil {
		return nil
	}
	return r.Error.apiError(b.name, nil)
}

func (b *backend) convertParts(raws []json.RawMessage, responseID string) (ai.Parts, bool, error) {
	conv := &converter{b: b, responseID: responseID}
	parts := make(ai.Parts, 0, len(raws))
	clientCall := false
	for _, raw := range raws {
		p, err := conv.part(raw)
		if err != nil {
			return nil, false, err
		}
		if call, ok := p.(ai.ToolCall); ok && !call.ProviderExecuted {
			clientCall = true
		}
		parts = append(parts, p)
	}
	return parts, clientCall, nil
}

type metadata struct {
	responseID    string
	modelVersion  string
	finishMessage string
	fields        map[string]json.RawMessage
}

func (m *metadata) merge(key string, raw json.RawMessage) {
	if len(raw) == 0 || string(raw) == "null" {
		return
	}
	if m.fields == nil {
		m.fields = map[string]json.RawMessage{}
	}
	m.fields[key] = mergeObjects(m.fields[key], raw)
}

func (m *metadata) take(body responseBody, cand candidate) {
	m.responseID = cmp.Or(body.ResponseID, m.responseID)
	m.modelVersion = cmp.Or(body.ModelVersion, m.modelVersion)
	m.finishMessage = cmp.Or(cand.FinishMessage, m.finishMessage)
	m.merge("groundingMetadata", cand.GroundingMetadata)
	m.merge("urlContextMetadata", cand.URLContextMetadata)
	m.merge("citationMetadata", cand.CitationMetadata)
	m.merge("safetyRatings", cand.SafetyRatings)
	m.merge("promptFeedback", body.PromptFeedback)
}

func (m *metadata) provider(b *backend) ai.ProviderData {
	return b.response(ResponseData{
		ResponseID:         m.responseID,
		ModelVersion:       m.modelVersion,
		FinishMessage:      m.finishMessage,
		GroundingMetadata:  m.fields["groundingMetadata"],
		URLContextMetadata: m.fields["urlContextMetadata"],
		CitationMetadata:   m.fields["citationMetadata"],
		SafetyRatings:      m.fields["safetyRatings"],
		PromptFeedback:     m.fields["promptFeedback"],
	})
}

func mergeObjects(prev, next json.RawMessage) json.RawMessage {
	if len(prev) == 0 {
		return next
	}
	var a, b map[string]json.RawMessage
	if json.Unmarshal(prev, &a) != nil || json.Unmarshal(next, &b) != nil {
		return next
	}
	out := maps.Clone(a)
	for k, v := range b {
		out[k] = mergeValues(a[k], v)
	}
	return pdata.JSON(providerName, out)
}

func mergeValues(prev, next json.RawMessage) json.RawMessage {
	var a, b []json.RawMessage
	if json.Unmarshal(prev, &a) == nil && json.Unmarshal(next, &b) == nil {
		return pdata.JSON(providerName, append(a, b...))
	}
	return next
}

type usageBody struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
}

func usageOf(raw json.RawMessage) ai.Usage {
	if len(raw) == 0 || string(raw) == "null" {
		return ai.Usage{}
	}
	var u usageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{
		Input:     u.PromptTokenCount + u.ToolUsePromptTokenCount,
		Output:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
		CacheRead: u.CachedContentTokenCount,
		Reasoning: u.ThoughtsTokenCount,
		Raw:       raw,
	}
}

func finishOf(raw string, blocked string, clientCall bool) (ai.FinishReason, string) {
	reason := strings.TrimPrefix(raw, "FINISH_REASON_")
	switch reason {
	case "":
		if blocked != "" {
			return ai.FinishContentFilter, blocked
		}
		return ai.FinishOther, raw
	case "STOP":
		if clientCall {
			return ai.FinishToolCalls, raw
		}
		return ai.FinishStop, raw
	case "MAX_TOKENS", "CONTINUATION":
		return ai.FinishLength, raw
	case "SAFETY", "ESCALATION", "PUP_LIMITED_DISABLED", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return ai.FinishContentFilter, raw
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE", "IMAGE_OTHER", "NO_IMAGE":
		return ai.FinishError, raw
	default:
		return ai.FinishOther, raw
	}
}

func blockReason(feedback json.RawMessage) string {
	var f struct {
		BlockReason string `json:"blockReason"`
	}
	if err := json.Unmarshal(feedback, &f); err != nil {
		return ""
	}
	return f.BlockReason
}
