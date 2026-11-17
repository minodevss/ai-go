package openai

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/openaiwire"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

type responseBody struct {
	ID                string            `json:"id"`
	Status            string            `json:"status"`
	Output            []json.RawMessage `json:"output"`
	Usage             json.RawMessage   `json:"usage"`
	ServiceTier       string            `json:"service_tier"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type outputItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Status    string `json:"status"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Summary   []struct {
		Text string `json:"text"`
	} `json:"summary"`
	Content []outputContent `json:"content"`
	Action  json.RawMessage `json:"action"`
	Queries json.RawMessage `json:"queries"`
	Code    string          `json:"code"`
	ContID  string          `json:"container_id"`
	Label   string          `json:"server_label"`
	Output  json.RawMessage `json:"output"`
	Err     json.RawMessage `json:"error"`
	Result  string          `json:"result"`
	Format  string          `json:"output_format"`
}

type outputContent struct {
	Type        string            `json:"type"`
	Text        string            `json:"text"`
	Refusal     string            `json:"refusal"`
	Annotations []json.RawMessage `json:"annotations"`
}

type converted struct {
	parts      ai.Parts
	clientCall bool
	refusal    bool
}

func (b responseBody) failure() error {
	if b.Status != "failed" && b.Error == nil {
		return nil
	}
	if b.Error == nil {
		return fmt.Errorf("openai: response %s failed", b.ID)
	}
	return transport.ErrorFromEvent(providerName, b.Error.Code, b.Error.Message)
}

func (b responseBody) metadata() ai.ProviderData {
	return responseProvider(ResponseData{ID: b.ID, ServiceTier: b.ServiceTier})
}

func convertOutput(items []json.RawMessage) (converted, error) {
	var out converted
	for _, raw := range items {
		c, err := convertItem(raw)
		if err != nil {
			return converted{}, err
		}
		out.parts = append(out.parts, c.parts...)
		out.clientCall = out.clientCall || c.clientCall
		out.refusal = out.refusal || c.refusal
	}
	return out, nil
}

func convertItem(raw json.RawMessage) (converted, error) {
	var it outputItem
	if err := json.Unmarshal(raw, &it); err != nil {
		return converted{}, fmt.Errorf("openai: decode output item: %w", err)
	}
	switch it.Type {
	case "reasoning":
		return converted{parts: ai.Parts{ai.Reasoning{Text: reasoningText(it), Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})}}}, nil
	case "message":
		return convertMessage(raw, it), nil
	case "function_call":
		return converted{parts: ai.Parts{functionToolCall(raw, it)}, clientCall: true}, nil
	case "web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "mcp_call":
		return converted{parts: providerTool(raw, it)}, nil
	default:
		return converted{parts: ai.Parts{ai.Custom{Kind: providerName + "." + it.Type, Data: raw, Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})}}}, nil
	}
}

func reasoningText(it outputItem) string {
	var texts []string
	for _, s := range it.Summary {
		texts = append(texts, s.Text)
	}
	if len(texts) == 0 {
		for _, c := range it.Content {
			if c.Type == "reasoning_text" {
				texts = append(texts, c.Text)
			}
		}
	}
	return strings.Join(texts, "\n\n")
}

func convertMessage(raw json.RawMessage, it outputItem) converted {
	var out converted
	if len(it.Content) == 0 {
		out.parts = ai.Parts{ai.Text{Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})}}
		return out
	}
	for i := range it.Content {
		t, refusal := contentText(it, i, raw)
		out.parts = append(out.parts, t)
		out.refusal = out.refusal || refusal
	}
	return out
}

func contentText(it outputItem, i int, raw json.RawMessage) (ai.Text, bool) {
	c := it.Content[i]
	d := partData{ID: it.ID, Index: i, Refusal: c.Type == "refusal"}
	if i == 0 {
		d.Item = raw
	}
	text := c.Text
	if d.Refusal {
		text = c.Refusal
	}
	return ai.Text{Text: text, Citations: openaiwire.Citations(providerName, c.Annotations), Provider: pdata.Encode(providerName, d)}, d.Refusal
}

func functionToolCall(raw json.RawMessage, it outputItem) ai.ToolCall {
	return ai.ToolCall{ID: it.CallID, Name: it.Name, Input: openaiwire.ToolInput(it.Arguments), Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})}
}

func providerTool(raw json.RawMessage, it outputItem) ai.Parts {
	call := ai.ToolCall{ID: it.ID, Name: providerToolName(it), Input: providerToolInput(it), ProviderExecuted: true, Provider: pdata.Encode(providerName, partData{ID: it.ID, Item: raw})}
	res := ai.ToolResult{CallID: it.ID, Name: call.Name, ProviderExecuted: true, IsError: it.Status == "failed" || len(it.Err) > 0 && string(it.Err) != "null", Provider: pdata.Encode(providerName, partData{ID: it.ID, Derived: true})}
	parts := ai.Parts{call}
	switch it.Type {
	case "mcp_call":
		res.Output = ai.ToolOutput{JSON: pdata.JSON(providerName, map[string]json.RawMessage{"output": nullable(it.Output), "error": nullable(it.Err)})}
	case "image_generation_call":
		res.Output = ai.ToolOutput{JSON: withoutKey(raw, "result")}
		parts = append(parts, res)
		if file, ok := generatedImage(it); ok {
			parts = append(parts, file)
		}
		return parts
	default:
		res.Output = ai.ToolOutput{JSON: raw}
	}
	return append(parts, res)
}

func providerToolName(it outputItem) string {
	switch it.Type {
	case "web_search_call":
		return "web_search"
	case "file_search_call":
		return "file_search"
	case "code_interpreter_call":
		return "code_interpreter"
	case "image_generation_call":
		return "image_generation"
	default:
		return "mcp." + it.Label + "." + it.Name
	}
}

func providerToolInput(it outputItem) json.RawMessage {
	switch it.Type {
	case "web_search_call":
		return cmpRaw(withoutKey(it.Action, "sources"), json.RawMessage("{}"))
	case "file_search_call":
		return pdata.JSON(providerName, map[string]json.RawMessage{"queries": nullable(it.Queries)})
	case "code_interpreter_call":
		return pdata.JSON(providerName, map[string]string{"code": it.Code, "container_id": it.ContID})
	case "mcp_call":
		return openaiwire.ToolInput(it.Arguments)
	default:
		return json.RawMessage("{}")
	}
}

func generatedImage(it outputItem) (ai.File, bool) {
	if it.Result == "" {
		return ai.File{}, false
	}
	data, err := base64.StdEncoding.DecodeString(it.Result)
	if err != nil {
		return ai.File{}, false
	}
	format := cmp.Or(it.Format, "png")
	return ai.File{MediaType: "image/" + format, Data: data, Provider: pdata.Encode(providerName, partData{ID: it.ID, Derived: true})}, true
}

func nullable(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("null")
	}
	return raw
}

func cmpRaw(a, b json.RawMessage) json.RawMessage {
	if len(a) > 0 && string(a) != "null" {
		return a
	}
	return b
}

func withoutKey(raw json.RawMessage, key string) json.RawMessage {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return raw
	}
	delete(fields, key)
	return pdata.JSON(providerName, fields)
}

func finishOf(status, incomplete string, c converted) (ai.FinishReason, string) {
	switch {
	case status == "incomplete":
		switch incomplete {
		case "max_output_tokens":
			return ai.FinishLength, incomplete
		case "content_filter":
			return ai.FinishContentFilter, incomplete
		default:
			return ai.FinishOther, cmp.Or(incomplete, status)
		}
	case status == "cancelled": //nolint:misspell // OpenAI status value
		return ai.FinishOther, status
	case c.clientCall:
		return ai.FinishToolCalls, cmp.Or(status, "completed")
	case c.refusal:
		return ai.FinishRefusal, "refusal"
	default:
		return ai.FinishStop, cmp.Or(status, "completed")
	}
}
