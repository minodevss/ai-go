package ui

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strconv"

	"github.com/minodevss/ai-go"
)

type chunk struct {
	Type            string          `json:"type"`
	ID              string          `json:"id,omitzero"`
	Delta           string          `json:"delta,omitzero"`
	ToolCallID      string          `json:"toolCallId,omitzero"`
	ToolName        string          `json:"toolName,omitzero"`
	InputTextDelta  string          `json:"inputTextDelta,omitzero"`
	Input           json.RawMessage `json:"input,omitzero"`
	Output          json.RawMessage `json:"output,omitzero"`
	ProviderExec    bool            `json:"providerExecuted,omitzero"`
	ApprovalID      string          `json:"approvalId,omitzero"`
	Reason          string          `json:"reason,omitzero"`
	ErrorText       string          `json:"errorText,omitzero"`
	SourceID        string          `json:"sourceId,omitzero"`
	URL             string          `json:"url,omitzero"`
	Title           *string         `json:"title,omitzero"`
	MediaType       string          `json:"mediaType,omitzero"`
	Filename        string          `json:"filename,omitzero"`
	Citations       []citationWire  `json:"citations,omitzero"`
	Provider        ai.ProviderData `json:"providerMetadata,omitzero"`
	MessageID       string          `json:"messageId,omitzero"`
	FinishReason    string          `json:"finishReason,omitzero"`
	MessageMetadata json.RawMessage `json:"messageMetadata,omitzero"`
}

type dataChunk struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitzero"`
	Data      json.RawMessage `json:"data"`
	Transient bool            `json:"transient,omitzero"`
}

func frame(v any) []byte {
	var buf bytes.Buffer
	buf.WriteString("data: ")
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		buf.Reset()
		buf.WriteString(`data: {"type":"error","errorText":"` + defaultErrText + `"}`)
		buf.WriteString("\n")
	}
	buf.WriteString("\n")
	return buf.Bytes()
}

func doneFrame() []byte { return []byte("data: [DONE]\n\n") }

type encoder struct {
	expose       Expose
	onError      func(error) string
	failures     map[string]error
	denied       map[string]bool
	cites        []ai.Citation
	textOpen     bool
	sources      int
	finishReason ai.FinishReason
}

func newEncoder(expose Expose, onError func(error) string, history []ai.Message) *encoder {
	e := &encoder{expose: expose, onError: onError, failures: map[string]error{}, denied: map[string]bool{}}
	for _, m := range history {
		for _, p := range m.Content {
			if r, ok := p.(ai.ToolApprovalResponse); ok && !r.Approved {
				e.denied[r.CallID] = true
			}
		}
	}
	return e
}

func (e *encoder) metadata(d ai.ProviderData) ai.ProviderData {
	if e.expose.Provider {
		return d
	}
	return nil
}

func (e *encoder) encode(p ai.StreamPart) []chunk {
	switch v := p.(type) {
	case ai.StepStart:
		return []chunk{{Type: "start-step"}}
	case ai.StepEnd:
		return []chunk{{Type: "finish-step"}}
	case ai.StreamFinish:
		e.finishReason = v.FinishReason
	case ai.TextStart:
		e.textOpen = true
		return []chunk{{Type: "text-start", ID: v.ID}}
	case ai.TextDelta:
		if v.Text != "" {
			return []chunk{{Type: "text-delta", ID: v.ID, Delta: v.Text}}
		}
	case ai.TextEnd:
		e.textOpen = false
		cites := citationsToWire(e.cites)
		if len(e.cites) == 0 {
			cites = nil
		}
		e.cites = nil
		return []chunk{{Type: "text-end", ID: v.ID, Citations: cites, Provider: e.metadata(v.Provider)}}
	case ai.ReasoningStart:
		return []chunk{{Type: "reasoning-start", ID: v.ID}}
	case ai.ReasoningDelta:
		if v.Text != "" {
			return []chunk{{Type: "reasoning-delta", ID: v.ID, Delta: v.Text}}
		}
	case ai.ReasoningEnd:
		return []chunk{{Type: "reasoning-end", ID: v.ID, Provider: e.metadata(v.Provider)}}
	case ai.ToolInputStart:
		return []chunk{{Type: "tool-input-start", ToolCallID: v.ID, ToolName: v.Name}}
	case ai.ToolInputDelta:
		if v.Text != "" {
			return []chunk{{Type: "tool-input-delta", ToolCallID: v.ID, InputTextDelta: v.Text}}
		}
	case ai.ToolCall:
		return []chunk{e.toolCall(v)}
	case ai.ToolResult:
		return []chunk{e.toolResult(v)}
	case ai.ToolApprovalRequest:
		return []chunk{{Type: "tool-approval-request", ApprovalID: v.ApprovalID, ToolCallID: v.CallID, Reason: v.Reason}}
	case ai.ToolFailed:
		e.failures[v.CallID] = v.Err
	case ai.File:
		f := fileToPart(v)
		return []chunk{{Type: "file", URL: f.URL, MediaType: f.MediaType, Provider: e.metadata(v.Provider)}}
	case ai.Citation:
		return e.citation(v)
	default:
	}
	return nil
}

func (e *encoder) toolCall(c ai.ToolCall) chunk {
	input := c.Input
	if len(bytes.TrimSpace(input)) == 0 {
		input = json.RawMessage("{}")
	}
	if !json.Valid(input) {
		raw, _ := json.Marshal(string(c.Input))
		return chunk{Type: "tool-input-error", ToolCallID: c.ID, ToolName: c.Name, Input: raw, ProviderExec: c.ProviderExecuted, ErrorText: e.onError(errors.New("ui: tool input is not valid JSON")), Provider: e.metadata(c.Provider)}
	}
	return chunk{Type: "tool-input-available", ToolCallID: c.ID, ToolName: c.Name, Input: slices.Clone(input), ProviderExec: c.ProviderExecuted, Provider: e.metadata(c.Provider)}
}

func (e *encoder) toolResult(r ai.ToolResult) chunk {
	if !r.IsError {
		out, err := outputToWire(r.Output)
		if err != nil {
			return chunk{Type: "tool-output-error", ToolCallID: r.CallID, ProviderExec: r.ProviderExecuted, ErrorText: e.onError(err)}
		}
		return chunk{Type: "tool-output-available", ToolCallID: r.CallID, Output: out, ProviderExec: r.ProviderExecuted, Provider: e.metadata(r.Provider)}
	}
	if e.denied[r.CallID] {
		return chunk{Type: "tool-output-denied", ToolCallID: r.CallID}
	}
	text := outputText(r.Output)
	if cause, failed := e.failures[r.CallID]; failed {
		text = e.onError(cause)
	} else if r.ProviderExecuted {
		text = e.onError(errors.New(text))
	}
	return chunk{Type: "tool-output-error", ToolCallID: r.CallID, ProviderExec: r.ProviderExecuted, ErrorText: text, Provider: e.metadata(r.Provider)}
}

func (e *encoder) citation(c ai.Citation) []chunk {
	if e.textOpen {
		e.cites = append(e.cites, c)
		return nil
	}
	e.sources++
	id := "source-" + strconv.Itoa(e.sources)
	if c.URL != "" {
		out := chunk{Type: "source-url", SourceID: id, URL: c.URL, Provider: e.metadata(c.Provider)}
		if c.Title != "" {
			out.Title = &c.Title
		}
		return []chunk{out}
	}
	return []chunk{{Type: "source-document", SourceID: id, MediaType: "text/plain", Title: &c.Title, Provider: e.metadata(c.Provider)}}
}

func uiFinishReason(r ai.FinishReason, aborted bool) string {
	if aborted {
		return "interrupted"
	}
	switch r {
	case ai.FinishStop:
		return "stop"
	case ai.FinishLength:
		return "length"
	case ai.FinishToolCalls, ai.FinishStepLimit, ai.FinishApprovalPending:
		return "tool-calls"
	case ai.FinishContentFilter, ai.FinishRefusal:
		return "content-filter"
	case ai.FinishError:
		return "error"
	case ai.FinishOther, ai.FinishPause:
		return "other"
	default:
		return ""
	}
}
