package ui

import (
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
)

const maxReasonBytes = 1024

const DefaultMaxAge = 10 * time.Minute

type toMessagesOptions struct {
	keys   [][]byte
	maxAge time.Duration
	bind   binding
}

type ToMessagesOption func(*toMessagesOptions)

// VerifyKey makes ToMessages accept assistant history signed with any of the keys; list the newest first and keep old keys while their messages are still in use.
func VerifyKey(keys ...[]byte) ToMessagesOption {
	return func(o *toMessagesOptions) { o.keys = append(o.keys, keys...) }
}

// MaxAge rejects signed messages older than d. Stateless history cannot tell a replayed pending approval from a fresh one, so a short age bounds the replay window.
// The default is DefaultMaxAge: zero selects it, and a negative d disables the age check.
func MaxAge(d time.Duration) ToMessagesOption {
	return func(o *toMessagesOptions) {
		switch {
		case d < 0:
			o.maxAge = 0
		case d == 0:
			o.maxAge = DefaultMaxAge
		default:
			o.maxAge = d
		}
	}
}

// Chat accepts only history signed for this chat and owner, the values given as Options.ChatID and Options.Owner when the messages were streamed.
func Chat(chatID, owner string) ToMessagesOption {
	return func(o *toMessagesOptions) { o.bind = binding{ChatID: chatID, Owner: owner} }
}

// Merge returns stored plus what the client may add: the new user message, approval responses that match an approval request in stored, and results for client-side tool calls. Provider data is taken from stored only.
func Merge(stored []ai.Message, req Request, tools []ai.Tool) ([]ai.Message, error) {
	var assistants []Message
	for _, m := range req.Messages {
		if m.Role == RoleAssistant {
			assistants = append(assistants, m)
		}
	}
	history, err := applyClientParts(slices.Clone(stored), assistants, tools)
	if err != nil {
		return nil, err
	}
	return appendNewUser(history, req)
}

// ToMessages rebuilds the history of a stateless chat. User text and files are taken as sent. Assistant messages are accepted only when VerifyKey is given and their signature verifies; without a key they are dropped and approval responses are rejected.
func ToMessages(req Request, tools []ai.Tool, opts ...ToMessagesOption) ([]ai.Message, error) {
	o := toMessagesOptions{maxAge: DefaultMaxAge}
	for _, opt := range opts {
		opt(&o)
	}
	for _, key := range o.keys {
		if err := checkKey(key); err != nil {
			return nil, err
		}
	}
	var history []ai.Message
	var assistants []Message
	for _, m := range req.Messages {
		switch m.Role {
		case RoleUser:
			um, err := userToAI(m, req.files, false)
			if err != nil {
				return nil, err
			}
			if len(um.Content) > 0 {
				history = append(history, um)
			}
		case RoleAssistant:
			if len(o.keys) == 0 {
				if hasPendingResponse(m) {
					return nil, requestError(http.StatusBadRequest, ErrApproval, "approval responses need a verified history")
				}
				continue
			}
			group, err := verifyMessages(o.keys, o.bind, o.maxAge, m.Metadata)
			if err != nil {
				return nil, err
			}
			history = append(history, group...)
			assistants = append(assistants, m)
		case RoleSystem:
		}
	}
	return applyClientParts(history, assistants, tools)
}

func hasPendingResponse(m Message) bool {
	return slices.ContainsFunc(m.Parts, func(p Part) bool {
		t, ok := p.(ToolPart)
		return ok && t.State == ToolApprovalResponded
	})
}

func appendNewUser(history []ai.Message, req Request) ([]ai.Message, error) {
	if req.Trigger == TriggerRegenerate || len(req.Messages) == 0 {
		return history, nil
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != RoleUser {
		return history, nil
	}
	um, err := userToAI(last, req.files, false)
	if err != nil {
		return nil, err
	}
	if len(um.Content) == 0 {
		return nil, badRequest("the new user message has no text or file parts")
	}
	return append(history, um), nil
}

type position struct {
	index int
	call  ai.ToolCall
}

func applyClientParts(history []ai.Message, assistants []Message, tools []ai.Tool) ([]ai.Message, error) {
	inserts := map[int][]ai.Part{}
	answered := map[string]bool{}
	for _, m := range assistants {
		for _, p := range m.Parts {
			t, ok := p.(ToolPart)
			if !ok {
				continue
			}
			var part ai.Part
			var idx int
			var err error
			switch {
			case t.Approval != nil && t.Approval.Approved != nil:
				idx, part, err = approvalResponse(history, t, answered)
			case t.State == ToolOutputAvailable || t.State == ToolOutputError:
				idx, part = clientResult(history, t, tools)
			}
			if err != nil {
				return nil, err
			}
			if part != nil {
				inserts[idx] = append(inserts[idx], part)
			}
		}
	}
	if len(inserts) == 0 {
		return history, nil
	}
	out := make([]ai.Message, 0, len(history)+len(inserts))
	for i, m := range history {
		out = append(out, m)
		if parts := inserts[i]; len(parts) > 0 {
			out = append(out, ai.Message{Role: ai.RoleTool, Content: orderResponsesFirst(parts)})
		}
	}
	return out, nil
}

func orderResponsesFirst(parts []ai.Part) ai.Parts {
	out := slices.Clone(parts)
	slices.SortStableFunc(out, func(a, b ai.Part) int {
		_, aResp := a.(ai.ToolApprovalResponse)
		_, bResp := b.(ai.ToolApprovalResponse)
		switch {
		case aResp && !bResp:
			return -1
		case !aResp && bResp:
			return 1
		default:
			return 0
		}
	})
	return out
}

func approvalResponse(history []ai.Message, t ToolPart, answered map[string]bool) (int, ai.Part, error) {
	idx := -1
	for i, m := range history {
		if m.Role != ai.RoleAssistant {
			continue
		}
		for _, p := range m.Content {
			if req, ok := p.(ai.ToolApprovalRequest); ok && req.ApprovalID == t.Approval.ID && req.CallID == t.CallID {
				idx = i
			}
		}
	}
	if idx < 0 {
		return 0, nil, requestError(http.StatusBadRequest, ErrApproval, "no pending approval %q for call %q", t.Approval.ID, t.CallID)
	}
	if answered[t.Approval.ID] || resolvedAfter(history, idx, t.CallID, t.Approval.ID) {
		return 0, nil, nil
	}
	answered[t.Approval.ID] = true
	reason := strings.ToValidUTF8(t.Approval.Reason[:min(len(t.Approval.Reason), maxReasonBytes)], "")
	return idx, ai.ToolApprovalResponse{ApprovalID: t.Approval.ID, CallID: t.CallID, Approved: *t.Approval.Approved, Reason: reason}, nil
}

func resolvedAfter(history []ai.Message, from int, callID, approvalID string) bool {
	for _, m := range history[from:] {
		for _, p := range m.Content {
			switch v := p.(type) {
			case ai.ToolResult:
				if v.CallID == callID {
					return true
				}
			case ai.ToolApprovalResponse:
				if v.ApprovalID == approvalID {
					return true
				}
			}
		}
	}
	return false
}

func clientResult(history []ai.Message, t ToolPart, tools []ai.Tool) (int, ai.Part) {
	pos, ok := findCall(history, t.CallID)
	if !ok || pos.call.ProviderExecuted || t.Preliminary {
		return 0, nil
	}
	i := slices.IndexFunc(tools, func(tool ai.Tool) bool { return tool.Name == pos.call.Name })
	if i < 0 || tools[i].Execute != nil || resolvedAfter(history, pos.index, t.CallID, "") {
		return 0, nil
	}
	res := ai.ToolResult{CallID: t.CallID, Name: pos.call.Name, Output: t.Output}
	if t.State == ToolOutputError {
		res.IsError = true
		res.Output = ai.ToolOutput{Text: t.ErrorText}
	}
	return pos.index, res
}

func findCall(history []ai.Message, callID string) (position, bool) {
	for i, m := range slices.Backward(history) {
		if m.Role != ai.RoleAssistant {
			continue
		}
		for _, p := range m.Content {
			if c, ok := p.(ai.ToolCall); ok && c.ID == callID {
				return position{index: i, call: c}, true
			}
		}
	}
	return position{}, false
}
