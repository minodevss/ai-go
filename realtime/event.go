package realtime

import (
	"encoding/json"
	"time"

	"github.com/minodevss/ai-go"
)

// Event is one of the types in this file. Providers produce every type except ToolResult and ApprovalRequest, which only the Session creates.
type Event interface{ isEvent() }

type AudioDelta struct {
	Data []byte
}

type TextDelta struct {
	Text string
}

// Transcript.Text is a delta while Final is false and the complete text once Final is true.
type Transcript struct {
	Role  ai.Role
	Text  string
	Final bool
}

type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

type ToolResult struct {
	CallID  string
	Name    string
	Output  ai.ToolOutput
	IsError bool
}

// ApprovalRequest asks the application to call Session.Approve with ApprovalID. Input is the exact input that runs if approved.
type ApprovalRequest struct {
	ApprovalID string
	CallID     string
	Name       string
	Input      json.RawMessage
	Reason     string
}

type Interrupted struct{}

// ToolCanceled reports that the provider withdrew a tool call; the Session has canceled its context.
type ToolCanceled struct {
	CallID string
}

type TurnEnd struct {
	Usage        ai.Usage
	FinishReason ai.FinishReason
}

// GoAway announces that the provider will end the session after the given time.
type GoAway struct {
	After time.Duration
}

// Resumable carries a handle for Config.Resume that continues the session after a reconnect.
type Resumable struct {
	Handle string
}

func (AudioDelta) isEvent()      {}
func (TextDelta) isEvent()       {}
func (Transcript) isEvent()      {}
func (ToolCall) isEvent()        {}
func (ToolResult) isEvent()      {}
func (ApprovalRequest) isEvent() {}
func (Interrupted) isEvent()     {}
func (ToolCanceled) isEvent()    {}
func (TurnEnd) isEvent()         {}
func (GoAway) isEvent()          {}
func (Resumable) isEvent()       {}
