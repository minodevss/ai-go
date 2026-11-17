package ai

type StreamPart interface{ isStreamPart() }

func (ToolCall) isStreamPart()            {}
func (ToolResult) isStreamPart()          {}
func (ToolApprovalRequest) isStreamPart() {}
func (File) isStreamPart()                {}
func (Citation) isStreamPart()            {}
func (Custom) isStreamPart()              {}

type StreamStart struct{}

type StepStart struct {
	Number int
}

type TextStart struct {
	ID string
}

type TextDelta struct {
	ID   string
	Text string
}

type TextEnd struct {
	ID       string
	Provider ProviderData
}

type ReasoningStart struct {
	ID string
}

type ReasoningDelta struct {
	ID   string
	Text string
}

type ReasoningEnd struct {
	ID       string
	Provider ProviderData
}

type ToolInputStart struct {
	ID   string
	Name string
}

type ToolInputDelta struct {
	ID   string
	Text string
}

type ToolInputEnd struct {
	ID string
}

// ToolFailed carries the real tool error for server-side observers; the model and the client only see the generic message.
type ToolFailed struct {
	CallID string
	Name   string
	Err    error
}

// StepEnd is the last part of a provider stream. The core re-emits it after the step's tools have run.
type StepEnd struct {
	Usage        Usage
	FinishReason FinishReason
	FinishRaw    string
	Warnings     []Warning
	Provider     ProviderData
}

type StreamFinish struct {
	FinishReason FinishReason
}

func (StreamStart) isStreamPart()    {}
func (StepStart) isStreamPart()      {}
func (TextStart) isStreamPart()      {}
func (TextDelta) isStreamPart()      {}
func (TextEnd) isStreamPart()        {}
func (ReasoningStart) isStreamPart() {}
func (ReasoningDelta) isStreamPart() {}
func (ReasoningEnd) isStreamPart()   {}
func (ToolInputStart) isStreamPart() {}
func (ToolInputDelta) isStreamPart() {}
func (ToolInputEnd) isStreamPart()   {}
func (ToolFailed) isStreamPart()     {}
func (StepEnd) isStreamPart()        {}
func (StreamFinish) isStreamPart()   {}
