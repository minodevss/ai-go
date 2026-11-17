package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"
	"time"

	"github.com/minodevss/ai-go/internal/schema"
)

var errConsumerStopped = errors.New("ai: stream stopped by consumer")

type runner struct {
	model      LanguageModel
	req        TextRequest
	stream     bool
	emit       func(StreamPart) bool
	output     *outputSpec
	maxSteps   int
	maxRetries int
	validators map[string]*schema.Validator
	messages   []Message
	newMsgs    []Message
	steps      []Step
	usage      Usage
	warnings   []Warning
	finish     FinishReason
	content    Parts
	provider   ProviderData
	raw        string
	pending    bool
	client     bool
}

type stepOutput struct {
	content  Parts
	usage    Usage
	finish   FinishReason
	raw      string
	warnings []Warning
	provider ProviderData
	header   http.Header
}

func newRunner(model LanguageModel, req TextRequest, stream bool, emit func(StreamPart) bool) *runner {
	r := &runner{
		model:      model,
		req:        req,
		stream:     stream,
		emit:       emit,
		maxSteps:   req.MaxSteps,
		maxRetries: defaultMaxRetries,
		validators: map[string]*schema.Validator{},
	}
	if r.maxSteps <= 0 {
		r.maxSteps = defaultMaxSteps
	}
	if req.MaxRetries != nil {
		r.maxRetries = max(*req.MaxRetries, 0)
	}
	return r
}

func (r *runner) run(ctx context.Context) (*TextResult, error) {
	err := r.loop(ctx)
	res := r.result()
	if err != nil {
		res.FinishReason = FinishError
		return res, err
	}
	if !r.emit(StreamFinish{FinishReason: r.finish}) {
		return res, errConsumerStopped
	}
	return res, nil
}

func (r *runner) result() *TextResult {
	res := &TextResult{
		Text:         joinText(r.lastStepContent()),
		Content:      r.content,
		Messages:     slices.Clone(r.newMsgs),
		Steps:        slices.Clone(r.steps),
		Usage:        r.usage,
		FinishReason: r.finish,
		FinishRaw:    r.raw,
		Warnings:     slices.Clone(r.warnings),
		Provider:     r.provider,
		output:       r.req.Output,
	}
	if len(r.steps) == 0 {
		res.Text = joinText(r.content)
	}
	return res
}

func (r *runner) lastStepContent() Parts {
	if len(r.steps) == 0 {
		return nil
	}
	return r.steps[len(r.steps)-1].Content
}

func (r *runner) loop(ctx context.Context) error {
	if err := r.setup(); err != nil {
		return err
	}
	if !r.emit(StreamStart{}) {
		return errConsumerStopped
	}
	done, err := r.resolveApprovals(ctx)
	if err != nil || done {
		return err
	}
	for {
		stop, err := r.runStep(ctx)
		if err != nil || stop {
			return err
		}
	}
}

func (r *runner) setup() error {
	if err := checkTools(r.req.Tools); err != nil {
		return err
	}
	if r.req.Output != nil {
		spec, err := r.req.Output.spec()
		if err != nil {
			return err
		}
		r.output = &spec
	}
	if r.model == nil {
		return errors.New("ai: nil model")
	}
	r.messages = normalizeMessages(r.req.Messages)
	return nil
}

func checkTools(tools []Tool) error {
	seen := make(map[string]bool, len(tools))
	for _, t := range tools {
		if seen[t.Name] {
			return fmt.Errorf("%w: %q", ErrDuplicateTool, t.Name)
		}
		seen[t.Name] = true
	}
	return nil
}

func normalizeMessages(msgs []Message) []Message {
	out := cloneMessages(msgs)
	for i := range out {
		out[i].Content = normalizeParts(out[i].Content)
	}
	return out
}

func (r *runner) stepRequest() StepRequest {
	return StepRequest{
		Model:           r.model,
		Instructions:    slices.Clone(r.req.Instructions),
		Messages:        cloneMessages(r.messages),
		Tools:           slices.Clone(r.req.Tools),
		ToolChoice:      r.req.ToolChoice,
		MaxOutputTokens: r.req.MaxOutputTokens,
		Temperature:     clonePtr(r.req.Temperature),
		TopP:            clonePtr(r.req.TopP),
		Reasoning:       r.req.Reasoning,
		Headers:         r.req.Headers.Clone(),
		Provider:        r.req.Provider.clone(),
	}
}

func (r *runner) call(sr StepRequest) *Call {
	c := &Call{
		Instructions:     sr.Instructions,
		Messages:         providerMessages(sr.Messages),
		Tools:            sr.Tools,
		ToolChoice:       sr.ToolChoice,
		MaxOutputTokens:  sr.MaxOutputTokens,
		Temperature:      sr.Temperature,
		TopP:             sr.TopP,
		TopK:             r.req.TopK,
		PresencePenalty:  clonePtr(r.req.PresencePenalty),
		FrequencyPenalty: clonePtr(r.req.FrequencyPenalty),
		Seed:             clonePtr(r.req.Seed),
		StopSequences:    slices.Clone(r.req.StopSequences),
		Reasoning:        sr.Reasoning,
		Headers:          sr.Headers,
		Provider:         sr.Provider,
	}
	if r.output != nil {
		c.Output = r.output.call()
	}
	return c
}

func providerMessages(msgs []Message) []Message {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		parts := make(Parts, 0, len(m.Content))
		for _, p := range normalizeParts(m.Content) {
			switch p.(type) {
			case ToolApprovalRequest, ToolApprovalResponse:
			default:
				parts = append(parts, p)
			}
		}
		if len(parts) == 0 {
			continue
		}
		m.Content = parts
		out = append(out, m)
	}
	return out
}

func (r *runner) runStep(ctx context.Context) (bool, error) {
	number := len(r.steps)
	sr := r.stepRequest()
	if r.req.PrepareStep != nil {
		state := StepState{Number: number, Steps: slices.Clone(r.steps)}
		if err := r.req.PrepareStep(ctx, state, &sr); err != nil {
			return false, fmt.Errorf("ai: prepare step %d: %w", number, err)
		}
	}
	if sr.Model == nil {
		return false, fmt.Errorf("ai: prepare step %d: nil model", number)
	}
	if err := checkTools(sr.Tools); err != nil {
		return false, err
	}
	if !r.emit(StepStart{Number: number}) {
		return false, errConsumerStopped
	}
	info := StepInfo{Provider: sr.Model.Provider(), ModelID: sr.Model.ModelID(), Number: number, Request: &sr}
	stepCtx := ctx
	if r.req.Telemetry != nil {
		stepCtx = r.req.Telemetry.StartStep(ctx, info)
	}
	step, err := r.executeStep(stepCtx, sr, number)
	if r.req.Telemetry != nil {
		r.req.Telemetry.FinishStep(stepCtx, info, step, err)
	}
	if err != nil {
		return false, err
	}
	if r.req.OnStepEnd != nil {
		if err := r.req.OnStepEnd(ctx, step); err != nil {
			return false, fmt.Errorf("ai: step %d end hook: %w", number, err)
		}
	}
	return r.decide(step), nil
}

func (r *runner) executeStep(ctx context.Context, sr StepRequest, number int) (Step, error) {
	out, err := r.callModel(ctx, sr.Model, r.call(sr))
	if err != nil {
		return Step{}, err
	}
	r.usage = r.usage.add(out.usage)
	calls := toolCallsOf(out.content)
	h, toolErr := r.handleCalls(ctx, sr.Tools, localCalls(calls))
	if toolErr != nil {
		if h.ran == 0 {
			return Step{}, toolErr
		}
		for _, c := range h.unfinished {
			h.results = append(h.results, *errorResult(c, "tool call was canceled"))
		}
	}
	content := slices.Clone(out.content)
	for _, req := range h.requests {
		content = append(content, req)
	}
	finish := out.finish
	if len(h.requests) > 0 {
		finish = FinishApprovalPending
	}
	step := Step{
		Number:       number,
		Content:      content,
		ToolCalls:    calls,
		ToolResults:  h.results,
		Usage:        out.usage,
		FinishReason: finish,
		FinishRaw:    out.raw,
		Warnings:     out.warnings,
		Provider:     out.provider,
		Header:       out.header,
	}
	r.record(step, h)
	if toolErr != nil {
		_ = r.publishTools(h)
		return step, toolErr
	}
	if err := r.publishTools(h); err != nil {
		return step, err
	}
	if !r.emit(StepEnd{Usage: out.usage, FinishReason: finish, FinishRaw: out.raw, Warnings: out.warnings, Provider: out.provider}) {
		return step, errConsumerStopped
	}
	return step, nil
}

func (r *runner) publishTools(h handled) error {
	for _, f := range h.failures {
		if !r.emit(f) {
			return errConsumerStopped
		}
	}
	for _, res := range h.results {
		if !r.emit(res) {
			return errConsumerStopped
		}
	}
	for _, req := range h.requests {
		if !r.emit(req) {
			return errConsumerStopped
		}
	}
	return nil
}

func (r *runner) record(step Step, h handled) {
	r.steps = append(r.steps, step)
	r.warnings = append(r.warnings, step.Warnings...)
	r.provider = step.Provider
	r.raw = step.FinishRaw
	r.finish = step.FinishReason
	r.content = slices.Concat(step.Content, resultParts(step.ToolResults))
	if len(step.Content) > 0 {
		r.addMessage(Message{Role: RoleAssistant, Content: slices.Clone(step.Content), Provider: step.Provider.clone()})
	}
	if len(step.ToolResults) > 0 {
		r.addMessage(Message{Role: RoleTool, Content: resultParts(step.ToolResults)})
	}
	r.pending = len(h.requests) > 0
	r.client = h.client > 0
}

func (r *runner) addMessage(m Message) {
	r.messages = append(r.messages, m)
	r.newMsgs = append(r.newMsgs, m)
}

func resultParts(results []ToolResult) Parts {
	out := make(Parts, len(results))
	for i, res := range results {
		out[i] = res
	}
	return out
}

func toolCallsOf(content Parts) []ToolCall {
	var calls []ToolCall
	for _, p := range content {
		if c, ok := p.(ToolCall); ok {
			calls = append(calls, c)
		}
	}
	return calls
}

func localCalls(calls []ToolCall) []callItem {
	var items []callItem
	for _, c := range calls {
		if !c.ProviderExecuted {
			items = append(items, callItem{call: c})
		}
	}
	return items
}

func (r *runner) decide(step Step) bool {
	switch {
	case r.pending, r.client:
		return true
	case step.FinishReason == FinishPause:
	case len(step.ToolResults) == 0:
		return true
	case slices.Contains([]FinishReason{FinishLength, FinishContentFilter, FinishRefusal, FinishError}, step.FinishReason):
		return true
	}
	for _, cond := range r.req.StopWhen {
		if cond != nil && cond(r.steps) {
			return true
		}
	}
	if len(r.steps) >= r.maxSteps {
		r.finish = FinishStepLimit
		return true
	}
	return false
}

func (r *runner) callModel(ctx context.Context, model LanguageModel, call *Call) (stepOutput, error) {
	if d := r.req.Timeouts.Step; d > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, d, &timeoutError{msg: fmt.Sprintf("ai: step timed out after %s", d)})
		defer cancel()
	}
	committed := false
	canRetry := func(err error) bool { return !committed && isRetryable(err) }
	return withRetry(ctx, r.maxRetries, canRetry, func(ctx context.Context) (stepOutput, error) {
		if r.stream {
			return r.attemptStream(ctx, model, call, &committed)
		}
		return r.attemptGenerate(ctx, model, call)
	})
}

func (r *runner) attemptGenerate(ctx context.Context, model LanguageModel, call *Call) (stepOutput, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	resp, err := model.Generate(ctx, call)
	if err != nil {
		return stepOutput{}, mapTimeout(ctx, err)
	}
	if resp == nil {
		return stepOutput{}, errors.New("ai: model returned no response")
	}
	return stepOutput{
		content:  normalizeParts(resp.Content),
		usage:    resp.Usage,
		finish:   resp.FinishReason,
		raw:      resp.FinishRaw,
		warnings: resp.Warnings,
		provider: resp.Provider,
		header:   resp.Header,
	}, nil
}

type chunkTimer struct {
	cancel context.CancelCauseFunc
	timer  *time.Timer
}

func (t *chunkTimer) arm(d time.Duration, retry bool) {
	if d <= 0 {
		return
	}
	cause := &timeoutError{msg: fmt.Sprintf("ai: no data from model for %s", d), retry: retry}
	t.timer = time.AfterFunc(d, func() { t.cancel(cause) })
}

func (t *chunkTimer) disarm() {
	if t.timer != nil {
		t.timer.Stop()
		t.timer = nil
	}
}

func (r *runner) attemptStream(ctx context.Context, model LanguageModel, call *Call, committed *bool) (stepOutput, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	timer := &chunkTimer{cancel: cancel}
	defer timer.disarm()

	timer.arm(r.req.Timeouts.FirstChunk, true)
	resp, err := model.Stream(ctx, call)
	if err != nil {
		return stepOutput{}, mapTimeout(ctx, err)
	}
	if resp == nil || resp.Parts == nil {
		return stepOutput{}, errors.New("ai: model returned no stream")
	}
	next, stop := iter.Pull2(resp.Parts)
	defer stop()

	col := newCollector()
	for received := 0; ; received++ {
		if received > 0 {
			timer.arm(r.req.Timeouts.Chunk, false)
		}
		part, perr, ok := next()
		timer.disarm()
		if perr != nil {
			return stepOutput{}, mapTimeout(ctx, perr)
		}
		if !ok {
			return stepOutput{}, errors.New("ai: model stream ended without a finish part")
		}
		*committed = true
		col.add(part)
		if col.end != nil {
			break
		}
		if !r.forward(part) {
			return stepOutput{}, errConsumerStopped
		}
	}
	end := col.end
	return stepOutput{
		content:  col.parts(),
		usage:    end.Usage,
		finish:   end.FinishReason,
		raw:      end.FinishRaw,
		warnings: end.Warnings,
		provider: end.Provider,
		header:   resp.Header,
	}, nil
}

func (r *runner) forward(p StreamPart) bool {
	switch p.(type) {
	case StreamStart, StepStart, StreamFinish, ToolFailed:
		return true
	default:
		return r.emit(p)
	}
}
