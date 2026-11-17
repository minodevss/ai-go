package ui

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"

	"github.com/minodevss/ai-go"
)

// Expose opts in to sending data the client never receives by default.
type Expose struct {
	Provider bool
	Usage    bool
	UsageRaw bool
	Warnings bool
}

// Options configure Respond, Stream and RespondText. The zero value streams to the request without signing or resume.
type Options struct {
	// OnFinish runs once, after the last step and before the finish chunk is written, with a context that survives client disconnects.
	OnFinish func(ctx context.Context, f Finish) error
	// OnError maps an error to the text sent to the client; the default is a generic message. It is also called for errors before streaming starts, so applications can log them.
	OnError func(err error) string
	// Key signs every assistant message so a stateless chat can verify it with VerifyKey.
	Key []byte
	// ChatID and Owner, when set, are signed into every assistant message; ToMessages accepts them only with the same Chat values.
	ChatID string
	Owner  string
	// History is the history the stream continues; when it does not end in a user message, the stream continues the last assistant message.
	History []ai.Message
	Expose  Expose
	Resume  *Resume
	// NewID creates message ids; the default is random.
	NewID func() string
}

// Finish describes a completed response. New holds completed steps only; after an abort, calls without results get synthetic error results.
type Finish struct {
	New          []ai.Message
	UI           Message
	Aborted      bool
	FinishReason ai.FinishReason
	Err          error
}

type DataOption func(*dataOptions)

type dataOptions struct {
	id        string
	transient bool
}

// Transient data parts reach the client's onData callback but are not stored in the message.
func Transient() DataOption { return func(o *dataOptions) { o.transient = true } }

// DataID lets a later data part with the same name and id replace this one.
func DataID(id string) DataOption { return func(o *dataOptions) { o.id = id } }

type sink interface {
	write(frame []byte) error
	end(final [][]byte) error
}

type options struct {
	Options
	newID func() string
}

func (o Options) prepare() (options, error) {
	out := options{Options: o, newID: o.NewID}
	if out.newID == nil {
		out.newID = rand.Text
	}
	if out.OnError == nil {
		out.OnError = func(error) string { return defaultErrText }
	}
	if o.Key != nil {
		if err := checkKey(o.Key); err != nil {
			return out, err
		}
	}
	if o.Resume != nil {
		if err := o.Resume.validate(); err != nil {
			return out, err
		}
	}
	return out, nil
}

// Respond runs factory with a context that ends with the request, or with one that outlives it when opts.Resume is set; always build the stream from that context. Failures reach the client as an error chunk and through opts.OnError.
func Respond(w http.ResponseWriter, r *http.Request, factory func(ctx context.Context) *ai.TextStream, opts Options) {
	Stream(w, r, func(ctx context.Context, uw *Writer) error {
		s := factory(ctx)
		if s == nil {
			return errors.New("ui: factory returned no stream")
		}
		return uw.Merge(s)
	}, opts)
}

// Stream is Respond for custom streams: fn writes data parts and merges any number of text streams through the Writer.
func Stream(w http.ResponseWriter, r *http.Request, fn func(ctx context.Context, uw *Writer) error, opts Options) {
	o, err := opts.prepare()
	if err != nil {
		failEarly(w, o, err)
		return
	}
	if o.Resume != nil {
		streamResumable(w, r, fn, o)
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	setStreamHeaders(w)
	w.WriteHeader(http.StatusOK)
	out := &httpSink{w: w, rc: http.NewResponseController(w)}
	if err := out.flush(); err != nil {
		return
	}
	produce(ctx, cancel, out, fn, o)
}

func failEarly(w http.ResponseWriter, o options, err error) {
	text := http.StatusText(http.StatusInternalServerError)
	if o.OnError != nil {
		text = o.OnError(err)
	}
	http.Error(w, text, http.StatusInternalServerError)
}

func setStreamHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set(streamHeader, streamVersion)
	h.Set("X-Accel-Buffering", "no")
}

type httpSink struct {
	w  http.ResponseWriter
	rc *http.ResponseController
}

func (s *httpSink) write(frame []byte) error {
	if _, err := s.w.Write(frame); err != nil {
		return err
	}
	return s.flush()
}

func (s *httpSink) flush() error {
	if err := s.rc.Flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return err
	}
	return nil
}

func (s *httpSink) end(final [][]byte) error {
	for _, f := range final {
		if err := s.write(f); err != nil {
			return err
		}
	}
	return nil
}

var errWriterClosed = errors.New("ui: writer used after the response ended")

type panicError struct{ value any }

func (e *panicError) Error() string { return fmt.Sprintf("ui: panic: %v", e.value) }

func produce(ctx context.Context, cancel context.CancelFunc, out sink, fn func(context.Context, *Writer) error, o options) {
	uw := &Writer{out: out, enc: newEncoder(o.Expose, o.OnError, o.History), cancel: cancel, messageID: o.newID()}
	prior := continuing(o.History)
	if len(prior) > 0 {
		uw.messageID = ""
	}
	uw.close(ctx, safely(ctx, uw, fn), o, prior)
}

func safely(ctx context.Context, uw *Writer, fn func(context.Context, *Writer) error) (err error) {
	defer func() {
		if v := recover(); v != nil {
			err = &panicError{value: v}
		}
	}()
	return fn(ctx, uw)
}

func continuing(history []ai.Message) []ai.Message {
	last := -1
	for i, m := range history {
		if m.Role == ai.RoleUser {
			last = i
		}
	}
	if last == len(history)-1 {
		return nil
	}
	return history[last+1:]
}

// Writer sends chunks of one assistant message. It is safe for concurrent use.
type Writer struct {
	mu        sync.Mutex
	out       sink
	enc       *encoder
	cancel    context.CancelFunc
	messageID string
	started   bool
	closed    bool
	sinkErr   error
	newMsgs   []ai.Message
	usage     ai.Usage
	warnings  []ai.Warning
	data      []DataPart
}

func (w *Writer) Data(name string, v any, opts ...DataOption) error {
	var o dataOptions
	for _, opt := range opts {
		opt(&o)
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("ui: encode data %q: %w", name, err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if !o.transient {
		w.recordData(DataPart{Name: name, ID: o.id, Data: raw})
	}
	return w.emitLocked(dataChunk{Type: "data-" + name, ID: o.id, Data: raw, Transient: o.transient})
}

func (w *Writer) recordData(p DataPart) {
	if p.ID != "" {
		if i := slices.IndexFunc(w.data, func(d DataPart) bool { return d.Name == p.Name && d.ID == p.ID }); i >= 0 {
			w.data[i] = p
			return
		}
	}
	w.data = append(w.data, p)
}

// Merge writes every part of s and records its completed steps for OnFinish. It returns the stream's error.
func (w *Writer) Merge(s *ai.TextStream) error {
	for p := range s.Parts() {
		if err := w.part(p); err != nil {
			break
		}
	}
	res, err := s.Result()
	w.mu.Lock()
	defer w.mu.Unlock()
	if res != nil {
		w.newMsgs = append(w.newMsgs, res.Messages...)
		w.usage = addUsage(w.usage, res.Usage)
		w.warnings = append(w.warnings, res.Warnings...)
	}
	if w.sinkErr != nil {
		return w.sinkErr
	}
	return err
}

func (w *Writer) part(p ai.StreamPart) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	chunks := w.enc.encode(p)
	if len(chunks) == 0 {
		return w.sinkErr
	}
	vs := make([]any, len(chunks))
	for i, c := range chunks {
		vs[i] = c
	}
	return w.emitLocked(vs...)
}

func (w *Writer) emitLocked(vs ...any) error {
	if w.closed {
		return errWriterClosed
	}
	if w.sinkErr != nil {
		return w.sinkErr
	}
	if !w.started {
		w.started = true
		vs = append([]any{chunk{Type: "start", MessageID: w.messageID}}, vs...)
	}
	for _, v := range vs {
		if err := w.out.write(frame(v)); err != nil {
			w.sinkErr = err
			w.cancel()
			return err
		}
	}
	return nil
}

func addUsage(a, b ai.Usage) ai.Usage {
	return ai.Usage{
		Input: a.Input + b.Input, Output: a.Output + b.Output,
		CacheRead: a.CacheRead + b.CacheRead, CacheWrite: a.CacheWrite + b.CacheWrite,
		Reasoning: a.Reasoning + b.Reasoning, Raw: b.Raw,
	}
}

func (w *Writer) close(ctx context.Context, runErr error, o options, prior []ai.Message) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closed = true
	aborted := ctx.Err() != nil || w.sinkErr != nil
	var tail [][]byte
	if !w.started {
		w.started = true
		tail = append(tail, frame(chunk{Type: "start", MessageID: w.messageID}))
	}
	if errors.Is(w.sinkErr, ErrStreamTooLarge) {
		tail = append(tail, frame(chunk{Type: "error", ErrorText: o.OnError(w.sinkErr)}))
	}
	newMsgs := w.newMsgs
	if aborted {
		newMsgs = closeOrphans(newMsgs)
	}
	f := Finish{New: newMsgs, Aborted: aborted, FinishReason: w.enc.finishReason}
	if runErr != nil && !aborted {
		f.Err = runErr
		f.FinishReason = ai.FinishError
		tail = append(tail, frame(chunk{Type: "error", ErrorText: o.OnError(runErr)}))
	}
	meta, err := w.finishMetadata(o, slices.Concat(prior, newMsgs))
	f.UI = w.uiMessage(slices.Concat(prior, newMsgs), meta)
	if err != nil {
		tail = append(tail, frame(chunk{Type: "error", ErrorText: o.OnError(err)}))
	}
	if o.OnFinish != nil {
		if err := o.OnFinish(context.WithoutCancel(ctx), f); err != nil {
			tail = append(tail, frame(chunk{Type: "error", ErrorText: o.OnError(err)}))
		}
	}
	tail = append(tail, frame(chunk{Type: "finish", FinishReason: uiFinishReason(f.FinishReason, aborted), MessageMetadata: meta}), doneFrame())
	if err := w.out.end(tail); err != nil {
		w.sinkErr = err
	}
}

func closeOrphans(msgs []ai.Message) []ai.Message {
	answered := map[string]bool{}
	for _, m := range msgs {
		for _, p := range m.Content {
			switch v := p.(type) {
			case ai.ToolResult:
				answered[v.CallID] = true
			case ai.ToolApprovalRequest:
				answered[v.CallID] = true
			}
		}
	}
	var results ai.Parts
	for _, m := range msgs {
		for _, p := range m.Content {
			if c, ok := p.(ai.ToolCall); ok && !c.ProviderExecuted && !answered[c.ID] {
				results = append(results, ai.ToolResult{CallID: c.ID, Name: c.Name, Output: ai.ToolOutput{Text: "tool call was interrupted"}, IsError: true})
			}
		}
	}
	if len(results) == 0 {
		return msgs
	}
	return append(slices.Clone(msgs), ai.Message{Role: ai.RoleTool, Content: results})
}

func (w *Writer) finishMetadata(o options, group []ai.Message) (json.RawMessage, error) {
	meta := map[string]any{}
	if o.Expose.Usage {
		usage := map[string]any{"input": w.usage.Input, "output": w.usage.Output, "cacheRead": w.usage.CacheRead, "cacheWrite": w.usage.CacheWrite, "reasoning": w.usage.Reasoning}
		if o.Expose.UsageRaw && len(w.usage.Raw) > 0 {
			usage["raw"] = w.usage.Raw
		}
		meta["usage"] = usage
	}
	if o.Expose.Warnings && len(w.warnings) > 0 {
		meta["warnings"] = w.warnings
	}
	var signErr error
	if o.Key != nil {
		sig, err := signMessages(o.Key, binding{ChatID: o.ChatID, Owner: o.Owner}, group, o.Expose.Provider)
		if err != nil {
			signErr = fmt.Errorf("ui: sign message: %w", err)
		} else {
			meta[metadataKey] = sig
		}
	}
	if len(meta) == 0 {
		return nil, signErr
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		return nil, errors.Join(signErr, err)
	}
	return raw, signErr
}

func (w *Writer) uiMessage(group []ai.Message, meta json.RawMessage) Message {
	msgs := FromMessages(group)
	m := Message{ID: w.messageID, Role: RoleAssistant, Metadata: meta}
	if len(msgs) > 0 {
		m.Parts = msgs[len(msgs)-1].Parts
		if msgs[len(msgs)-1].Role != RoleAssistant {
			m.Parts = nil
		}
	}
	for _, d := range w.data {
		m.Parts = append(m.Parts, d)
	}
	return m
}

// RespondText streams only the text deltas as a plain text response, for the AI SDK's TextStreamChatTransport. It ignores Key, Resume and Expose.
func RespondText(w http.ResponseWriter, r *http.Request, factory func(ctx context.Context) *ai.TextStream, opts Options) {
	o, err := Options{OnFinish: opts.OnFinish, OnError: opts.OnError}.prepare()
	if err != nil {
		failEarly(w, o, err)
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	out := &httpSink{w: w, rc: http.NewResponseController(w)}
	s := factory(ctx)
	if s == nil {
		return
	}
	var writeErr error
	for text := range s.Text() {
		if writeErr = out.write([]byte(text)); writeErr != nil {
			cancel()
			break
		}
	}
	res, err := s.Result()
	if o.OnFinish == nil || res == nil {
		return
	}
	aborted := ctx.Err() != nil || writeErr != nil
	msgs := res.Messages
	if aborted {
		msgs = closeOrphans(msgs)
	}
	f := Finish{New: msgs, Aborted: aborted, FinishReason: res.FinishReason, UI: Message{Role: RoleAssistant}}
	if all := FromMessages(msgs); len(all) > 0 {
		f.UI.Parts = all[len(all)-1].Parts
	}
	if err != nil && !aborted {
		f.Err = err
	}
	if err := o.OnFinish(context.WithoutCancel(ctx), f); err != nil {
		o.OnError(err)
	}
}
