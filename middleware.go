package ai

import (
	"context"
	"iter"
	"slices"
	"strconv"
	"strings"
)

// Middleware hooks run in order: TransformRequest on a clone of the call, then the Wrap functions around the model.
type Middleware struct {
	TransformRequest func(ctx context.Context, c *Call) error
	WrapGenerate     func(ctx context.Context, c *Call, next func(context.Context, *Call) (*Response, error)) (*Response, error)
	WrapStream       func(ctx context.Context, c *Call, next func(context.Context, *Call) (*StreamResponse, error)) (*StreamResponse, error)
}

type wrappedModel struct {
	inner LanguageModel
	mw    Middleware
}

func WrapModel(m LanguageModel, mw Middleware) LanguageModel {
	return &wrappedModel{inner: m, mw: mw}
}

func (w *wrappedModel) Provider() string { return w.inner.Provider() }

func (w *wrappedModel) ModelID() string { return w.inner.ModelID() }

func (w *wrappedModel) Unwrap() LanguageModel { return w.inner }

func (w *wrappedModel) Generate(ctx context.Context, c *Call) (*Response, error) {
	c = c.Clone()
	if w.mw.TransformRequest != nil {
		if err := w.mw.TransformRequest(ctx, c); err != nil {
			return nil, err
		}
	}
	if w.mw.WrapGenerate == nil {
		return w.inner.Generate(ctx, c)
	}
	return w.mw.WrapGenerate(ctx, c, w.inner.Generate)
}

func (w *wrappedModel) Stream(ctx context.Context, c *Call) (*StreamResponse, error) {
	c = c.Clone()
	if w.mw.TransformRequest != nil {
		if err := w.mw.TransformRequest(ctx, c); err != nil {
			return nil, err
		}
	}
	if w.mw.WrapStream == nil {
		return w.inner.Stream(ctx, c)
	}
	return w.mw.WrapStream(ctx, c, w.inner.Stream)
}

// As searches model and everything it wraps depth-first, following Unwrap() with LanguageModel, EmbeddingModel or ImageModel results and Unwrap() []LanguageModel, and returns the first match.
func As[T any](model any) (T, bool) {
	if v, ok := model.(T); ok {
		return v, true
	}
	switch u := model.(type) {
	case interface{ Unwrap() LanguageModel }:
		return As[T](u.Unwrap())
	case interface{ Unwrap() EmbeddingModel }:
		return As[T](u.Unwrap())
	case interface{ Unwrap() ImageModel }:
		return As[T](u.Unwrap())
	case interface{ Unwrap() []LanguageModel }:
		for _, child := range u.Unwrap() {
			if v, ok := As[T](child); ok {
				return v, true
			}
		}
	}
	return *new(T), false
}

func DefaultSettings(req TextRequest) Middleware {
	req = req.Clone()
	return Middleware{
		TransformRequest: func(_ context.Context, c *Call) error {
			if c.MaxOutputTokens == 0 {
				c.MaxOutputTokens = req.MaxOutputTokens
			}
			if c.Temperature == nil {
				c.Temperature = clonePtr(req.Temperature)
			}
			if c.TopP == nil {
				c.TopP = clonePtr(req.TopP)
			}
			if c.TopK == 0 {
				c.TopK = req.TopK
			}
			if c.PresencePenalty == nil {
				c.PresencePenalty = clonePtr(req.PresencePenalty)
			}
			if c.FrequencyPenalty == nil {
				c.FrequencyPenalty = clonePtr(req.FrequencyPenalty)
			}
			if c.Seed == nil {
				c.Seed = clonePtr(req.Seed)
			}
			if len(c.StopSequences) == 0 {
				c.StopSequences = slices.Clone(req.StopSequences)
			}
			if c.Reasoning == "" {
				c.Reasoning = req.Reasoning
			}
			c.Headers = mergeHeaders(c.Headers, req.Headers)
			c.Provider = c.Provider.merged(req.Provider)
			return nil
		},
	}
}

func ExtractReasoning(tag string) Middleware {
	open, closing := "<"+tag+">", "</"+tag+">"
	return Middleware{
		WrapGenerate: func(ctx context.Context, c *Call, next func(context.Context, *Call) (*Response, error)) (*Response, error) {
			resp, err := next(ctx, c)
			if err != nil {
				return nil, err
			}
			out := *resp
			out.Content = splitReasoning(resp.Content, open, closing)
			return &out, nil
		},
		WrapStream: func(ctx context.Context, c *Call, next func(context.Context, *Call) (*StreamResponse, error)) (*StreamResponse, error) {
			resp, err := next(ctx, c)
			if err != nil {
				return nil, err
			}
			out := *resp
			out.Parts = extractStream(resp.Parts, open, closing)
			return &out, nil
		},
	}
}

func splitReasoning(content Parts, open, closing string) Parts {
	out := make(Parts, 0, len(content))
	for _, p := range content {
		t, ok := p.(Text)
		if !ok {
			out = append(out, p)
			continue
		}
		rest := t.Text
		for {
			before, afterOpen, found := strings.Cut(rest, open)
			if !found {
				break
			}
			thought, afterClose, closed := strings.Cut(afterOpen, closing)
			if !closed {
				thought, afterClose = afterOpen, ""
			}
			if before != "" {
				out = append(out, Text{Text: before})
			}
			out = append(out, Reasoning{Text: thought})
			rest = afterClose
		}
		if rest != "" || len(out) == 0 || len(t.Provider) > 0 || len(t.Citations) > 0 {
			t.Text = rest
			out = append(out, t)
		}
	}
	return out
}

type reasoningSplitter struct {
	open, closing string
	id            string
	seq           int
	inside        bool
	opened        bool
	pending       string
}

func extractStream(src iter.Seq2[StreamPart, error], open, closing string) iter.Seq2[StreamPart, error] {
	return func(yield func(StreamPart, error) bool) {
		sp := &reasoningSplitter{open: open, closing: closing}
		for part, err := range src {
			if err != nil {
				yield(nil, err)
				return
			}
			var out []StreamPart
			switch v := part.(type) {
			case TextStart:
				sp.reset(v.ID)
			case TextDelta:
				out = sp.feed(v.Text)
			case TextEnd:
				out = sp.finish(v)
			default:
				out = []StreamPart{part}
			}
			for _, o := range out {
				if !yield(o, nil) {
					return
				}
			}
		}
	}
}

func (s *reasoningSplitter) reset(id string) {
	*s = reasoningSplitter{open: s.open, closing: s.closing, id: id}
}

func (s *reasoningSplitter) blockID() string { return s.id + "#" + strconv.Itoa(s.seq) }

func (s *reasoningSplitter) marker() string {
	if s.inside {
		return s.closing
	}
	return s.open
}

func (s *reasoningSplitter) feed(text string) []StreamPart {
	buf := s.pending + text
	s.pending = ""
	var out []StreamPart
	for {
		marker := s.marker()
		i := strings.Index(buf, marker)
		if i < 0 {
			keep := partialSuffix(buf, marker)
			out = s.emit(out, buf[:len(buf)-keep])
			s.pending = buf[len(buf)-keep:]
			return out
		}
		out = s.emit(out, buf[:i])
		out = s.closeBlock(out, nil)
		s.inside = !s.inside
		s.seq++
		buf = buf[i+len(marker):]
	}
}

func (s *reasoningSplitter) finish(end TextEnd) []StreamPart {
	out := s.emit(nil, s.pending)
	s.pending = ""
	if !s.opened && !s.inside && len(end.Provider) > 0 {
		s.opened = true
		out = append(out, TextStart{ID: s.blockID()})
	}
	return s.closeBlock(out, end.Provider)
}

func (s *reasoningSplitter) emit(out []StreamPart, text string) []StreamPart {
	if text == "" {
		return out
	}
	if !s.opened {
		s.opened = true
		if s.inside {
			out = append(out, ReasoningStart{ID: s.blockID()})
		} else {
			out = append(out, TextStart{ID: s.blockID()})
		}
	}
	if s.inside {
		return append(out, ReasoningDelta{ID: s.blockID(), Text: text})
	}
	return append(out, TextDelta{ID: s.blockID(), Text: text})
}

func (s *reasoningSplitter) closeBlock(out []StreamPart, provider ProviderData) []StreamPart {
	if !s.opened {
		return out
	}
	s.opened = false
	if s.inside {
		return append(out, ReasoningEnd{ID: s.blockID()})
	}
	return append(out, TextEnd{ID: s.blockID(), Provider: provider})
}

func partialSuffix(buf, marker string) int {
	for n := min(len(marker)-1, len(buf)); n > 0; n-- {
		if strings.HasSuffix(buf, marker[:n]) {
			return n
		}
	}
	return 0
}
