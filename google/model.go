package google

import (
	"context"
	"net/http"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

// LanguageModel talks to generateContent and keeps every response part in Provider data for replay.
type LanguageModel struct {
	backend *backend
	id      string
}

func (m *LanguageModel) Provider() string { return m.backend.name }

func (m *LanguageModel) ModelID() string { return m.id }

func (m *LanguageModel) Generate(ctx context.Context, c *ai.Call) (*ai.Response, error) {
	req, warnings, err := m.request(c)
	if err != nil {
		return nil, err
	}
	var body responseBody
	header, err := m.backend.client.JSON(ctx, transport.Request{Path: m.backend.modelPath(m.id) + ":generateContent", Body: req, Header: c.Headers}, &body)
	if err != nil {
		return nil, fail(err)
	}
	return m.convert(body, warnings, header)
}

func (m *LanguageModel) convert(body responseBody, warnings []ai.Warning, header http.Header) (*ai.Response, error) {
	if err := body.failure(m.backend); err != nil {
		return nil, err
	}
	cand := body.first()
	parts, clientCall, err := m.backend.convertParts(cand.Content.Parts, body.ResponseID)
	if err != nil {
		return nil, err
	}
	var meta metadata
	meta.take(body, cand)
	parts = m.backend.withCitations(parts, meta.fields["groundingMetadata"])
	finish, raw := finishOf(cand.FinishReason, blockReason(meta.fields["promptFeedback"]), clientCall)
	return &ai.Response{
		Content:      parts,
		Usage:        usageOf(body.UsageMetadata),
		FinishReason: finish,
		FinishRaw:    raw,
		Warnings:     warnings,
		Provider:     meta.provider(m.backend),
		Header:       header,
	}, nil
}

func (b *backend) withCitations(parts ai.Parts, grounding []byte) ai.Parts {
	var indices []int
	var texts []string
	for i, p := range parts {
		if t, ok := p.(ai.Text); ok {
			indices = append(indices, i)
			texts = append(texts, t.Text)
		}
	}
	if len(texts) == 0 {
		return parts
	}
	for n, cites := range b.citations(texts, grounding) {
		if len(cites) == 0 {
			continue
		}
		t, ok := parts[indices[n]].(ai.Text)
		if !ok {
			continue
		}
		t.Citations = cites
		parts[indices[n]] = t
	}
	return parts
}

var _ ai.LanguageModel = (*LanguageModel)(nil)
