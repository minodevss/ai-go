package luma

import (
	"context"
	"errors"
	"net/http"
	"net/url"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

type generation struct {
	ID            string `json:"id"`
	State         string `json:"state"`
	FailureReason string `json:"failure_reason"`
	Assets        struct {
		Video string `json:"video"`
		Image string `json:"image"`
	} `json:"assets"`
}

func (p *Provider) create(ctx context.Context, path string, body map[string]any, header http.Header) (*generation, error) {
	var g generation
	if _, err := p.client.JSON(ctx, transport.Request{Path: path, Body: body, Header: header}, &g); err != nil {
		return nil, mediaio.FixError(err)
	}
	if g.ID == "" {
		return nil, errors.New("luma: generation has no id")
	}
	return &g, nil
}

func (p *Provider) get(ctx context.Context, id string) (*generation, error) {
	var g generation
	if _, err := p.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/generations/" + url.PathEscape(id)}, &g); err != nil {
		return nil, mediaio.FixError(err)
	}
	return &g, nil
}

func (g *generation) terminal() bool { return g.State == "completed" || g.State == "failed" }

func (g *generation) failure() error {
	if g.State != "failed" {
		return nil
	}
	reason := g.FailureReason
	if reason == "" {
		reason = "generation failed"
	}
	if mediaio.LooksModerated(reason) {
		return mediaio.Blocked(providerName, reason)
	}
	return mediaio.Failed(providerName, mediaio.CodeJobFailed, reason)
}

func (p *Provider) file(ctx context.Context, assetURL, fallback string, limit int64, keepURL bool) (ai.File, error) {
	if assetURL == "" {
		return ai.File{}, errors.New("luma: completed generation has no asset url")
	}
	return p.dl.Result(ctx, assetURL, fallback, limit, keepURL)
}

func inputURL(f ai.File) (string, error) {
	if f.URL == "" {
		return "", errors.New("luma: input files must be URLs; Luma accepts no image bytes, so host the file and pass its URL")
	}
	return f.URL, nil
}
