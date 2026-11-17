package replicate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	defaultWait = 60 * time.Second
	minCancel   = 5 * time.Second
)

var versionID = regexp.MustCompile(`^[0-9a-f]{64}$`)

type prediction struct {
	ID     string          `json:"id"`
	Status string          `json:"status"`
	Output json.RawMessage `json:"output"`
	Error  json.RawMessage `json:"error"`
}

type createBody struct {
	Version string         `json:"version,omitzero"`
	Input   map[string]any `json:"input"`
}

type createOptions struct {
	wait        *time.Duration
	cancelAfter time.Duration
	waitDefault time.Duration
}

func (p *Provider) create(ctx context.Context, model string, input map[string]any, o createOptions, header http.Header) (*prediction, error) {
	req := transport.Request{Header: header.Clone()}
	if req.Header == nil {
		req.Header = http.Header{}
	}
	body := createBody{Input: input}
	switch {
	case strings.Contains(model, ":") || versionID.MatchString(model):
		body.Version = model
		req.Path = "/predictions"
	default:
		owner, name, ok := strings.Cut(model, "/")
		if !ok || owner == "" || name == "" || strings.Contains(name, "/") {
			return nil, fmt.Errorf("replicate: model %q must be owner/name, owner/name:version or a version id", model)
		}
		req.Path = "/models/" + url.PathEscape(owner) + "/" + url.PathEscape(name) + "/predictions"
	}
	req.Body = body
	if wait := waitHeader(o); wait != "" {
		req.Header.Set("Prefer", wait)
	}
	if o.cancelAfter > 0 {
		req.Header.Set("Cancel-After", strconv.Itoa(int(max(o.cancelAfter, minCancel)/time.Second))+"s")
	}
	var pred prediction
	if _, err := p.client.JSON(ctx, req, &pred); err != nil {
		return nil, mediaio.FixError(err)
	}
	if pred.ID == "" {
		return nil, errors.New("replicate: prediction has no id")
	}
	return &pred, nil
}

func waitHeader(o createOptions) string {
	wait := o.waitDefault
	if o.wait != nil {
		wait = *o.wait
	}
	if wait <= 0 {
		return ""
	}
	return "wait=" + strconv.Itoa(int(min(max(wait, time.Second), defaultWait)/time.Second))
}

func (p *Provider) get(ctx context.Context, id string) (*prediction, error) {
	var pred prediction
	if _, err := p.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/predictions/" + url.PathEscape(id)}, &pred); err != nil {
		return nil, mediaio.FixError(err)
	}
	return &pred, nil
}

func (pr *prediction) terminal() bool {
	switch pr.Status {
	case "succeeded", "failed", "canceled", "aborted":
		return true
	default:
		return false
	}
}

func (pr *prediction) failure() error {
	message := errorText(pr.Error)
	switch pr.Status {
	case "succeeded":
		return nil
	case "canceled", "aborted":
		return mediaio.Failed(providerName, mediaio.CodeJobCanceled, "prediction "+pr.Status)
	case "failed":
		if mediaio.LooksModerated(message) {
			return mediaio.Blocked(providerName, message)
		}
		if message == "" {
			message = "prediction failed"
		}
		return mediaio.Failed(providerName, mediaio.CodeJobFailed, message)
	default:
		return fmt.Errorf("replicate: unexpected prediction status %q", pr.Status)
	}
}

func errorText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	return string(raw)
}

func (pr *prediction) outputURLs() ([]string, error) {
	var one string
	if json.Unmarshal(pr.Output, &one) == nil && one != "" {
		return []string{one}, nil
	}
	var many []string
	if json.Unmarshal(pr.Output, &many) == nil && len(many) > 0 {
		return many, nil
	}
	return nil, errors.New("replicate: prediction output has no file urls; its data may have been removed or the model does not return files")
}

func (p *Provider) files(ctx context.Context, pr *prediction, fallback string, limit int64, keepURL bool) ([]ai.File, error) {
	urls, err := pr.outputURLs()
	if err != nil {
		return nil, err
	}
	files := make([]ai.File, 0, len(urls))
	for _, u := range urls {
		f, err := p.dl.Result(ctx, u, fallback, limit, keepURL)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	return files, nil
}

func parseSize(size string) (width, height int, err error) {
	w, h, ok := strings.Cut(strings.ToLower(size), "x")
	width, wErr := strconv.Atoi(w)
	height, hErr := strconv.Atoi(h)
	if !ok || wErr != nil || hErr != nil || width <= 0 || height <= 0 {
		return 0, 0, fmt.Errorf("replicate: size %q must be WIDTHxHEIGHT", size)
	}
	return width, height, nil
}
