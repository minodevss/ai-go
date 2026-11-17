package fal

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/retry"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	statusQueued     = "IN_QUEUE"
	statusInProgress = "IN_PROGRESS"
	statusCompleted  = "COMPLETED"
	maxResultBytes   = 64 << 20
)

type ticket struct {
	RequestID   string `json:"requestId"`
	ResponseURL string `json:"responseUrl"`
	StatusURL   string `json:"statusUrl"`
	CancelURL   string `json:"cancelUrl"`
	KeepURL     bool   `json:"keepUrl,omitzero"`
}

type submitResponse struct {
	RequestID   string `json:"request_id"`
	ResponseURL string `json:"response_url"`
	StatusURL   string `json:"status_url"`
	CancelURL   string `json:"cancel_url"`
}

type statusResponse struct {
	Status        string `json:"status"`
	QueuePosition int    `json:"queue_position"`
	Error         string `json:"error"`
	ErrorType     string `json:"error_type"`
}

type requestHeaders struct {
	Priority     string
	StartTimeout int
	NoRetry      bool
}

func (h requestHeaders) header(base http.Header) http.Header {
	out := base.Clone()
	if out == nil {
		out = http.Header{}
	}
	if h.Priority != "" {
		out.Set("X-Fal-Queue-Priority", h.Priority)
	}
	if h.StartTimeout > 0 {
		out.Set("X-Fal-Request-Start-Timeout", strconv.Itoa(h.StartTimeout))
	}
	if h.NoRetry {
		out.Set("X-Fal-No-Retry", "1")
	}
	return out
}

func (p *Provider) submit(ctx context.Context, modelID string, body map[string]any, header http.Header) (ticket, error) {
	var resp submitResponse
	_, err := p.queue.JSON(ctx, transport.Request{Path: "/" + escapeModel(modelID), Body: body, Header: header}, &resp)
	if err != nil {
		return ticket{}, err
	}
	t := ticket{RequestID: resp.RequestID, ResponseURL: resp.ResponseURL, StatusURL: resp.StatusURL, CancelURL: resp.CancelURL}
	if t.RequestID == "" || t.StatusURL == "" || t.ResponseURL == "" {
		return ticket{}, errors.New("fal: queue response has no request id or urls")
	}
	for _, u := range []string{t.StatusURL, t.ResponseURL, t.CancelURL} {
		if u == "" {
			continue
		}
		if _, err := p.queuePath(u); err != nil {
			return ticket{}, err
		}
	}
	return t, nil
}

func (p *Provider) status(ctx context.Context, t ticket) (statusResponse, error) {
	var st statusResponse
	path, err := p.queuePath(t.StatusURL)
	if err != nil {
		return st, err
	}
	_, err = p.queue.JSON(ctx, transport.Request{Method: http.MethodGet, Path: path}, &st)
	return st, err
}

func (p *Provider) result(ctx context.Context, t ticket) (json.RawMessage, error) {
	path, err := p.queuePath(t.ResponseURL)
	if err != nil {
		return nil, err
	}
	resp, err := p.queue.Do(ctx, transport.Request{Method: http.MethodGet, Path: path})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResultBytes))
	if err != nil {
		return nil, fmt.Errorf("fal: read result: %w", err)
	}
	return raw, nil
}

func (p *Provider) cancel(ctx context.Context, t ticket) error {
	if t.CancelURL == "" {
		return nil
	}
	path, err := p.queuePath(t.CancelURL)
	if err != nil {
		return err
	}
	resp, err := p.queue.Do(ctx, transport.Request{Method: http.MethodPut, Path: path})
	if err != nil {
		return err
	}
	return resp.Body.Close()
}

func (p *Provider) queuePath(raw string) (string, error) {
	u, err := url.Parse(raw)
	base, baseErr := url.Parse(p.queueURL)
	if err != nil || baseErr != nil || u.User != nil || u.Scheme != base.Scheme || u.Host != base.Host || !strings.HasPrefix(raw, p.queueURL+"/") {
		return "", fmt.Errorf("fal: queue url %q is not on the configured queue host", raw)
	}
	return strings.TrimPrefix(raw, p.queueURL), nil
}

func (s statusResponse) failure() error {
	if s.Error == "" {
		return nil
	}
	return &ai.APIError{Provider: providerName, Code: cmp.Or(s.ErrorType, "request_failed"), Message: transport.Truncate(s.Error)}
}

func (p *Provider) await(ctx context.Context, t ticket) (json.RawMessage, error) {
	interval := p.pollInterval
	for {
		st, err := p.status(ctx, t)
		if err != nil {
			return nil, p.abandon(ctx, t, err)
		}
		if st.Status == statusCompleted {
			if err := st.failure(); err != nil {
				return nil, err
			}
			return p.result(ctx, t)
		}
		if err := retry.Sleep(ctx, interval); err != nil {
			return nil, p.abandon(ctx, t, err)
		}
		interval = min(interval+interval/2, maxImagePoll)
	}
}

func (p *Provider) abandon(ctx context.Context, t ticket, cause error) error {
	if ctx.Err() == nil {
		return cause
	}
	if err := p.cancel(context.WithoutCancel(ctx), t); err != nil {
		return errors.Join(cause, fmt.Errorf("fal: cancel request %s: %w", t.RequestID, err))
	}
	return cause
}

func escapeModel(id string) string {
	parts := strings.Split(strings.Trim(id, "/"), "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

type outputFile struct {
	URL         string `json:"url"`
	ContentType string `json:"content_type"`
}
