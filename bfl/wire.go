package bfl

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
)

type submitResponse struct {
	ID         string  `json:"id"`
	PollingURL string  `json:"polling_url"`
	Cost       float64 `json:"cost"`
}

type pollResponse struct {
	ID       string          `json:"id"`
	Status   string          `json:"status"`
	Result   json.RawMessage `json:"result"`
	Progress *float64        `json:"progress"`
	Details  json.RawMessage `json:"details"`
}

func decode(raw []byte, out any) error {
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("bfl: decode response: %w", err)
	}
	return nil
}

func (r *pollResponse) sample() string {
	var res struct {
		Sample string `json:"sample"`
	}
	if json.Unmarshal(r.Result, &res) != nil {
		return ""
	}
	return res.Sample
}

func (r *pollResponse) moderationReasons() []string {
	var d struct {
		Reasons []string `json:"Moderation Reasons"`
	}
	if json.Unmarshal(r.Details, &d) != nil {
		return nil
	}
	return d.Reasons
}

func (r *pollResponse) pending() bool {
	switch r.Status {
	case "Pending", "Reasoning", "Generating", "Queued", "Processing":
		return true
	default:
		return false
	}
}

func (r *pollResponse) failure() error {
	switch r.Status {
	case "Ready":
		return nil
	case "Request Moderated", "Content Moderated":
		message := r.Status
		if reasons := r.moderationReasons(); len(reasons) > 0 {
			message += ": " + strings.Join(reasons, ", ")
		}
		return mediaio.Blocked(providerName, message)
	case "Task not found":
		return &ai.APIError{Provider: providerName, StatusCode: http.StatusNotFound, Code: "task_not_found", Message: "task not found or expired"}
	case "Error", "Failed":
		message := "generation failed"
		if len(r.Details) > 0 && string(r.Details) != "null" {
			message += ": " + string(r.Details)
		}
		return mediaio.Failed(providerName, mediaio.CodeJobFailed, message)
	default:
		return mediaio.Failed(providerName, mediaio.CodeJobFailed, fmt.Sprintf("unexpected task status %q", r.Status))
	}
}

func (p *Provider) pollTarget(s submitResponse) string {
	if s.PollingURL != "" {
		return s.PollingURL
	}
	return strings.TrimRight(p.baseURL, "/") + "/v1/get_result?id=" + url.QueryEscape(s.ID)
}
