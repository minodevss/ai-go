package ui

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
)

const (
	defaultMaxBody     = 1 << 20
	defaultMaxMessages = 200
	defaultMaxParts    = 100
	defaultMaxDepth    = 32
)

type Trigger string

const (
	TriggerSubmit     Trigger = "submit-message"
	TriggerRegenerate Trigger = "regenerate-message"
)

// Limits bounds what ReadRequest accepts; zero fields use the defaults: 1 MiB body, 200 messages, 100 parts per message, JSON depth 32.
type Limits struct {
	MaxBodyBytes int64
	MaxMessages  int
	MaxParts     int
	MaxDepth     int
	FilePolicy   FilePolicy
}

// Request is the useChat request body. Everything in it is untrusted client input.
type Request struct {
	ID        string
	Messages  []Message
	Trigger   Trigger
	MessageID string
	Extra     map[string]json.RawMessage
	files     FilePolicy
}

func (l Limits) withDefaults() Limits {
	if l.MaxBodyBytes <= 0 {
		l.MaxBodyBytes = defaultMaxBody
	}
	if l.MaxMessages <= 0 {
		l.MaxMessages = defaultMaxMessages
	}
	if l.MaxParts <= 0 {
		l.MaxParts = defaultMaxParts
	}
	if l.MaxDepth <= 0 {
		l.MaxDepth = defaultMaxDepth
	}
	return l
}

// ReadRequest reads and validates a useChat request. Every failure is a *RequestError that names the HTTP status to answer with.
func ReadRequest(r *http.Request, lim Limits) (Request, error) {
	lim = lim.withDefaults()
	if mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err != nil || mt != "application/json" {
		return Request{}, requestError(http.StatusUnsupportedMediaType, ErrInvalidRequest, "content type must be application/json")
	}
	body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, lim.MaxBodyBytes))
	if err != nil {
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			return Request{}, requestError(http.StatusRequestEntityTooLarge, ErrRequestTooBig, "body exceeds %d bytes", lim.MaxBodyBytes)
		}
		return Request{}, badRequest("read body: %v", err)
	}
	return parseRequest(body, lim)
}

func parseRequest(body []byte, lim Limits) (Request, error) {
	if err := checkDepth(body, lim.MaxDepth); err != nil {
		return Request{}, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return Request{}, badRequest("body is not a JSON object")
	}
	req := Request{files: lim.FilePolicy}
	for key, raw := range fields {
		var err error
		switch key {
		case "id":
			err = json.Unmarshal(raw, &req.ID)
		case "trigger":
			err = json.Unmarshal(raw, &req.Trigger)
		case "messageId":
			err = json.Unmarshal(raw, &req.MessageID)
		case "messages":
			err = json.Unmarshal(raw, &req.Messages)
		default:
			if req.Extra == nil {
				req.Extra = map[string]json.RawMessage{}
			}
			req.Extra[key] = raw
		}
		if err != nil {
			return Request{}, badRequest("field %q: %v", key, err)
		}
	}
	return req, req.validate(lim)
}

func (q Request) validate(lim Limits) error {
	if len(q.Messages) > lim.MaxMessages {
		return badRequest("more than %d messages", lim.MaxMessages)
	}
	switch q.Trigger {
	case "", TriggerSubmit, TriggerRegenerate:
	default:
		return badRequest("unknown trigger %q", q.Trigger)
	}
	for i, m := range q.Messages {
		switch m.Role {
		case RoleSystem, RoleUser, RoleAssistant:
		default:
			return badRequest("message %d has unknown role %q", i, m.Role)
		}
		if len(m.Parts) > lim.MaxParts {
			return badRequest("message %d has more than %d parts", i, lim.MaxParts)
		}
		if m.Role != RoleUser {
			continue
		}
		if _, err := userToAI(m, lim.FilePolicy, false); err != nil {
			return err
		}
	}
	return nil
}

func checkDepth(body []byte, maxDepth int) error {
	dec := json.NewDecoder(bytes.NewReader(body))
	depth := 0
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) && depth == 0 {
			return nil
		}
		if err != nil {
			return badRequest("body is not valid JSON")
		}
		if d, ok := tok.(json.Delim); ok {
			if d == '{' || d == '[' {
				depth++
				if depth > maxDepth {
					return badRequest("JSON nesting deeper than %d", maxDepth)
				}
			} else {
				depth--
			}
		}
	}
}
