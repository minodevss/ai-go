// Package ui serves the Vercel AI SDK UI message stream protocol v1 (useChat) from Go.
package ui

import (
	"errors"
	"fmt"
	"net/http"
)

const (
	streamHeader   = "x-vercel-ai-ui-message-stream"
	streamVersion  = "v1"
	defaultErrText = "An error occurred."
)

var (
	ErrNoStream       = errors.New("ui: no stream")
	ErrBusy           = errors.New("ui: chat already has an active stream")
	ErrTooManyStreams = errors.New("ui: too many active streams for owner")
	ErrStreamTooLarge = errors.New("ui: stream exceeds size limit")
	ErrBadSignature   = errors.New("ui: invalid message signature")
	ErrWeakKey        = errors.New("ui: signing key must be at least 32 bytes")
	ErrInvalidRequest = errors.New("ui: invalid request")
	ErrRequestTooBig  = errors.New("ui: request too large")
	ErrApproval       = errors.New("ui: invalid tool approval")
	ErrFile           = errors.New("ui: file not allowed")
)

// RequestError is returned for every problem caused by the client's request; Status is the suggested HTTP status.
type RequestError struct {
	Status int
	Err    error
}

func (e *RequestError) Error() string { return e.Err.Error() }

func (e *RequestError) Unwrap() error { return e.Err }

func requestError(status int, sentinel error, format string, args ...any) error {
	return &RequestError{Status: status, Err: fmt.Errorf("%w: %s", sentinel, fmt.Sprintf(format, args...))}
}

func badRequest(format string, args ...any) error {
	return requestError(http.StatusBadRequest, ErrInvalidRequest, format, args...)
}
