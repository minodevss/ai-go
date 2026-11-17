// Package record is an http.RoundTripper that records exchanges to JSON files under testdata and replays them offline.
package record

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
)

const redacted = "REDACTED"

var update = flag.Bool("update", false, "re-record HTTP fixtures against the real upstream")

type File struct {
	Exchanges []Exchange `json:"exchanges"`
}

type Exchange struct {
	Request  Request  `json:"request"`
	Response Response `json:"response"`
}

// Request and Response keep JSON bodies as JSON and any other body as BodyLines, one element per line.
type Request struct {
	Method    string          `json:"method"`
	URL       string          `json:"url"`
	Header    http.Header     `json:"header,omitzero"`
	Body      json.RawMessage `json:"body,omitzero"`
	BodyLines []string        `json:"bodyLines,omitzero"`
}

type Response struct {
	Status    int             `json:"status"`
	Header    http.Header     `json:"header,omitzero"`
	Body      json.RawMessage `json:"body,omitzero"`
	BodyLines []string        `json:"bodyLines,omitzero"`
}

type Option func(*options)

type options struct {
	upstream    http.RoundTripper
	matchBody   bool
	rewrite     func([]byte) []byte
	redactNames []string
}

// Transport replays the exchanges in order, or records them when the -update flag is set.
type Transport struct {
	tb       testing.TB
	path     string
	opts     options
	record   bool
	mu       sync.Mutex
	file     File
	next     int
	requests []Request
}

func Updating() bool { return *update }

func WithUpstream(rt http.RoundTripper) Option {
	return func(o *options) { o.upstream = rt }
}

// WithBodyMatch makes replay fail when a request body is not JSON-equal to the recorded one.
func WithBodyMatch() Option {
	return func(o *options) { o.matchBody = true }
}

// WithBodyRewrite changes recorded request and response bodies before they are stored, to scrub identifiers.
func WithBodyRewrite(f func(body []byte) []byte) Option {
	return func(o *options) { o.rewrite = f }
}

// WithRedactedHeaders redacts the named headers, matched case-insensitively, in recordings. The live request keeps them.
func WithRedactedHeaders(names ...string) Option {
	return func(o *options) {
		for _, name := range names {
			o.redactNames = append(o.redactNames, strings.ToLower(name))
		}
	}
}

func New(tb testing.TB, path string, opts ...Option) *Transport {
	tb.Helper()
	t := &Transport{tb: tb, path: path, record: *update}
	t.opts.upstream = http.DefaultTransport
	t.opts.redactNames = []string{"authorization", "x-api-key", "api-key", "x-goog-api-key", "xi-api-key", "x-key", "cookie", "set-cookie", "proxy-authorization", "openai-organization", "openai-project"}
	for _, opt := range opts {
		opt(&t.opts)
	}
	if t.record {
		tb.Cleanup(t.save)
		return t
	}
	file, err := Load(path)
	if err != nil {
		tb.Fatalf("%v (run with -update to record it)", err)
	}
	t.file = file
	return t
}

func Load(path string) (File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return File{}, fmt.Errorf("record: load fixture: %w", err)
	}
	var f File
	if err := json.Unmarshal(raw, &f); err != nil {
		return File{}, fmt.Errorf("record: decode fixture %s: %w", path, err)
	}
	return f, nil
}

func Save(path string, f File) error {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(f); err != nil {
		return fmt.Errorf("record: encode fixture: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("record: create fixture directory: %w", err)
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		return fmt.Errorf("record: write fixture: %w", err)
	}
	return nil
}

func (r Request) Bytes() []byte { return bodyBytes(r.Body, r.BodyLines) }

func (r Response) Bytes() []byte { return bodyBytes(r.Body, r.BodyLines) }

func (t *Transport) Requests() []Request {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.requests)
}

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	body, err := readBody(req)
	if err != nil {
		return nil, err
	}
	rec := Request{Method: req.Method, URL: sanitizeURL(req.URL.String()), Header: t.redact(req.Header)}
	rec.Body, rec.BodyLines = splitBody(body)
	t.mu.Lock()
	t.requests = append(t.requests, rec)
	t.mu.Unlock()
	if t.record {
		return t.forward(req, rec, body)
	}
	return t.replay(rec, body)
}

func (t *Transport) replay(rec Request, body []byte) (*http.Response, error) {
	t.mu.Lock()
	i := t.next
	t.next++
	t.mu.Unlock()
	if i >= len(t.file.Exchanges) {
		return nil, fmt.Errorf("record: %s has %d exchanges, request %d (%s %s) has no recording", t.path, len(t.file.Exchanges), i+1, rec.Method, rec.URL)
	}
	ex := t.file.Exchanges[i]
	if rec.Method != ex.Request.Method || pathOf(rec.URL) != pathOf(sanitizeURL(ex.Request.URL)) {
		return nil, fmt.Errorf("record: request %d is %s %s, recorded %s %s", i+1, rec.Method, pathOf(rec.URL), ex.Request.Method, pathOf(sanitizeURL(ex.Request.URL)))
	}
	if t.opts.matchBody {
		if err := compareBodies(i+1, t.path, ex.Request.Bytes(), body); err != nil {
			return nil, err
		}
	}
	out := ex.Response.Bytes()
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", ex.Response.Status, http.StatusText(ex.Response.Status)),
		StatusCode:    ex.Response.Status,
		Header:        ex.Response.Header.Clone(),
		Body:          io.NopCloser(bytes.NewReader(out)),
		ContentLength: int64(len(out)),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
	}, nil
}

func (t *Transport) forward(req *http.Request, rec Request, body []byte) (*http.Response, error) {
	if t.opts.rewrite != nil {
		rec.Body, rec.BodyLines = splitBody(t.opts.rewrite(bytes.Clone(body)))
	}
	t.mu.Lock()
	idx := len(t.file.Exchanges)
	t.file.Exchanges = append(t.file.Exchanges, Exchange{Request: rec})
	t.mu.Unlock()
	resp, err := t.opts.upstream.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	resp.Body = &tee{src: resp.Body, done: func(body []byte) {
		if t.opts.rewrite != nil {
			body = t.opts.rewrite(bytes.Clone(body))
		}
		out := Response{Status: resp.StatusCode, Header: t.redact(resp.Header)}
		out.Body, out.BodyLines = splitBody(body)
		t.mu.Lock()
		t.file.Exchanges[idx].Response = out
		t.mu.Unlock()
	}}
	return resp, nil
}

func (t *Transport) save() {
	t.tb.Helper()
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := Save(t.path, t.file); err != nil {
		t.tb.Errorf("%v", err)
	}
}

func (t *Transport) redact(h http.Header) http.Header {
	out := h.Clone()
	out.Del("Content-Length")
	out.Del("Transfer-Encoding")
	for k, values := range out {
		name := strings.ToLower(k)
		switch {
		case slices.Contains(t.opts.redactNames, name) || containsAny(name, secretNameParts):
			out[k] = []string{redacted}
		case containsAny(name, urlNameParts):
			out[k] = mapValues(values, sanitizeURL)
		}
	}
	return out
}

var (
	secretNameParts = []string{"key", "token", "secret", "authorization", "cookie", "password", "credential", "signature"}
	urlNameParts    = []string{"url", "location"}
	secretQueryKeys = []string{"key", "token", "secret", "sig", "auth", "credential", "password", "upload_id", "uploadid", "x-amz-", "x-goog-", "expires"}
)

func containsAny(s string, parts []string) bool {
	return slices.ContainsFunc(parts, func(p string) bool { return strings.Contains(s, p) })
}

func mapValues(values []string, f func(string) string) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = f(v)
	}
	return out
}

func sanitizeURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		base, _, _ := strings.Cut(raw, "?")
		return base
	}
	u.User = nil
	u.Fragment = ""
	if u.RawQuery != "" {
		pairs := strings.Split(u.RawQuery, "&")
		for i, pair := range pairs {
			name, _, _ := strings.Cut(pair, "=")
			if decoded, err := url.QueryUnescape(name); err == nil && containsAny(strings.ToLower(decoded), secretQueryKeys) {
				pairs[i] = name + "=" + redacted
			}
		}
		u.RawQuery = strings.Join(pairs, "&")
	}
	return u.String()
}

type tee struct {
	src  io.ReadCloser
	buf  bytes.Buffer
	once sync.Once
	done func([]byte)
}

func (r *tee) Read(p []byte) (int, error) {
	n, err := r.src.Read(p)
	r.buf.Write(p[:n])
	if errors.Is(err, io.EOF) {
		r.finish()
	}
	return n, err
}

func (r *tee) Close() error {
	r.finish()
	return r.src.Close()
}

func (r *tee) finish() { r.once.Do(func() { r.done(r.buf.Bytes()) }) }

func readBody(req *http.Request) ([]byte, error) {
	if req.Body == nil {
		return nil, nil
	}
	defer req.Body.Close()
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, fmt.Errorf("record: read request body: %w", err)
	}
	req.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

func splitBody(body []byte) (json.RawMessage, []string) {
	if len(body) == 0 {
		return nil, nil
	}
	if json.Valid(body) {
		return json.RawMessage(bytes.Clone(body)), nil
	}
	return nil, strings.Split(string(body), "\n")
}

func bodyBytes(raw json.RawMessage, lines []string) []byte {
	if len(raw) > 0 {
		var b bytes.Buffer
		if err := json.Compact(&b, raw); err != nil {
			return raw
		}
		return b.Bytes()
	}
	return []byte(strings.Join(lines, "\n"))
}

func compareBodies(request int, path string, want, got []byte) error {
	var w, g any
	if err := json.Unmarshal(want, &w); err != nil {
		return fmt.Errorf("record: request %d body differs from %s: recorded body is not JSON: %w", request, path, err)
	}
	if err := json.Unmarshal(got, &g); err != nil {
		return fmt.Errorf("record: request %d body differs from %s: sent body is not JSON: %w", request, path, err)
	}
	if diff := cmp.Diff(w, g); diff != "" {
		return fmt.Errorf("record: request %d body differs from %s (-recorded +sent):\n%s", request, path, diff)
	}
	return nil
}

func pathOf(url string) string {
	_, rest, ok := strings.Cut(url, "://")
	if !ok {
		return url
	}
	_, path, _ := strings.Cut(rest, "/")
	return "/" + path
}
