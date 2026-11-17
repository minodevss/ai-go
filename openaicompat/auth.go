package openaicompat

import (
	"encoding/json"
	"net/http"

	"github.com/minodevss/ai-go/internal/transport"
)

type bearerTransport struct {
	base  http.RoundTripper
	token TokenSource
}

func withToken(client *http.Client, token TokenSource) *http.Client {
	if token == nil {
		return client
	}
	out := http.Client{}
	base := http.DefaultTransport
	if client != nil {
		out = *client
		if client.Transport != nil {
			base = client.Transport
		}
	}
	out.Transport = bearerTransport{base: base, token: token}
	return &out
}

func (t bearerTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	tok, err := t.token(r.Context())
	if err != nil {
		if r.Body != nil {
			r.Body.Close()
		}
		return nil, err
	}
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+tok)
	return t.base.RoundTrip(r)
}

func parseError(body []byte) (code, message string) {
	if code, message = transport.ParseError(body); message != "" {
		return code, message
	}
	var env struct {
		Message string          `json:"message"`
		Detail  string          `json:"detail"`
		Type    string          `json:"type"`
		Code    json.RawMessage `json:"code"`
	}
	if err := json.Unmarshal(body, &env); err != nil && env.Message == "" {
		return code, ""
	}
	var s string
	if json.Unmarshal(env.Code, &s) != nil {
		var n json.Number
		if json.Unmarshal(env.Code, &n) == nil {
			s = n.String()
		}
	}
	if s == "" {
		s = env.Type
	}
	if env.Message != "" {
		return s, env.Message
	}
	return s, env.Detail
}
