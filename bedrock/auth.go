package bedrock

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go/internal/sigv4"
)

const (
	signingName   = "bedrock"
	errorTypeKey  = "x-amzn-errortype"
	typeField     = "__type"
	maxErrorBytes = 1 << 20
)

type authTransport struct {
	base        http.RoundTripper
	credentials Credentials
	region      string
	err         error
}

func newHTTPClient(base *http.Client, credentials Credentials, region string, err error) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	return &http.Client{
		Transport:     &authTransport{base: rt, credentials: credentials, region: region, err: err},
		CheckRedirect: base.CheckRedirect,
		Jar:           base.Jar,
		Timeout:       base.Timeout,
	}
}

func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.err != nil {
		closeBody(req)
		return nil, t.err
	}
	if t.credentials != nil {
		signed, err := t.sign(req)
		if err != nil {
			closeBody(req)
			return nil, err
		}
		req = signed
	}
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp.StatusCode < http.StatusMultipleChoices {
		return resp, err
	}
	return tagErrorType(resp), nil
}

func (t *authTransport) sign(req *http.Request) (*http.Request, error) {
	keys, err := t.credentials.Retrieve(req.Context())
	if err != nil {
		return nil, fmt.Errorf("bedrock: retrieve credentials: %w", err)
	}
	if keys.AccessKeyID == "" || keys.SecretAccessKey == "" {
		return nil, ErrNoCredentials
	}
	if t.region == "" {
		return nil, ErrNoRegion
	}
	payload, err := bodyBytes(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: read request body: %w", err)
	}
	signed := req.Clone(req.Context())
	sigv4.Signer{
		Credentials: sigv4.Credentials{AccessKeyID: keys.AccessKeyID, SecretAccessKey: keys.SecretAccessKey, SessionToken: keys.SessionToken},
		Region:      t.region,
		Service:     signingName,
	}.Sign(signed, payload, time.Now())
	return signed, nil
}

func bodyBytes(req *http.Request) ([]byte, error) {
	if req.Body == nil || req.Body == http.NoBody {
		return nil, nil
	}
	if req.GetBody != nil {
		rc, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		defer rc.Close()
		return io.ReadAll(rc)
	}
	raw, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	req.Body = io.NopCloser(bytes.NewReader(raw))
	return raw, nil
}

func closeBody(req *http.Request) {
	if req.Body != nil {
		req.Body.Close()
	}
}

func tagErrorType(resp *http.Response) *http.Response {
	// AWS restJson1 services send the error type in the x-amzn-errortype header: https://smithy.io/2.0/aws/protocols/aws-restjson1-protocol.html#operation-error-serialization
	typ := sanitizeErrorType(resp.Header.Get(errorTypeKey))
	if typ == "" {
		return resp
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBytes))
	resp.Body.Close()
	if err != nil {
		raw = nil
	}
	if retry := modelErrorRetry(typ, raw); retry != nil {
		resp.Header.Set("X-Should-Retry", strconv.FormatBool(*retry))
	}
	tagged, err := withType(raw, typ)
	if err != nil {
		tagged = raw
	}
	resp.Body = io.NopCloser(bytes.NewReader(tagged))
	resp.ContentLength = int64(len(tagged))
	resp.Header.Del("Content-Length")
	return resp
}

func withType(body []byte, typ string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil || fields == nil {
		fields = map[string]json.RawMessage{}
		if len(body) > 0 {
			text, err := json.Marshal(string(body))
			if err != nil {
				return nil, err
			}
			fields["message"] = text
		}
	}
	if _, ok := fields[typeField]; !ok {
		name, err := json.Marshal(typ)
		if err != nil {
			return nil, err
		}
		fields[typeField] = name
	}
	return json.Marshal(fields)
}

func sanitizeErrorType(v string) string {
	if i := strings.LastIndex(v, "#"); i >= 0 {
		v = v[i+1:]
	}
	name, _, _ := strings.Cut(v, ":")
	return strings.TrimSpace(name)
}
