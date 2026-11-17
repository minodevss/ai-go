// Package sigv4 signs HTTP requests with AWS Signature Version 4 using only the standard library.
package sigv4

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

const (
	algorithm     = "AWS4-HMAC-SHA256"
	dateFormat    = "20060102T150405Z"
	scopeFormat   = "20060102"
	terminator    = "aws4_request"
	headerDate    = "X-Amz-Date"
	headerToken   = "X-Amz-Security-Token"
	headerAuth    = "Authorization"
	headerHost    = "host"
	hexUpper      = "0123456789ABCDEF"
	unreservedSet = "-._~"
)

type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

type Signer struct {
	Credentials Credentials
	Region      string
	Service     string
}

type target struct {
	method  string
	path    string
	query   string
	host    string
	header  http.Header
	payload []byte
}

type signature struct {
	canonicalRequest string
	stringToSign     string
	authorization    string
	header           http.Header
}

func ignoredHeader(key string) bool {
	switch key {
	case "authorization", "user-agent", "x-amzn-trace-id", "expect", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate", "content-length":
		return true
	}
	return false
}

// Sign sets X-Amz-Date, X-Amz-Security-Token (when the credentials have a session token) and Authorization on r. It signs every header present on r except hop-by-hop ones, and the path exactly as r would send it. The caller must not change r afterwards.
func (s Signer) Sign(r *http.Request, payload []byte, now time.Time) {
	host := r.Host
	if host == "" {
		host = r.URL.Host
	}
	sig := s.authorize(target{
		method:  cmpMethod(r.Method),
		path:    r.URL.EscapedPath(),
		query:   r.URL.RawQuery,
		host:    host,
		header:  r.Header,
		payload: payload,
	}, now)
	r.Header = sig.header
}

func cmpMethod(m string) string {
	if m == "" {
		return http.MethodGet
	}
	return m
}

func (s Signer) authorize(t target, now time.Time) signature {
	now = now.UTC()
	header := t.header.Clone()
	if header == nil {
		header = http.Header{}
	}
	header.Set(headerDate, now.Format(dateFormat))
	if s.Credentials.SessionToken != "" {
		header.Set(headerToken, s.Credentials.SessionToken)
	}
	t.header = header
	canonical, signedHeaders := canonicalRequest(t)
	scope := strings.Join([]string{now.Format(scopeFormat), s.Region, s.Service, terminator}, "/")
	toSign := strings.Join([]string{algorithm, now.Format(dateFormat), scope, hashHex([]byte(canonical))}, "\n")
	sum := hex.EncodeToString(mac(s.signingKey(now), toSign))
	auth := algorithm + " Credential=" + s.Credentials.AccessKeyID + "/" + scope + ", SignedHeaders=" + signedHeaders + ", Signature=" + sum
	header.Set(headerAuth, auth)
	return signature{canonicalRequest: canonical, stringToSign: toSign, authorization: auth, header: header}
}

func (s Signer) signingKey(now time.Time) []byte {
	key := mac([]byte("AWS4"+s.Credentials.SecretAccessKey), now.Format(scopeFormat))
	key = mac(key, s.Region)
	key = mac(key, s.Service)
	return mac(key, terminator)
}

func canonicalRequest(t target) (string, string) {
	headers, signed := canonicalHeaders(t)
	return strings.Join([]string{
		t.method,
		canonicalPath(t.path),
		canonicalQuery(t.query),
		headers,
		signed,
		hashHex(t.payload),
	}, "\n"), signed
}

func canonicalHeaders(t target) (string, string) {
	values := map[string][]string{headerHost: {t.host}}
	for name, vs := range t.header {
		key := strings.ToLower(name)
		if ignoredHeader(key) || key == headerHost {
			continue
		}
		values[key] = append(values[key], vs...)
	}
	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	slices.Sort(names)
	var b strings.Builder
	for _, name := range names {
		cleaned := make([]string, len(values[name]))
		for i, v := range values[name] {
			cleaned[i] = strings.Join(strings.Fields(v), " ")
		}
		b.WriteString(name + ":" + strings.Join(cleaned, ",") + "\n")
	}
	return b.String(), strings.Join(names, ";")
}

func canonicalPath(path string) string {
	if path == "" {
		return "/"
	}
	var kept []string
	for seg := range strings.SplitSeq(path, "/") {
		switch seg {
		case "", ".":
		case "..":
			if len(kept) > 0 {
				kept = kept[:len(kept)-1]
			}
		default:
			kept = append(kept, URIEncode(seg))
		}
	}
	out := "/" + strings.Join(kept, "/")
	if len(kept) > 0 && strings.HasSuffix(path, "/") {
		out += "/"
	}
	return out
}

func canonicalQuery(raw string) string {
	if raw == "" {
		return ""
	}
	type pair struct{ name, value string }
	var pairs []pair
	for part := range strings.SplitSeq(raw, "&") {
		if part == "" {
			continue
		}
		name, value, _ := strings.Cut(part, "=")
		pairs = append(pairs, pair{URIEncode(unescape(name)), URIEncode(unescape(value))})
	}
	slices.SortFunc(pairs, func(a, b pair) int {
		if c := strings.Compare(a.name, b.name); c != 0 {
			return c
		}
		return strings.Compare(a.value, b.value)
	})
	out := make([]string, len(pairs))
	for i, p := range pairs {
		out[i] = p.name + "=" + p.value
	}
	return strings.Join(out, "&")
}

func unescape(s string) string {
	if out, err := url.QueryUnescape(s); err == nil {
		return out
	}
	return s
}

// URIEncode percent-encodes every byte except A-Z a-z 0-9 - . _ ~, with uppercase hex, as the SigV4 specification requires.
func URIEncode(s string) string {
	var b strings.Builder
	for i := range len(s) {
		c := s[i]
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || strings.IndexByte(unreservedSet, c) >= 0 {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hexUpper[c>>4])
		b.WriteByte(hexUpper[c&15])
	}
	return b.String()
}

func hashHex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func mac(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}
