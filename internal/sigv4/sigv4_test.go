package sigv4

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

const (
	suiteDir     = "testdata/aws4_testsuite"
	suiteRegion  = "us-east-1"
	suiteService = "service"
	suiteToken   = "6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267"
)

var suiteTime = time.Date(2015, 8, 30, 12, 36, 0, 0, time.UTC)

func suiteCredentials() Credentials {
	return Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", path, err)
	}
	return string(raw)
}

func parseRequest(t *testing.T, raw string) target {
	t.Helper()
	head, body, _ := strings.Cut(raw, "\n\n")
	lines := strings.Split(strings.TrimRight(head, "\n"), "\n")
	method, rest, _ := strings.Cut(lines[0], " ")
	uri, _, _ := strings.Cut(rest, " HTTP/")
	path, query, _ := strings.Cut(uri, "?")
	header := http.Header{}
	var last string
	host := ""
	for _, line := range lines[1:] {
		if strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t") {
			values := header[last]
			values[len(values)-1] += " " + strings.TrimSpace(line)
			continue
		}
		name, value, ok := strings.Cut(line, ":")
		if !ok {
			t.Fatalf("malformed header line %q", line)
		}
		if strings.EqualFold(name, "host") {
			host = value
			continue
		}
		last = http.CanonicalHeaderKey(name)
		header[last] = append(header[last], value)
	}
	return target{method: method, path: path, query: query, host: host, header: header, payload: []byte(body)}
}

func suiteCases(t *testing.T) map[string]string {
	t.Helper()
	cases := map[string]string{}
	err := filepath.WalkDir(suiteDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".req") {
			return nil
		}
		cases[strings.TrimSuffix(filepath.Base(path), ".req")] = strings.TrimSuffix(path, ".req")
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir() error = %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("no test vectors found")
	}
	return cases
}

func TestSigner_authorize_awsTestSuite(t *testing.T) {
	for name, base := range suiteCases(t) {
		t.Run(name, func(t *testing.T) {
			creds := suiteCredentials()
			if name == "get-vanilla-with-session-token" {
				creds.SessionToken = suiteToken
			}
			sig := Signer{Credentials: creds, Region: suiteRegion, Service: suiteService}.authorize(parseRequest(t, readFile(t, base+".req")), suiteTime)
			if diff := cmp.Diff(readFile(t, base+".creq"), sig.canonicalRequest); diff != "" {
				t.Errorf("canonical request mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(readFile(t, base+".sts"), sig.stringToSign); diff != "" {
				t.Errorf("string to sign mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(readFile(t, base+".authz"), sig.authorization); diff != "" {
				t.Errorf("authorization mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSigner_Sign_sessionTokenIsSigned(t *testing.T) {
	r, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.amazonaws.com/", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	creds := suiteCredentials()
	creds.SessionToken = "tok"
	Signer{Credentials: creds, Region: suiteRegion, Service: suiteService}.Sign(r, nil, suiteTime)
	if got := r.Header.Get(headerToken); got != "tok" {
		t.Errorf("%s = %q, want tok", headerToken, got)
	}
	if got := r.Header.Get(headerAuth); !strings.Contains(got, "SignedHeaders=host;x-amz-date;x-amz-security-token,") {
		t.Errorf("Authorization = %q, want x-amz-security-token signed", got)
	}
}

func TestSigner_Sign_ignoresHopByHopHeaders(t *testing.T) {
	r, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.amazonaws.com/", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	r.Header.Set("User-Agent", "ai-go")
	r.Header.Set("Content-Type", "application/json")
	Signer{Credentials: suiteCredentials(), Region: suiteRegion, Service: suiteService}.Sign(r, []byte("{}"), suiteTime)
	if got := r.Header.Get(headerAuth); !strings.Contains(got, "SignedHeaders=content-type;host;x-amz-date,") {
		t.Errorf("Authorization = %q, want content-type;host;x-amz-date signed", got)
	}
}

func TestSigner_Sign_modelIDIsDoubleEncoded(t *testing.T) {
	r, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext() error = %v", err)
	}
	sig := Signer{Credentials: suiteCredentials(), Region: suiteRegion, Service: "bedrock"}.authorize(target{
		method: http.MethodPost, path: r.URL.EscapedPath(), host: r.URL.Host,
	}, suiteTime)
	want := "POST\n/model/anthropic.claude-3-sonnet-20240229-v1%253A0/converse\n"
	if !strings.HasPrefix(sig.canonicalRequest, want) {
		t.Errorf("canonical request = %q, want prefix %q", sig.canonicalRequest, want)
	}
}

func TestURIEncode(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{in: "AZaz09-._~", want: "AZaz09-._~"},
		{in: "a b", want: "a%20b"},
		{in: "a/b:c", want: "a%2Fb%3Ac"},
		{in: "é", want: "%C3%A9"},
		{in: "%", want: "%25"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			if got := URIEncode(tt.in); got != tt.want {
				t.Errorf("URIEncode(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
