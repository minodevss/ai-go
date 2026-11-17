package bedrock_test

import (
	"context"
	"errors"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
	"github.com/minodevss/ai-go/internal/sigv4"
)

var authorizationPattern = regexp.MustCompile(`^AWS4-HMAC-SHA256 Credential=([^/]+)/(\d{8})/([^/]+)/bedrock/aws4_request, SignedHeaders=([a-z0-9;-]+), Signature=([0-9a-f]{64})$`)

func hello() ai.TextRequest { return ai.TextRequest{Messages: []ai.Message{ai.User("hi")}} }

func TestProvider_signsRequestsWithSigV4(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	p := f.provider(bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", SessionToken: "session-token"})))
	if _, err := ai.GenerateText(t.Context(), p.Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	got := f.last(t)
	m := authorizationPattern.FindStringSubmatch(got.Header.Get("Authorization"))
	if m == nil {
		t.Fatalf("Authorization = %q, want an AWS4-HMAC-SHA256 header for service bedrock", got.Header.Get("Authorization"))
	}
	if m[1] != "AKIDEXAMPLE" || m[3] != "us-east-1" {
		t.Errorf("credential scope = %v, want AKIDEXAMPLE in us-east-1", m[1:4])
	}
	if want := "accept;content-type;host;x-amz-date;x-amz-security-token"; m[4] != want {
		t.Errorf("signed headers = %q, want %q", m[4], want)
	}
	if got.Header.Get("X-Amz-Security-Token") != "session-token" {
		t.Errorf("X-Amz-Security-Token = %q, want session-token", got.Header.Get("X-Amz-Security-Token"))
	}
}

func TestProvider_signatureVerifiesOnTheServer(t *testing.T) {
	var verified atomic.Bool
	f := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		date, err := time.Parse("20060102T150405Z", r.Header.Get("X-Amz-Date"))
		if err != nil {
			t.Errorf("X-Amz-Date = %q: %v", r.Header.Get("X-Amz-Date"), err)
			return
		}
		got := r.Header.Get("Authorization")
		again := r.Clone(r.Context())
		parsed := authorizationPattern.FindStringSubmatch(got)
		if parsed == nil {
			t.Errorf("Authorization = %q, want an AWS4-HMAC-SHA256 header", got)
			return
		}
		signed := strings.Split(parsed[4], ";")
		for name := range again.Header {
			if !slices.Contains(signed, strings.ToLower(name)) || strings.EqualFold(name, "X-Amz-Date") {
				again.Header.Del(name)
			}
		}
		payload := readAll(t, r)
		sigv4.Signer{
			Credentials: sigv4.Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"},
			Region:      "us-east-1",
			Service:     "bedrock",
		}.Sign(again, payload, date)
		verified.Store(again.Header.Get("Authorization") == got)
		jsonReply(minimalResponse)(w, r)
	})
	if _, err := ai.GenerateText(t.Context(), f.provider().Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if !verified.Load() {
		t.Error("server could not reproduce the Authorization header from the request it received")
	}
}

func TestProvider_bearerToken(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	p := bedrock.New(bedrock.WithAPIKey("api-key-1"), bedrock.WithBaseURL(f.URL))
	if _, err := ai.GenerateText(t.Context(), p.Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	got := f.last(t).Header
	if got.Get("Authorization") != "Bearer api-key-1" || got.Get("X-Amz-Date") != "" {
		t.Errorf("headers = %v, want a bearer token and no SigV4 signature", got)
	}
}

func TestProvider_bearerTokenFromEnvironment(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	if _, err := ai.GenerateText(t.Context(), bedrock.New(bedrock.WithBaseURL(f.URL)).Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if got := f.last(t).Header.Get("Authorization"); got != "Bearer env-key" {
		t.Errorf("Authorization = %q, want the environment bearer token to win over environment keys", got)
	}
}

func TestProvider_credentialsOptionBeatsEnvironmentBearer(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
	p := bedrock.New(bedrock.WithRegion("eu-west-1"), bedrock.WithBaseURL(f.URL), bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "AKID", SecretAccessKey: "s"})))
	if _, err := ai.GenerateText(t.Context(), p.Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	m := authorizationPattern.FindStringSubmatch(f.last(t).Header.Get("Authorization"))
	if m == nil || m[3] != "eu-west-1" {
		t.Errorf("Authorization = %q, want a SigV4 signature for eu-west-1", f.last(t).Header.Get("Authorization"))
	}
}

func TestProvider_environmentCredentials(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_REGION", "ap-northeast-2")
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIDENV")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	t.Setenv("AWS_SESSION_TOKEN", "tok")
	if _, err := ai.GenerateText(t.Context(), bedrock.New(bedrock.WithBaseURL(f.URL)).Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	got := f.last(t).Header
	m := authorizationPattern.FindStringSubmatch(got.Get("Authorization"))
	if m == nil || m[1] != "AKIDENV" || m[3] != "ap-northeast-2" || got.Get("X-Amz-Security-Token") != "tok" {
		t.Errorf("headers = %v, want AKIDENV in ap-northeast-2 with the session token", got)
	}
}

func TestProvider_defaultEndpoint(t *testing.T) {
	var host string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		host = r.URL.Host
		return nil, errors.New("bedrock: stop")
	})}
	p := bedrock.New(bedrock.WithRegion("us-west-2"), bedrock.WithAPIKey("k"), bedrock.WithHTTPClient(client))
	if _, err := ai.GenerateText(t.Context(), p.Model(testModel), noRetries(hello())); err == nil {
		t.Fatal("GenerateText() error = nil, want the stub error")
	}
	if host != "bedrock-runtime.us-west-2.amazonaws.com" {
		t.Errorf("host = %q, want bedrock-runtime.us-west-2.amazonaws.com", host)
	}
}

func TestProvider_missingCredentialsAndRegion(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("AWS_REGION", "")
	f := newFakeServer(t, jsonReply(minimalResponse))
	tests := []struct {
		name string
		p    *bedrock.Provider
		want error
	}{
		{name: "no credentials", p: bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithBaseURL(f.URL)), want: bedrock.ErrNoCredentials},
		{name: "no region", p: bedrock.New(bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "a", SecretAccessKey: "b"}))), want: bedrock.ErrNoRegion},
		{name: "no region with endpoint", p: bedrock.New(bedrock.WithBaseURL(f.URL), bedrock.WithCredentials(bedrock.StaticCredentials(bedrock.Keys{AccessKeyID: "a", SecretAccessKey: "b"}))), want: bedrock.ErrNoRegion},
		{name: "no region with bearer token", p: bedrock.New(bedrock.WithAPIKey("k")), want: bedrock.ErrNoRegion},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateText(t.Context(), tt.p.Model(testModel), noRetries(hello()))
			if !errors.Is(err, tt.want) {
				t.Errorf("GenerateText() error = %v, want %v", err, tt.want)
			}
		})
	}
	if f.count() != 0 {
		t.Errorf("server saw %d requests, want 0", f.count())
	}
}

func TestProvider_credentialsAreRetrievedPerRequest(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	var calls atomic.Int32
	creds := bedrock.CredentialsFunc(func(context.Context) (bedrock.Keys, error) {
		n := calls.Add(1)
		return bedrock.Keys{AccessKeyID: "AKID" + strconv.Itoa(int(n)), SecretAccessKey: "s"}, nil
	})
	p := f.provider(bedrock.WithCredentials(creds))
	for range 2 {
		if _, err := ai.GenerateText(t.Context(), p.Model(testModel), hello()); err != nil {
			t.Fatalf("GenerateText() error = %v", err)
		}
	}
	if got := f.last(t).Header.Get("Authorization"); !strings.Contains(got, "Credential=AKID2/") {
		t.Errorf("Authorization = %q, want the second retrieval AKID2", got)
	}
}

func TestProvider_credentialsError(t *testing.T) {
	errSTS := errors.New("bedrock: sts unavailable")
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithCredentials(bedrock.CredentialsFunc(func(context.Context) (bedrock.Keys, error) { return bedrock.Keys{}, errSTS })))
	_, err := ai.GenerateText(t.Context(), p.Model(testModel), noRetries(hello()))
	if !errors.Is(err, errSTS) {
		t.Errorf("GenerateText() error = %v, want %v", err, errSTS)
	}
}

func TestProvider_headersAndLimiter(t *testing.T) {
	f := newFakeServer(t, jsonReply(minimalResponse))
	var waited atomic.Int32
	p := f.provider(
		bedrock.WithHeaders(http.Header{"x-team": {"search"}}),
		bedrock.WithLimiter(limiterFunc(func(context.Context) error { waited.Add(1); return nil })),
	)
	if _, err := ai.GenerateText(t.Context(), p.Model(testModel), hello()); err != nil {
		t.Fatalf("GenerateText() error = %v", err)
	}
	if got := f.last(t).Header.Get("X-Team"); got != "search" || waited.Load() != 1 {
		t.Errorf("X-Team = %q, limiter waits = %d, want search and 1", got, waited.Load())
	}
}

func TestProvider_LanguageModel(t *testing.T) {
	m, err := bedrock.New().LanguageModel("amazon.nova-pro-v1:0")
	if err != nil || m.Provider() != "bedrock" || m.ModelID() != "amazon.nova-pro-v1:0" {
		t.Errorf("LanguageModel() = %v, %v, want bedrock amazon.nova-pro-v1:0", m, err)
	}
}

func TestKeys_environment(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "A")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "B")
	t.Setenv("AWS_SESSION_TOKEN", "C")
	got, err := bedrock.EnvCredentials().Retrieve(t.Context())
	if err != nil || got != (bedrock.Keys{AccessKeyID: "A", SecretAccessKey: "B", SessionToken: "C"}) {
		t.Errorf("EnvCredentials().Retrieve() = %+v, %v", got, err)
	}
}
