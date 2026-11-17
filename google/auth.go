package google

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
	defaultTokenURI    = "https://oauth2.googleapis.com/token"
	defaultMetadataURL = "metadata.google.internal"
	tokenRefreshMargin = time.Minute
	maxTokenBody       = 1 << 20
)

// TokenSource returns a current OAuth 2.0 access token and must be safe for concurrent use.
type TokenSource func(ctx context.Context) (string, error)

type serviceAccountKey struct {
	Type         string `json:"type"`
	ClientEmail  string `json:"client_email"`
	PrivateKey   string `json:"private_key"`
	PrivateKeyID string `json:"private_key_id"`
	TokenURI     string `json:"token_uri"`
}

type tokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

type bearerTransport struct {
	base   http.RoundTripper
	tokens TokenSource
	host   string
	mu     *sync.Mutex
}

type tokenCache struct {
	mu     sync.Mutex
	fetch  func(ctx context.Context) (tokenResponse, error)
	token  string
	expiry time.Time
}

func bearerClient(c *http.Client, ts TokenSource) *http.Client {
	var out http.Client
	if c != nil {
		out = *c
	}
	if ts == nil {
		ts = DefaultTokenSource(c)
	}
	base := out.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	out.Transport = &bearerTransport{base: base, tokens: ts, mu: new(sync.Mutex)}
	return &out
}

func (t *bearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !t.trusted(req) {
		return t.base.RoundTrip(req)
	}
	token, err := t.tokens(req.Context())
	if err != nil {
		return nil, fmt.Errorf("google: get access token: %w", err)
	}
	out := req.Clone(req.Context())
	out.Header.Set("Authorization", "Bearer "+token)
	return t.base.RoundTrip(out)
}

func (t *bearerTransport) trusted(req *http.Request) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.host == "" {
		t.host = req.URL.Host
	}
	return req.URL.Host == t.host
}

// MetadataTokenSource fetches tokens from the Google Cloud metadata server; GCE_METADATA_HOST overrides its host.
func MetadataTokenSource(c *http.Client) TokenSource {
	cache := &tokenCache{fetch: func(ctx context.Context) (tokenResponse, error) {
		host := os.Getenv("GCE_METADATA_HOST")
		if host == "" {
			host = defaultMetadataURL
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+host+"/computeMetadata/v1/instance/service-accounts/default/token", nil)
		if err != nil {
			return tokenResponse{}, err
		}
		req.Header.Set("Metadata-Flavor", "Google")
		return doTokenRequest(c, req)
	}}
	return cache.get
}

// ServiceAccountTokenSource exchanges a JWT signed with a service account key for access tokens.
func ServiceAccountTokenSource(keyJSON []byte, c *http.Client) (TokenSource, error) {
	var key serviceAccountKey
	if err := json.Unmarshal(keyJSON, &key); err != nil {
		return nil, fmt.Errorf("google: decode service account key: %w", err)
	}
	if key.Type != "service_account" {
		return nil, fmt.Errorf("google: credentials of type %q are not supported, want service_account", key.Type)
	}
	priv, err := parsePrivateKey(key.PrivateKey)
	if err != nil {
		return nil, err
	}
	if key.TokenURI == "" {
		key.TokenURI = defaultTokenURI
	}
	cache := &tokenCache{fetch: func(ctx context.Context) (tokenResponse, error) {
		assertion, err := signJWT(key, priv, time.Now())
		if err != nil {
			return tokenResponse{}, err
		}
		form := url.Values{"grant_type": {"urn:ietf:params:oauth:grant-type:jwt-bearer"}, "assertion": {assertion}}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, key.TokenURI, strings.NewReader(form.Encode()))
		if err != nil {
			return tokenResponse{}, err
		}
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		return doTokenRequest(c, req)
	}}
	return cache.get, nil
}

// DefaultTokenSource uses the key file named by GOOGLE_APPLICATION_CREDENTIALS, or else the metadata server.
func DefaultTokenSource(c *http.Client) TokenSource {
	var (
		mu  sync.Mutex
		src TokenSource
	)
	resolve := func() (TokenSource, error) {
		mu.Lock()
		defer mu.Unlock()
		if src != nil {
			return src, nil
		}
		path := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
		if path == "" {
			src = MetadataTokenSource(c)
			return src, nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("google: read credentials: %w", err)
		}
		next, err := ServiceAccountTokenSource(raw, c)
		if err != nil {
			return nil, err
		}
		src = next
		return src, nil
	}
	return func(ctx context.Context) (string, error) {
		s, err := resolve()
		if err != nil {
			return "", err
		}
		return s(ctx)
	}
}

func (tc *tokenCache) get(ctx context.Context) (string, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	if tc.token != "" && time.Until(tc.expiry) > tokenRefreshMargin {
		return tc.token, nil
	}
	res, err := tc.fetch(ctx)
	if err != nil {
		return "", err
	}
	tc.token = res.AccessToken
	tc.expiry = time.Now().Add(time.Duration(res.ExpiresIn) * time.Second)
	return tc.token, nil
}

func doTokenRequest(c *http.Client, req *http.Request) (tokenResponse, error) {
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return tokenResponse{}, fmt.Errorf("google: request token: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxTokenBody))
	if err != nil {
		return tokenResponse{}, fmt.Errorf("google: read token response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return tokenResponse{}, fmt.Errorf("google: token endpoint returned status %d", resp.StatusCode)
	}
	var out tokenResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return tokenResponse{}, fmt.Errorf("google: decode token response: %w", err)
	}
	if out.AccessToken == "" {
		return tokenResponse{}, errors.New("google: token response has no access token")
	}
	return out, nil
}

func parsePrivateKey(pemText string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(pemText))
	if block == nil {
		return nil, errors.New("google: service account private key is not PEM")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("google: parse service account private key: %w", err)
	}
	priv, ok := parsed.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("google: service account private key is not RSA")
	}
	return priv, nil
}

func signJWT(key serviceAccountKey, priv *rsa.PrivateKey, now time.Time) (string, error) {
	header := map[string]string{"alg": "RS256", "typ": "JWT"}
	if key.PrivateKeyID != "" {
		header["kid"] = key.PrivateKeyID
	}
	claims := map[string]any{
		"iss":   key.ClientEmail,
		"scope": cloudPlatformScope,
		"aud":   key.TokenURI,
		"iat":   now.Unix(),
		"exp":   now.Add(time.Hour).Unix(),
	}
	h, err := json.Marshal(header)
	if err != nil {
		return "", fmt.Errorf("google: encode jwt header: %w", err)
	}
	c, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("google: encode jwt claims: %w", err)
	}
	signing := base64.RawURLEncoding.EncodeToString(h) + "." + base64.RawURLEncoding.EncodeToString(c)
	digest := sha256.Sum256([]byte(signing))
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, digest[:])
	if err != nil {
		return "", fmt.Errorf("google: sign jwt: %w", err)
	}
	return signing + "." + base64.RawURLEncoding.EncodeToString(sig), nil
}
