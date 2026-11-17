package google_test

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/minodevss/ai-go/google"
)

func serviceAccountKey(t *testing.T, tokenURI string) ([]byte, *rsa.PublicKey) {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatalf("MarshalPKCS8PrivateKey() error = %v", err)
	}
	key, err := json.Marshal(map[string]string{
		"type":           "service_account",
		"client_email":   "bot@proj.iam.gserviceaccount.com",
		"private_key":    string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"private_key_id": "kid-1",
		"token_uri":      tokenURI,
	})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	return key, &priv.PublicKey
}

func tokenServer(t *testing.T, pub *rsa.PublicKey, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if err := r.ParseForm(); err != nil {
			t.Errorf("ParseForm() error = %v", err)
		}
		if got := r.Form.Get("grant_type"); got != "urn:ietf:params:oauth:grant-type:jwt-bearer" {
			t.Errorf("grant_type = %q", got)
		}
		checkJWT(t, r.Form.Get("assertion"), pub)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"ya29.sa","expires_in":3600,"token_type":"Bearer"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func checkJWT(t *testing.T, jwt string, pub *rsa.PublicKey) {
	t.Helper()
	segs := strings.Split(jwt, ".")
	if len(segs) != 3 {
		t.Fatalf("assertion has %d segments, want 3", len(segs))
	}
	var header map[string]string
	var claims map[string]any
	decodeSegment(t, segs[0], &header)
	decodeSegment(t, segs[1], &claims)
	if header["alg"] != "RS256" || header["typ"] != "JWT" || header["kid"] != "kid-1" {
		t.Errorf("header = %v", header)
	}
	iat, _ := claims["iat"].(float64)
	exp, _ := claims["exp"].(float64)
	if claims["iss"] != "bot@proj.iam.gserviceaccount.com" || claims["scope"] != "https://www.googleapis.com/auth/cloud-platform" || exp-iat != 3600 {
		t.Errorf("claims = %v", claims)
	}
	sig, err := base64.RawURLEncoding.DecodeString(segs[2])
	if err != nil {
		t.Fatalf("signature is not base64url: %v", err)
	}
	digest := sha256.Sum256([]byte(segs[0] + "." + segs[1]))
	if err := rsa.VerifyPKCS1v15(pub, crypto.SHA256, digest[:], sig); err != nil {
		t.Errorf("signature does not verify: %v", err)
	}
}

func decodeSegment(t *testing.T, seg string, out any) {
	t.Helper()
	raw, err := base64.RawURLEncoding.DecodeString(seg)
	if err != nil {
		t.Fatalf("segment is not base64url: %v", err)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("segment is not JSON: %v", err)
	}
}

func TestServiceAccountTokenSource(t *testing.T) {
	var calls atomic.Int32
	_, pub := serviceAccountKey(t, "")
	srv := tokenServer(t, pub, &calls)
	keyJSON, pub2 := serviceAccountKey(t, srv.URL)
	*pub = *pub2
	ts, err := google.ServiceAccountTokenSource(keyJSON, nil)
	if err != nil {
		t.Fatalf("ServiceAccountTokenSource() error = %v", err)
	}
	for range 3 {
		tok, err := ts(t.Context())
		if err != nil || tok != "ya29.sa" {
			t.Fatalf("token = %q, %v", tok, err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("token endpoint calls = %d, want 1 (cached)", got)
	}
}

func TestServiceAccountTokenSource_rejects(t *testing.T) {
	tests := []struct {
		name string
		key  string
	}{
		{name: "not json", key: `{`},
		{name: "authorized user", key: `{"type":"authorized_user"}`},
		{name: "bad pem", key: `{"type":"service_account","private_key":"nope"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := google.ServiceAccountTokenSource([]byte(tt.key), nil); err == nil {
				t.Error("ServiceAccountTokenSource() error = nil, want an error")
			}
		})
	}
}

func TestMetadataTokenSource(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/computeMetadata/v1/instance/service-accounts/default/token" || r.Header.Get("Metadata-Flavor") != "Google" {
			t.Errorf("request = %s with Metadata-Flavor %q", r.URL.Path, r.Header.Get("Metadata-Flavor"))
		}
		_, _ = w.Write([]byte(`{"access_token":"ya29.meta","expires_in":3599,"token_type":"Bearer"}`))
	}))
	t.Cleanup(srv.Close)
	t.Setenv("GCE_METADATA_HOST", strings.TrimPrefix(srv.URL, "http://"))
	ts := google.MetadataTokenSource(nil)
	for range 2 {
		if tok, err := ts(t.Context()); err != nil || tok != "ya29.meta" {
			t.Fatalf("token = %q, %v", tok, err)
		}
	}
	if calls.Load() != 1 {
		t.Errorf("metadata calls = %d, want 1 (cached)", calls.Load())
	}
}

func TestMetadataTokenSource_failure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNotFound) }))
	t.Cleanup(srv.Close)
	t.Setenv("GCE_METADATA_HOST", strings.TrimPrefix(srv.URL, "http://"))
	if _, err := google.MetadataTokenSource(nil)(t.Context()); err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("token error = %v, want status 404", err)
	}
}

func TestDefaultTokenSource(t *testing.T) {
	t.Run("credentials file", func(t *testing.T) {
		var calls atomic.Int32
		_, pub := serviceAccountKey(t, "")
		srv := tokenServer(t, pub, &calls)
		keyJSON, pub2 := serviceAccountKey(t, srv.URL)
		*pub = *pub2
		path := filepath.Join(t.TempDir(), "key.json")
		if err := os.WriteFile(path, keyJSON, 0o600); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path)
		if tok, err := google.DefaultTokenSource(nil)(t.Context()); err != nil || tok != "ya29.sa" {
			t.Errorf("token = %q, %v", tok, err)
		}
	})
	t.Run("missing file", func(t *testing.T) {
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "none.json"))
		if _, err := google.DefaultTokenSource(nil)(t.Context()); err == nil {
			t.Error("token error = nil, want a read error")
		}
	})
	t.Run("metadata server", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"access_token":"ya29.default","expires_in":3600}`))
		}))
		t.Cleanup(srv.Close)
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
		t.Setenv("GCE_METADATA_HOST", strings.TrimPrefix(srv.URL, "http://"))
		if tok, err := google.DefaultTokenSource(nil)(t.Context()); err != nil || tok != "ya29.default" {
			t.Errorf("token = %q, %v", tok, err)
		}
	})
}
