package ollama

import "testing"

func TestNormalizeHost(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{name: "empty", raw: "", want: "http://localhost:11434"},
		{name: "bare host", raw: "example.com", want: "http://example.com:11434"},
		{name: "host and port", raw: "0.0.0.0:11434", want: "http://0.0.0.0:11434"},
		{name: "custom port", raw: "10.0.0.5:8080", want: "http://10.0.0.5:8080"},
		{name: "http url", raw: "http://gpu-box", want: "http://gpu-box:11434"},
		{name: "https default port", raw: "https://ollama.com", want: "https://ollama.com:443"},
		{name: "https with port", raw: "https://ollama.example:8443/", want: "https://ollama.example:8443"},
		{name: "ipv6", raw: "[::1]", want: "http://[::1]:11434"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeHost(tt.raw); got != tt.want {
				t.Errorf("normalizeHost(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}
