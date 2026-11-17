// Package mediaio downloads generated media and encodes input media for the hosted generation providers.
package mediaio

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	MaxImageBytes = 64 << 20
	MaxVideoBytes = 512 << 20
	MaxInputBytes = 20 << 20
)

type Downloader struct {
	Provider string
	Client   *transport.Client
}

// Download fetches a result URL without the provider's credentials, because result hosts are not the API host.
// Only https URLs that resolve to public addresses are fetched, on every redirect hop, unless the client allows unrestricted downloads.
func (d Downloader) Download(ctx context.Context, rawURL string, fallbackType string, limit int64) (ai.File, error) {
	u, err := url.Parse(rawURL)
	unrestricted := d.Client.UnrestrictedDownloads()
	if err != nil || u.Host == "" || u.User != nil || (u.Scheme != "https" && (!unrestricted || u.Scheme != "http")) {
		return ai.File{}, fmt.Errorf("%s: result url %q is not an https url", d.Provider, rawURL)
	}
	if err := d.Client.Wait(ctx); err != nil {
		return ai.File{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, http.NoBody)
	if err != nil {
		return ai.File{}, fmt.Errorf("%s: build download request: %w", d.Provider, err)
	}
	client := d.Client.HTTPClient()
	if !unrestricted {
		var closeIdle func()
		client, closeIdle = publicClient()
		defer closeIdle()
	}
	resp, err := client.Do(req)
	if err != nil {
		return ai.File{}, fmt.Errorf("%s: download result: %w", d.Provider, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<10))
		return ai.File{}, &ai.APIError{Provider: d.Provider, StatusCode: resp.StatusCode, Code: "download_failed", Message: "result download failed", Body: string(body)}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return ai.File{}, fmt.Errorf("%s: read result: %w", d.Provider, err)
	}
	if int64(len(data)) > limit {
		return ai.File{}, fmt.Errorf("%s: result exceeds the %d MB limit", d.Provider, limit>>20)
	}
	mediaType := fallbackType
	if ct, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type")); err == nil && ct != "application/octet-stream" && ct != "binary/octet-stream" {
		mediaType = ct
	}
	return ai.File{MediaType: mediaType, Data: data}, nil
}

// Result is a generated file: downloaded into Data, or left as a URL when keepURL is set.
func (d Downloader) Result(ctx context.Context, rawURL, fallbackType string, limit int64, keepURL bool) (ai.File, error) {
	if keepURL {
		return ai.File{MediaType: fallbackType, URL: rawURL}, nil
	}
	return d.Download(ctx, rawURL, fallbackType, limit)
}

// Input returns the URL form of an input file: its URL when set, otherwise a base64 data URI.
func Input(provider string, f ai.File) (string, error) {
	switch {
	case f.URL != "":
		return f.URL, nil
	case len(f.Data) == 0:
		return "", fmt.Errorf("%s: input file %q has neither data nor url", provider, f.Filename)
	case len(f.Data) > MaxInputBytes:
		return "", fmt.Errorf("%s: input file exceeds %d MB", provider, MaxInputBytes>>20)
	}
	mediaType := f.MediaType
	if mediaType == "" {
		mediaType = http.DetectContentType(f.Data)
	}
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(f.Data), nil
}

// Base64 returns the raw base64 of an input file that has Data.
func Base64(provider string, f ai.File) (string, error) {
	if len(f.Data) == 0 {
		return "", fmt.Errorf("%s: input file %q needs Data", provider, f.Filename)
	}
	if len(f.Data) > MaxInputBytes {
		return "", fmt.Errorf("%s: input file exceeds %d MB", provider, MaxInputBytes>>20)
	}
	return base64.StdEncoding.EncodeToString(f.Data), nil
}

func ImageType(format string) string {
	switch strings.ToLower(format) {
	case "jpeg", "jpg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	case "gif":
		return "image/gif"
	default:
		return "image/png"
	}
}

func DecodeOptions[T any](provider string, d ai.ProviderData) (T, error) {
	var o T
	raw, ok := d[provider]
	if !ok {
		return o, nil
	}
	if err := json.Unmarshal(raw, &o); err != nil {
		return *new(T), fmt.Errorf("%s: decode provider options: %w", provider, err)
	}
	return o, nil
}

func AudioName(mediaType string) string {
	ext := ".mp3"
	switch base, _, _ := strings.Cut(mediaType, ";"); strings.TrimSpace(base) {
	case "audio/wav", "audio/x-wav", "audio/wave":
		ext = ".wav"
	case "audio/webm":
		ext = ".webm"
	case "audio/ogg":
		ext = ".ogg"
	case "audio/flac":
		ext = ".flac"
	case "audio/mp4", "audio/m4a", "audio/x-m4a":
		ext = ".m4a"
	case "audio/mpeg", "":
	default:
		if exts, err := mime.ExtensionsByType(mediaType); err == nil && len(exts) > 0 {
			ext = exts[0]
		}
	}
	return "audio" + ext
}
