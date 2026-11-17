package google

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxFileBytes     = 2 << 30
	maxFilesPageSize = 100
	filePollInterval = 2 * time.Second
	defaultFileType  = "application/octet-stream"
	versionSuffix    = "/v1beta"
	filesPrefix      = "files/"
)

var validFileID = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?$`)

type filesAPI struct {
	b *backend
}

type fileBody struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	MimeType    string `json:"mimeType"`
	URI         string `json:"uri"`
	State       string `json:"state"`
	Error       *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type fileEnvelope struct {
	File fileBody `json:"file"`
}

type fileListBody struct {
	Files         []fileBody `json:"files"`
	NextPageToken string     `json:"nextPageToken"`
}

type fileRef struct {
	Name string `json:"name"`
}

type fileCreateBody struct {
	File struct {
		DisplayName string `json:"displayName,omitzero"`
	} `json:"file"`
}

// FileID references a file stored with the Files API instead of sending its bytes. Set MediaType on the result when it is known.
func FileID(id string) ai.File {
	name := filesPrefix + strings.TrimPrefix(id, filesPrefix)
	return ai.File{URL: defaultBaseURL + "/" + name, Provider: pdata.Encode(providerName, fileRef{Name: name})}
}

// FileIDOf returns the file id held by a File that UploadFile, GetFile, ListFiles or FileID produced.
func FileIDOf(f ai.File) (string, bool) {
	id := strings.TrimPrefix(fileName(f), filesPrefix)
	return id, id != ""
}

func fileName(f ai.File) string {
	return pdata.Decode[fileRef](providerName, f.Provider).Name
}

func (p *Provider) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	return filesAPI{p.backend}.upload(ctx, c)
}

func (p *Provider) GetFile(ctx context.Context, id string) (*ai.File, error) {
	return filesAPI{p.backend}.get(ctx, id)
}

func (p *Provider) ListFiles(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	return filesAPI{p.backend}.list(ctx, c)
}

func (p *Provider) DeleteFile(ctx context.Context, id string) error {
	return filesAPI{p.backend}.remove(ctx, id)
}

func (m *LanguageModel) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	return filesAPI{m.backend}.upload(ctx, c)
}

func (m *LanguageModel) GetFile(ctx context.Context, id string) (*ai.File, error) {
	return filesAPI{m.backend}.get(ctx, id)
}

func (m *LanguageModel) ListFiles(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	return filesAPI{m.backend}.list(ctx, c)
}

func (m *LanguageModel) DeleteFile(ctx context.Context, id string) error {
	return filesAPI{m.backend}.remove(ctx, id)
}

func (a filesAPI) upload(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	if err := a.supported(); err != nil {
		return nil, err
	}
	if c.ExpiresAfter != 0 {
		return nil, errors.New("google: files expire 48 hours after upload and the lifetime cannot be changed")
	}
	data, err := io.ReadAll(io.LimitReader(c.Data, maxFileBytes+1))
	if err != nil {
		return nil, fmt.Errorf("google: read file: %w", err)
	}
	if len(data) > maxFileBytes {
		return nil, fmt.Errorf("google: file exceeds the %d GB limit", maxFileBytes>>30)
	}
	mediaType := cmp.Or(c.MediaType, defaultFileType)
	var meta fileCreateBody
	meta.File.DisplayName = c.Name
	start := http.Header{
		"Content-Type":                        {"application/json"},
		"X-Goog-Upload-Protocol":              {"resumable"},
		"X-Goog-Upload-Command":               {"start"},
		"X-Goog-Upload-Header-Content-Length": {strconv.Itoa(len(data))},
		"X-Goog-Upload-Header-Content-Type":   {mediaType},
	}
	body, err := transport.EncodeJSON(meta)
	if err != nil {
		return nil, fmt.Errorf("google: encode file metadata: %w", err)
	}
	resp, err := a.send(ctx, a.root()+"/upload"+versionSuffix+"/files", start, c.Headers, bytes.NewReader(body), true)
	if err != nil {
		return nil, err
	}
	uploadURL := resp.Header.Get("X-Goog-Upload-Url")
	if err := resp.Body.Close(); err != nil {
		return nil, fmt.Errorf("google: close upload response: %w", err)
	}
	if uploadURL == "" {
		return nil, errors.New("google: upload start returned no upload url")
	}
	finalize := http.Header{
		"X-Goog-Upload-Offset":  {"0"},
		"X-Goog-Upload-Command": {"upload, finalize"},
	}
	resp, err = a.send(ctx, uploadURL, finalize, nil, bytes.NewReader(data), false)
	if err != nil {
		return nil, err
	}
	var env fileEnvelope
	if err := decodeBody(resp, &env); err != nil {
		return nil, err
	}
	return a.ready(ctx, env.File, mediaType)
}

func (a filesAPI) ready(ctx context.Context, f fileBody, mediaType string) (*ai.File, error) {
	for f.State == "PROCESSING" {
		timer := time.NewTimer(filePollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		next, err := a.fetch(ctx, f.Name)
		if err != nil {
			return nil, err
		}
		f = next
	}
	if f.State == "FAILED" {
		msg := "processing failed"
		if f.Error != nil && f.Error.Message != "" {
			msg = f.Error.Message
		}
		return nil, fmt.Errorf("google: file %s: %s", f.Name, msg)
	}
	return fileOf(f, mediaType), nil
}

func (a filesAPI) get(ctx context.Context, id string) (*ai.File, error) {
	if err := a.supported(); err != nil {
		return nil, err
	}
	f, err := a.fetch(ctx, id)
	if err != nil {
		return nil, err
	}
	return fileOf(f, ""), nil
}

func (a filesAPI) fetch(ctx context.Context, id string) (fileBody, error) {
	var f fileBody
	name, err := fileResource(id)
	if err != nil {
		return f, err
	}
	_, err = a.b.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/" + name}, &f)
	return f, fail(err)
}

func (a filesAPI) list(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	if err := a.supported(); err != nil {
		return nil, err
	}
	q := url.Values{}
	if c.Limit > 0 {
		q.Set("pageSize", strconv.Itoa(min(c.Limit, maxFilesPageSize)))
	}
	if c.After != "" {
		q.Set("pageToken", c.After)
	}
	path := "/files"
	if len(q) > 0 {
		path += "?" + q.Encode()
	}
	var body fileListBody
	if _, err := a.b.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: path, Header: c.Headers}, &body); err != nil {
		return nil, fail(err)
	}
	list := &ai.FileList{Files: make([]ai.File, len(body.Files)), Next: body.NextPageToken}
	for i, f := range body.Files {
		list.Files[i] = *fileOf(f, "")
	}
	return list, nil
}

func (a filesAPI) remove(ctx context.Context, id string) error {
	if err := a.supported(); err != nil {
		return err
	}
	name, err := fileResource(id)
	if err != nil {
		return err
	}
	var empty json.RawMessage
	_, err = a.b.client.JSON(ctx, transport.Request{Method: http.MethodDelete, Path: "/" + name}, &empty)
	return fail(err)
}

func (a filesAPI) supported() error {
	if a.b.vertex {
		return fmt.Errorf("%w: the Gemini Files API is not part of Vertex AI", ai.ErrUnsupported)
	}
	return nil
}

func (a filesAPI) root() string {
	return strings.TrimSuffix(a.b.client.URL(""), versionSuffix)
}

func (a filesAPI) send(ctx context.Context, target string, header, extra http.Header, body io.Reader, authenticate bool) (*http.Response, error) {
	return a.b.send(ctx, http.MethodPost, target, header, extra, body, authenticate)
}

func (b *backend) send(ctx context.Context, method, target string, header, extra http.Header, body io.Reader, authenticate bool) (*http.Response, error) {
	if err := b.client.Wait(ctx); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return nil, fmt.Errorf("google: build request: %w", err)
	}
	if authenticate {
		maps.Copy(req.Header, b.client.Headers())
	}
	for _, h := range []http.Header{extra, header} {
		for k, v := range h {
			req.Header[http.CanonicalHeaderKey(k)] = append([]string(nil), v...)
		}
	}
	resp, err := b.client.HTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("google: send request: %w", err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		raw = nil
	}
	return nil, fail(b.client.APIError(resp, raw))
}

func decodeBody(resp *http.Response, out any) error {
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return fmt.Errorf("google: read response: %w", err)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("google: decode response: %w", err)
	}
	return nil
}

func fileResource(id string) (string, error) {
	bare := strings.TrimPrefix(id, filesPrefix)
	if !validFileID.MatchString(bare) {
		return "", fmt.Errorf("google: invalid file id %q", id)
	}
	return filesPrefix + bare, nil
}

func fileOf(b fileBody, mediaType string) *ai.File {
	return &ai.File{
		Filename:  b.DisplayName,
		MediaType: cmp.Or(b.MimeType, mediaType),
		URL:       b.URI,
		Provider:  pdata.Encode(providerName, fileRef{Name: b.Name}),
	}
}

var (
	_ ai.Files = (*Provider)(nil)
	_ ai.Files = (*LanguageModel)(nil)
)
