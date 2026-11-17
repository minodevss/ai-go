package anthropic

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxFileBytes   = 500 << 20
	minFileExpiry  = time.Hour
	maxFileExpiry  = 90 * 24 * time.Hour
	maxFileListing = 1000
)

type filesAPI struct {
	client *transport.Client
}

type fileBody struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
	MIMEType string `json:"mime_type"`
}

type fileListBody struct {
	Data     []fileBody `json:"data"`
	NextPage *string    `json:"next_page"`
}

type fileDeleted struct {
	ID string `json:"id"`
}

// FileIDOf returns the file id held by a File that UploadFile, GetFile, ListFiles or FileID produced, and false if it holds none.
func FileIDOf(f ai.File) (string, bool) {
	id := dataOf(f.Provider).FileID
	return id, id != ""
}

// UploadFile stores a file with the Files API, which needs no beta header. Files are limited to 500 MB, ExpiresAfter must be between one hour and 90 days when set, and Purpose does not apply.
func (p *Provider) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	return filesAPI{p.client}.upload(ctx, c)
}

func (p *Provider) GetFile(ctx context.Context, id string) (*ai.File, error) {
	return filesAPI{p.client}.get(ctx, id)
}

// ListFiles returns files newest first; Limit is 1 to 1000 and After is the cursor of the previous page.
func (p *Provider) ListFiles(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	return filesAPI{p.client}.list(ctx, c)
}

func (p *Provider) DeleteFile(ctx context.Context, id string) error {
	return filesAPI{p.client}.remove(ctx, id)
}

func (m *LanguageModel) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	return filesAPI{m.client}.upload(ctx, c)
}

func (m *LanguageModel) GetFile(ctx context.Context, id string) (*ai.File, error) {
	return filesAPI{m.client}.get(ctx, id)
}

func (m *LanguageModel) ListFiles(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	return filesAPI{m.client}.list(ctx, c)
}

func (m *LanguageModel) DeleteFile(ctx context.Context, id string) error {
	return filesAPI{m.client}.remove(ctx, id)
}

func (a filesAPI) upload(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	if c.ExpiresAfter != 0 && (c.ExpiresAfter < minFileExpiry || c.ExpiresAfter > maxFileExpiry) {
		return nil, fmt.Errorf("anthropic: file expiry %s is outside %s to %s", c.ExpiresAfter, minFileExpiry, maxFileExpiry)
	}
	data, err := io.ReadAll(io.LimitReader(c.Data, maxFileBytes+1))
	if err != nil {
		return nil, fmt.Errorf("anthropic: read file: %w", err)
	}
	if len(data) > maxFileBytes {
		return nil, fmt.Errorf("anthropic: file exceeds the %d MB limit", maxFileBytes>>20)
	}
	f := form.New()
	f.File("file", cmp.Or(c.Name, "file"), c.MediaType, bytes.NewReader(data))
	if c.ExpiresAfter > 0 {
		f.Field("expires_in_seconds", strconv.Itoa(int(c.ExpiresAfter/time.Second)))
	}
	req, err := f.Request("/v1/files")
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	req.Header = c.Headers
	var body fileBody
	if _, err := a.client.JSON(ctx, req, &body); err != nil {
		return nil, err
	}
	return fileOf(body, c.MediaType), nil
}

func (a filesAPI) get(ctx context.Context, id string) (*ai.File, error) {
	if err := checkID(id); err != nil {
		return nil, err
	}
	var body fileBody
	if _, err := a.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/v1/files/" + url.PathEscape(id)}, &body); err != nil {
		return nil, err
	}
	return fileOf(body, ""), nil
}

func (a filesAPI) list(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	if c.Limit > maxFileListing {
		return nil, fmt.Errorf("anthropic: file list limit %d exceeds %d", c.Limit, maxFileListing)
	}
	q := url.Values{}
	if c.Limit > 0 {
		q.Set("limit", strconv.Itoa(c.Limit))
	}
	if c.After != "" {
		q.Set("page", c.After)
	}
	path := "/v1/files"
	if len(q) > 0 {
		path += "?" + q.Encode()
	}
	var body fileListBody
	if _, err := a.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: path, Header: c.Headers}, &body); err != nil {
		return nil, err
	}
	list := &ai.FileList{Files: make([]ai.File, len(body.Data))}
	for i, b := range body.Data {
		list.Files[i] = *fileOf(b, "")
	}
	if body.NextPage != nil {
		list.Next = *body.NextPage
	}
	return list, nil
}

func (a filesAPI) remove(ctx context.Context, id string) error {
	if err := checkID(id); err != nil {
		return err
	}
	var body fileDeleted
	if _, err := a.client.JSON(ctx, transport.Request{Method: http.MethodDelete, Path: "/v1/files/" + url.PathEscape(id)}, &body); err != nil {
		return err
	}
	if body.ID != id {
		return fmt.Errorf("anthropic: file %s was not deleted", id)
	}
	return nil
}

func fileOf(b fileBody, mediaType string) *ai.File {
	return &ai.File{Filename: b.Filename, MediaType: cmp.Or(b.MIMEType, mediaType), Provider: FileID(b.ID)}
}

var (
	_ ai.Files = (*Provider)(nil)
	_ ai.Files = (*LanguageModel)(nil)
)
