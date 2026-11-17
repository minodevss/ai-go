package openai

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxFileBytes   = 512 << 20
	defaultPurpose = "user_data"
)

type filesAPI struct {
	client *transport.Client
}

type fileBody struct {
	ID       string `json:"id"`
	Filename string `json:"filename"`
}

type fileListBody struct {
	Data    []fileBody `json:"data"`
	HasMore bool       `json:"has_more"`
	LastID  string     `json:"last_id"`
}

type fileDeleted struct {
	Deleted bool `json:"deleted"`
}

func (p *Provider) UploadFile(ctx context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	return filesAPI{p.client}.upload(ctx, c)
}

func (p *Provider) GetFile(ctx context.Context, id string) (*ai.File, error) {
	return filesAPI{p.client}.get(ctx, id)
}

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
	data, err := readLimited(c.Data, maxFileBytes, "file")
	if err != nil {
		return nil, err
	}
	return a.uploadBytes(ctx, c, data)
}

func (a filesAPI) uploadBytes(ctx context.Context, c *ai.FileUploadCall, data []byte) (*ai.File, error) {
	f := form.New()
	f.File("file", cmp.Or(c.Name, "file"), c.MediaType, bytes.NewReader(data))
	f.Field("purpose", cmp.Or(c.Purpose, defaultPurpose))
	if c.ExpiresAfter > 0 {
		f.Field("expires_after[anchor]", "created_at")
		f.Field("expires_after[seconds]", strconv.Itoa(int(c.ExpiresAfter/time.Second)))
	}
	req, err := f.Request("/files")
	if err != nil {
		return nil, err
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
	if _, err := a.client.JSON(ctx, transport.Request{Method: http.MethodGet, Path: "/files/" + escapeID(id)}, &body); err != nil {
		return nil, err
	}
	return fileOf(body, ""), nil
}

func (a filesAPI) list(ctx context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	q := url.Values{}
	if c.Limit > 0 {
		q.Set("limit", strconv.Itoa(c.Limit))
	}
	if c.After != "" {
		q.Set("after", c.After)
	}
	if c.Purpose != "" {
		q.Set("purpose", c.Purpose)
	}
	path := "/files"
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
	if body.HasMore {
		list.Next = body.LastID
	}
	return list, nil
}

func (a filesAPI) remove(ctx context.Context, id string) error {
	if err := checkID(id); err != nil {
		return err
	}
	var body fileDeleted
	if _, err := a.client.JSON(ctx, transport.Request{Method: http.MethodDelete, Path: "/files/" + escapeID(id)}, &body); err != nil {
		return err
	}
	if !body.Deleted {
		return fmt.Errorf("openai: file %s was not deleted", id)
	}
	return nil
}

func fileOf(b fileBody, mediaType string) *ai.File {
	return &ai.File{Filename: b.Filename, MediaType: mediaType, Provider: FileID(b.ID)}
}

var (
	_ ai.Files = (*Provider)(nil)
	_ ai.Files = (*LanguageModel)(nil)
)

func escapeID(id string) string { return url.PathEscape(id) }

func checkID(id string) error {
	if id == "" || id == "." || id == ".." {
		return fmt.Errorf("openai: invalid id %q", id)
	}
	return nil
}
