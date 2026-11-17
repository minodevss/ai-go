package aitest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strconv"
	"sync"

	"github.com/minodevss/ai-go"
)

const fileIDKey = "aitest"

// Files is an in-memory ai.Files. File ids are in Provider["aitest"] as {"id":"..."}.
type Files struct {
	mu    sync.Mutex
	files map[string]storedFile
	order []string
}

type storedFile struct {
	name      string
	mediaType string
	data      []byte
	purpose   string
}

func NewFiles() *Files { return &Files{files: map[string]storedFile{}} }

func FileID(f ai.File) string {
	v, _ := ai.ProviderValue[struct {
		ID string `json:"id"`
	}](f.Provider, fileIDKey)
	return v.ID
}

func (s *Files) Data(id string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, ok := s.files[id]
	return slices.Clone(f.data), ok
}

func (s *Files) UploadFile(_ context.Context, c *ai.FileUploadCall) (*ai.File, error) {
	data, err := io.ReadAll(c.Data)
	if err != nil {
		return nil, fmt.Errorf("aitest: read file: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := "file_" + strconv.Itoa(len(s.order)+1)
	s.files[id] = storedFile{name: c.Name, mediaType: c.MediaType, data: data, purpose: c.Purpose}
	s.order = append(s.order, id)
	return s.describe(id), nil
}

func (s *Files) GetFile(_ context.Context, id string) (*ai.File, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.files[id]; !ok {
		return nil, &ai.APIError{Provider: "aitest", StatusCode: 404, Message: "no such file"}
	}
	return s.describe(id), nil
}

func (s *Files) ListFiles(_ context.Context, c *ai.FileListCall) (*ai.FileList, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := s.order
	if c.After != "" {
		i := slices.Index(ids, c.After)
		if i < 0 {
			return nil, &ai.APIError{Provider: "aitest", StatusCode: 404, Message: "no such cursor"}
		}
		ids = ids[i+1:]
	}
	list := &ai.FileList{}
	for _, id := range ids {
		if c.Purpose != "" && s.files[id].purpose != c.Purpose {
			continue
		}
		if c.Limit > 0 && len(list.Files) == c.Limit {
			list.Next = FileID(list.Files[len(list.Files)-1])
			break
		}
		list.Files = append(list.Files, *s.describe(id))
	}
	return list, nil
}

func (s *Files) DeleteFile(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.files[id]; !ok {
		return &ai.APIError{Provider: "aitest", StatusCode: 404, Message: "no such file"}
	}
	delete(s.files, id)
	s.order = slices.DeleteFunc(s.order, func(o string) bool { return o == id })
	return nil
}

func (s *Files) describe(id string) *ai.File {
	f := s.files[id]
	return &ai.File{Filename: f.name, MediaType: f.mediaType, Provider: ai.ProviderData{}.WithRaw(fileIDKey, json.RawMessage(`{"id":`+strconv.Quote(id)+`}`))}
}

var _ ai.Files = (*Files)(nil)
