package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type FileUpload struct {
	Name         string
	MediaType    string
	Data         io.Reader
	Purpose      string
	ExpiresAfter time.Duration
	MaxRetries   *int
	Headers      http.Header
	Provider     ProviderData
}

type FileListRequest struct {
	Limit      int
	After      string
	Purpose    string
	MaxRetries *int
	Headers    http.Header
	Provider   ProviderData
}

// UploadFile stores a file with the provider behind model, which may be a model or a provider that implements Files; wrappers are searched with As. The returned File holds the provider's file id in Provider and no Data. A failed upload is retried only when Data is an io.Seeker.
func UploadFile(ctx context.Context, model any, up FileUpload) (File, error) {
	files, err := filesOf(model)
	if err != nil {
		return File{}, err
	}
	if up.Data == nil {
		return File{}, errors.New("ai: file upload: no data")
	}
	call := &FileUploadCall{
		Name:         up.Name,
		MediaType:    up.MediaType,
		Data:         up.Data,
		Purpose:      up.Purpose,
		ExpiresAfter: up.ExpiresAfter,
		Headers:      up.Headers.Clone(),
		Provider:     up.Provider.clone(),
	}
	rewind := rewinder(up.Data)
	attempts := 0
	canRetry := func(err error) bool { return rewind != nil && isRetryable(err) }
	f, err := withRetry(ctx, retriesOf(up.MaxRetries), canRetry, func(ctx context.Context) (*File, error) {
		if attempts++; attempts > 1 {
			if err := rewind(); err != nil {
				return nil, fmt.Errorf("ai: file upload: rewind data: %w", err)
			}
		}
		return files.UploadFile(ctx, call)
	})
	return fileResult(f, err)
}

func GetFile(ctx context.Context, model any, id string) (File, error) {
	files, err := filesOf(model)
	if err != nil {
		return File{}, err
	}
	f, err := withRetry(ctx, defaultMaxRetries, isRetryable, func(ctx context.Context) (*File, error) {
		return files.GetFile(ctx, id)
	})
	return fileResult(f, err)
}

// ListFiles returns one page; pass FileList.Next as After for the next. Next is empty on the last page.
func ListFiles(ctx context.Context, model any, req FileListRequest) (*FileList, error) {
	files, err := filesOf(model)
	if err != nil {
		return nil, err
	}
	call := &FileListCall{Limit: req.Limit, After: req.After, Purpose: req.Purpose, Headers: req.Headers.Clone(), Provider: req.Provider.clone()}
	list, err := withRetry(ctx, retriesOf(req.MaxRetries), isRetryable, func(ctx context.Context) (*FileList, error) {
		return files.ListFiles(ctx, call)
	})
	if err != nil {
		return nil, err
	}
	if list == nil {
		return nil, errors.New("ai: file list: provider returned no list")
	}
	return list, nil
}

func DeleteFile(ctx context.Context, model any, id string) error {
	files, err := filesOf(model)
	if err != nil {
		return err
	}
	_, err = withRetry(ctx, defaultMaxRetries, isRetryable, func(ctx context.Context) (struct{}, error) {
		return struct{}{}, files.DeleteFile(ctx, id)
	})
	return err
}

func filesOf(model any) (Files, error) {
	files, ok := As[Files](model)
	if !ok {
		return nil, fmt.Errorf("%w: %T does not store files", ErrUnsupported, model)
	}
	return files, nil
}

func fileResult(f *File, err error) (File, error) {
	if err != nil {
		return File{}, err
	}
	if f == nil {
		return File{}, errors.New("ai: file: provider returned no file")
	}
	return *f, nil
}
