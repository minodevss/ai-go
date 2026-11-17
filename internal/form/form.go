// Package form builds multipart/form-data request bodies for provider adapters.
package form

import (
	"bytes"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/textproto"
	"strconv"
	"strings"

	"github.com/minodevss/ai-go/internal/transport"
)

type Form struct {
	buf bytes.Buffer
	w   *multipart.Writer
	err error
}

func New() *Form {
	f := &Form{}
	f.w = multipart.NewWriter(&f.buf)
	return f
}

func (f *Form) Field(name, value string) {
	if f.err != nil || value == "" {
		return
	}
	f.err = f.w.WriteField(name, value)
}

func (f *Form) Flag(name string, v *bool) {
	if v != nil {
		f.Field(name, strconv.FormatBool(*v))
	}
}

func (f *Form) File(field, name, mediaType string, data io.Reader) {
	if f.err != nil {
		return
	}
	if mediaType != "" {
		canonical, err := canonicalMediaType(mediaType)
		if err != nil {
			f.err = err
			return
		}
		mediaType = canonical
	}
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, escapeQuotes(field), escapeQuotes(name)))
	if mediaType != "" {
		h.Set("Content-Type", mediaType)
	}
	part, err := f.w.CreatePart(h)
	if err != nil {
		f.err = err
		return
	}
	_, f.err = io.Copy(part, data)
}

func (f *Form) Request(path string) (transport.Request, error) {
	if f.err != nil {
		return transport.Request{}, fmt.Errorf("form: build body: %w", f.err)
	}
	if err := f.w.Close(); err != nil {
		return transport.Request{}, fmt.Errorf("form: build body: %w", err)
	}
	return transport.Request{Path: path, Raw: &f.buf, ContentType: f.w.FormDataContentType()}, nil
}

func canonicalMediaType(mediaType string) (string, error) {
	if strings.ContainsFunc(mediaType, func(r rune) bool { return r < ' ' || r == 0x7f }) {
		return "", fmt.Errorf("form: invalid media type %q: control characters", mediaType)
	}
	parsed, params, err := mime.ParseMediaType(mediaType)
	if err != nil {
		return "", fmt.Errorf("form: invalid media type %q: %w", mediaType, err)
	}
	canonical := mime.FormatMediaType(parsed, params)
	if canonical == "" {
		return "", fmt.Errorf("form: invalid media type %q", mediaType)
	}
	return canonical, nil
}

func escapeQuotes(s string) string {
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\r", "", "\n", "").Replace(s)
}
