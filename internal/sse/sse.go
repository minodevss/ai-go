// Package sse reads Server-Sent Events streams as defined by the WHATWG HTML standard.
package sse

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strings"
)

const maxEventBytes = 16 << 20

var ErrTooLarge = errors.New("sse: event too large")

type Event struct {
	Name string
	Data string
	ID   string
}

type Reader struct {
	scan    *bufio.Scanner
	started bool
	lastID  string
}

func NewReader(r io.Reader) *Reader {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 64<<10), maxEventBytes)
	s.Split(scanLines)
	return &Reader{scan: s}
}

func (r *Reader) Next() (Event, error) {
	var (
		name    string
		data    strings.Builder
		hasData bool
		size    int
	)
	for r.scan.Scan() {
		line := r.scan.Bytes()
		if size += len(line) + 1; size > maxEventBytes {
			return Event{}, ErrTooLarge
		}
		if !r.started {
			r.started = true
			line = bytes.TrimPrefix(line, []byte("\xef\xbb\xbf"))
		}
		if len(line) == 0 {
			if !hasData {
				name = ""
				size = 0
				continue
			}
			return Event{Name: name, Data: data.String(), ID: r.lastID}, nil
		}
		if line[0] == ':' {
			continue
		}
		field, value, _ := strings.Cut(string(line), ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			name = value
		case "data":
			if hasData {
				data.WriteByte('\n')
			}
			data.WriteString(value)
			hasData = true
		case "id":
			if !strings.Contains(value, "\x00") {
				r.lastID = value
			}
		default:
		}
	}
	if err := r.scan.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return Event{}, ErrTooLarge
		}
		return Event{}, err
	}
	return Event{}, io.EOF
}

func scanLines(data []byte, atEOF bool) (int, []byte, error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	i := bytes.IndexAny(data, "\r\n")
	if i < 0 {
		if atEOF {
			return len(data), data, nil
		}
		return 0, nil, nil
	}
	if data[i] == '\n' {
		return i + 1, data[:i], nil
	}
	if i+1 == len(data) && !atEOF {
		return 0, nil, nil
	}
	if i+1 < len(data) && data[i+1] == '\n' {
		return i + 2, data[:i], nil
	}
	return i + 1, data[:i], nil
}
