// Package eventstream reads and writes the application/vnd.amazon.eventstream binary framing used by AWS streaming APIs.
package eventstream

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

const (
	preludeLength  = 12
	trailerLength  = 4
	minMessage     = preludeLength + trailerLength
	maxMessage     = 16 << 20
	maxValueLength = 1<<16 - 1
)

type HeaderType byte

const (
	TypeTrue HeaderType = iota
	TypeFalse
	TypeByte
	TypeShort
	TypeInteger
	TypeLong
	TypeBytes
	TypeString
	TypeTimestamp
	TypeUUID
)

var (
	ErrChecksum = errors.New("eventstream: checksum mismatch")
	ErrFormat   = errors.New("eventstream: malformed message")
)

type Header struct {
	Name  string
	Type  HeaderType
	Value []byte
}

type Message struct {
	Headers []Header
	Payload []byte
}

// String returns the value of the first string header with the given name.
func (m Message) String(name string) (string, bool) {
	for _, h := range m.Headers {
		if h.Name == name && h.Type == TypeString {
			return string(h.Value), true
		}
	}
	return "", false
}

type Decoder struct {
	r io.Reader
}

func NewDecoder(r io.Reader) *Decoder { return &Decoder{r: r} }

// Next returns io.EOF after the last complete message and io.ErrUnexpectedEOF when the stream ends inside one.
func (d *Decoder) Next() (Message, error) {
	var prelude [preludeLength]byte
	if _, err := io.ReadFull(d.r, prelude[:1]); err != nil {
		return Message{}, err
	}
	if _, err := io.ReadFull(d.r, prelude[1:]); err != nil {
		return Message{}, unexpected(err)
	}
	total := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])
	if crc32.ChecksumIEEE(prelude[:8]) != binary.BigEndian.Uint32(prelude[8:12]) {
		return Message{}, fmt.Errorf("%w: prelude", ErrChecksum)
	}
	if total < minMessage || total > maxMessage || uint64(headersLen) > uint64(total)-minMessage {
		return Message{}, fmt.Errorf("%w: lengths %d and %d", ErrFormat, total, headersLen)
	}
	rest := make([]byte, total-preludeLength)
	if _, err := io.ReadFull(d.r, rest); err != nil {
		return Message{}, unexpected(err)
	}
	sum := crc32.NewIEEE()
	sum.Write(prelude[:])
	sum.Write(rest[:len(rest)-trailerLength])
	if sum.Sum32() != binary.BigEndian.Uint32(rest[len(rest)-trailerLength:]) {
		return Message{}, fmt.Errorf("%w: message", ErrChecksum)
	}
	headers, err := decodeHeaders(rest[:headersLen])
	if err != nil {
		return Message{}, err
	}
	return Message{Headers: headers, Payload: rest[headersLen : len(rest)-trailerLength]}, nil
}

func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

func decodeHeaders(b []byte) ([]Header, error) {
	var out []Header
	for len(b) > 0 {
		nameLen := int(b[0])
		if len(b) < 1+nameLen+1 {
			return nil, fmt.Errorf("%w: header name", ErrFormat)
		}
		h := Header{Name: string(b[1 : 1+nameLen]), Type: HeaderType(b[1+nameLen])}
		b = b[2+nameLen:]
		size, err := valueSize(h.Type, b)
		if err != nil {
			return nil, err
		}
		if len(b) < size.prefix+size.value {
			return nil, fmt.Errorf("%w: header %q value", ErrFormat, h.Name)
		}
		h.Value = bytes.Clone(b[size.prefix : size.prefix+size.value])
		b = b[size.prefix+size.value:]
		out = append(out, h)
	}
	return out, nil
}

type sizes struct{ prefix, value int }

func valueSize(t HeaderType, b []byte) (sizes, error) {
	switch t {
	case TypeTrue, TypeFalse:
		return sizes{}, nil
	case TypeByte:
		return sizes{value: 1}, nil
	case TypeShort:
		return sizes{value: 2}, nil
	case TypeInteger:
		return sizes{value: 4}, nil
	case TypeLong, TypeTimestamp:
		return sizes{value: 8}, nil
	case TypeUUID:
		return sizes{value: 16}, nil
	case TypeBytes, TypeString:
		if len(b) < 2 {
			return sizes{}, fmt.Errorf("%w: header value length", ErrFormat)
		}
		return sizes{prefix: 2, value: int(binary.BigEndian.Uint16(b))}, nil
	default:
		return sizes{}, fmt.Errorf("%w: header type %d", ErrFormat, t)
	}
}

// Encode returns the wire form of m. It is the inverse of Decoder.Next for messages that fit the framing limits.
func Encode(m Message) ([]byte, error) {
	var headers bytes.Buffer
	for _, h := range m.Headers {
		if len(h.Name) > 255 {
			return nil, fmt.Errorf("%w: header name %q is too long", ErrFormat, h.Name)
		}
		headers.WriteByte(byte(len(h.Name)))
		headers.WriteString(h.Name)
		headers.WriteByte(byte(h.Type))
		if h.Type == TypeBytes || h.Type == TypeString {
			if len(h.Value) > maxValueLength {
				return nil, fmt.Errorf("%w: header %q value is too long", ErrFormat, h.Name)
			}
			headers.Write(binary.BigEndian.AppendUint16(nil, uint16(len(h.Value))))
		}
		headers.Write(h.Value)
	}
	total := preludeLength + headers.Len() + len(m.Payload) + trailerLength
	if total > maxMessage {
		return nil, fmt.Errorf("%w: message of %d bytes is too long", ErrFormat, total)
	}
	out := binary.BigEndian.AppendUint32(nil, uint32(total))
	out = binary.BigEndian.AppendUint32(out, uint32(headers.Len()))
	out = binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(out))
	out = append(out, headers.Bytes()...)
	out = append(out, m.Payload...)
	return binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(out)), nil
}

// StringHeader builds a string-typed header.
func StringHeader(name, value string) Header {
	return Header{Name: name, Type: TypeString, Value: []byte(value)}
}
