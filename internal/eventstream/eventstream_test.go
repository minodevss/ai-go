package eventstream_test

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io"
	"testing"
	"testing/iotest"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/internal/eventstream"
)

const transcribeDocFrame = "AAAAUwAAAEP1RHpYBTpkYXRlCAAAAWiXUkMLEDpjaHVuay1zaWduYXR1cmUGACCt6Zy+uymwEK2SrLp/zVBI5eGn83jdBwCaRUBJA+eaDafqjqI="

func mustEncode(t *testing.T, m eventstream.Message) []byte {
	t.Helper()
	raw, err := eventstream.Encode(m)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	return raw
}

func TestDecoder_Next_awsDocumentationFrame(t *testing.T) {
	raw, err := base64.StdEncoding.DecodeString(transcribeDocFrame)
	if err != nil {
		t.Fatalf("DecodeString() error = %v", err)
	}
	d := eventstream.NewDecoder(bytes.NewReader(raw))
	msg, err := d.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	var names []string
	for _, h := range msg.Headers {
		names = append(names, h.Name)
	}
	if diff := cmp.Diff([]string{":date", ":chunk-signature"}, names); diff != "" {
		t.Errorf("header names mismatch (-want +got):\n%s", diff)
	}
	if msg.Headers[0].Type != eventstream.TypeTimestamp || msg.Headers[1].Type != eventstream.TypeBytes || len(msg.Headers[1].Value) != 32 {
		t.Errorf("headers = %+v, want a timestamp and a 32 byte signature", msg.Headers)
	}
	if len(msg.Payload) != 0 {
		t.Errorf("payload length = %d, want 0", len(msg.Payload))
	}
	if _, err := d.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("second Next() error = %v, want io.EOF", err)
	}
	again, err := eventstream.Encode(msg)
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	if !bytes.Equal(again, raw) {
		t.Errorf("Encode(Decode(frame)) differs from the frame")
	}
}

func TestDecoder_Next_event(t *testing.T) {
	want := eventstream.Message{
		Headers: []eventstream.Header{
			eventstream.StringHeader(":event-type", "contentBlockDelta"),
			eventstream.StringHeader(":content-type", "application/json"),
			eventstream.StringHeader(":message-type", "event"),
		},
		Payload: []byte(`{"contentBlockIndex":0,"delta":{"text":"Hi"}}`),
	}
	raw := append(mustEncode(t, want), mustEncode(t, eventstream.Message{Headers: want.Headers[:1]})...)
	for name, r := range map[string]io.Reader{"whole": bytes.NewReader(raw), "one byte at a time": iotest.OneByteReader(bytes.NewReader(raw))} {
		t.Run(name, func(t *testing.T) {
			d := eventstream.NewDecoder(r)
			got, err := d.Next()
			if err != nil {
				t.Fatalf("Next() error = %v", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("Next() mismatch (-want +got):\n%s", diff)
			}
			if kind, ok := got.String(":event-type"); !ok || kind != "contentBlockDelta" {
				t.Errorf("String(:event-type) = %q, %v", kind, ok)
			}
			if _, err := d.Next(); err != nil {
				t.Errorf("second Next() error = %v", err)
			}
			if _, err := d.Next(); !errors.Is(err, io.EOF) {
				t.Errorf("third Next() error = %v, want io.EOF", err)
			}
		})
	}
}

func TestDecoder_Next_errors(t *testing.T) {
	good := mustEncode(t, eventstream.Message{Headers: []eventstream.Header{eventstream.StringHeader(":event-type", "x")}, Payload: []byte("body")})
	flip := func(i int) []byte {
		out := bytes.Clone(good)
		out[i] ^= 0xff
		return out
	}
	tests := []struct {
		name string
		raw  []byte
		want error
	}{
		{name: "prelude checksum", raw: flip(9), want: eventstream.ErrChecksum},
		{name: "message checksum", raw: flip(len(good) - 1), want: eventstream.ErrChecksum},
		{name: "payload corruption", raw: flip(len(good) - 6), want: eventstream.ErrChecksum},
		{name: "truncated prelude", raw: good[:5], want: io.ErrUnexpectedEOF},
		{name: "truncated body", raw: good[:len(good)-3], want: io.ErrUnexpectedEOF},
		{name: "empty", raw: nil, want: io.EOF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := eventstream.NewDecoder(bytes.NewReader(tt.raw)).Next(); !errors.Is(err, tt.want) {
				t.Errorf("Next() error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestDecoder_Next_rejectsBadLengths(t *testing.T) {
	short := []byte{0, 0, 0, 8, 0, 0, 0, 0}
	short = append(short, crcOf(short)...)
	if _, err := eventstream.NewDecoder(bytes.NewReader(short)).Next(); !errors.Is(err, eventstream.ErrFormat) {
		t.Errorf("Next() error = %v, want ErrFormat for a total length below the minimum", err)
	}
	huge := []byte{0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0}
	huge = append(huge, crcOf(huge)...)
	if _, err := eventstream.NewDecoder(bytes.NewReader(huge)).Next(); !errors.Is(err, eventstream.ErrFormat) {
		t.Errorf("Next() error = %v, want ErrFormat for an oversized message", err)
	}
	over := []byte{0, 0, 0, 16, 0, 0, 0, 200}
	over = append(over, crcOf(over)...)
	if _, err := eventstream.NewDecoder(bytes.NewReader(over)).Next(); !errors.Is(err, eventstream.ErrFormat) {
		t.Errorf("Next() error = %v, want ErrFormat for headers longer than the message", err)
	}
}

func TestEncode_rejectsLongHeaderName(t *testing.T) {
	long := string(make([]byte, 256))
	if _, err := eventstream.Encode(eventstream.Message{Headers: []eventstream.Header{eventstream.StringHeader(long, "x")}}); !errors.Is(err, eventstream.ErrFormat) {
		t.Errorf("Encode() error = %v, want ErrFormat", err)
	}
}
