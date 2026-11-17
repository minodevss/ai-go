package eventstream_test

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io"
	"testing"

	"github.com/minodevss/ai-go/internal/eventstream"
)

func FuzzDecoder(f *testing.F) {
	doc, err := base64.StdEncoding.DecodeString(transcribeDocFrame)
	if err != nil {
		f.Fatalf("DecodeString() error = %v", err)
	}
	event, err := eventstream.Encode(eventstream.Message{
		Headers: []eventstream.Header{
			eventstream.StringHeader(":event-type", "contentBlockDelta"),
			eventstream.StringHeader(":message-type", "event"),
			{Name: "n", Type: eventstream.TypeLong, Value: make([]byte, 8)},
			{Name: "t", Type: eventstream.TypeTrue},
			{Name: "u", Type: eventstream.TypeUUID, Value: make([]byte, 16)},
		},
		Payload: []byte(`{"contentBlockIndex":0,"delta":{"text":"Hi"}}`),
	})
	if err != nil {
		f.Fatalf("Encode() error = %v", err)
	}
	f.Add(doc)
	f.Add(event)
	f.Add(append(bytes.Clone(event), doc...))
	f.Add(event[:len(event)/2])
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := eventstream.NewDecoder(bytes.NewReader(data))
		var decoded []eventstream.Message
		for {
			msg, err := d.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return
			}
			decoded = append(decoded, msg)
		}
		var again []byte
		for _, msg := range decoded {
			raw, err := eventstream.Encode(msg)
			if err != nil {
				t.Fatalf("Encode(decoded message) error = %v", err)
			}
			again = append(again, raw...)
		}
		if !bytes.Equal(again, data) {
			t.Errorf("re-encoding decoded messages differs from the input")
		}
	})
}
