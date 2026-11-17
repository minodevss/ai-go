package ui

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/minodevss/ai-go"
)

const (
	metadataKey    = "aigo"
	signingContext = "ai-go/ui/v2\x00"
	minKeyBytes    = 32
	signatureV1    = 1
)

type signature struct {
	V   int    `json:"v"`
	At  int64  `json:"at"`
	Sig string `json:"sig"`
	Msg string `json:"msg"`
}

func checkKey(key []byte) error {
	if len(key) < minKeyBytes {
		return ErrWeakKey
	}
	return nil
}

type binding struct {
	ChatID string
	Owner  string
}

func mac(key []byte, bind binding, at int64, msg string) string {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(signingContext))
	h.Write([]byte(strconv.FormatInt(at, 10)))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(len(bind.ChatID))))
	h.Write([]byte{0})
	h.Write([]byte(bind.ChatID))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(len(bind.Owner))))
	h.Write([]byte{0})
	h.Write([]byte(bind.Owner))
	h.Write([]byte{0})
	h.Write([]byte(msg))
	return hex.EncodeToString(h.Sum(nil))
}

func signMessages(key []byte, bind binding, msgs []ai.Message, withProvider bool) (signature, error) {
	if !withProvider {
		msgs = stripProvider(msgs)
	}
	if msgs == nil {
		msgs = []ai.Message{}
	}
	raw, err := json.Marshal(msgs)
	if err != nil {
		return signature{}, err
	}
	msg := base64.StdEncoding.EncodeToString(raw)
	at := time.Now().Unix()
	return signature{V: signatureV1, At: at, Sig: mac(key, bind, at, msg), Msg: msg}, nil
}

func verifyMessages(keys [][]byte, bind binding, maxAge time.Duration, metadata json.RawMessage) ([]ai.Message, error) {
	var meta map[string]json.RawMessage
	if err := json.Unmarshal(metadata, &meta); err != nil || meta[metadataKey] == nil {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "assistant message is not signed")
	}
	var sig signature
	if err := json.Unmarshal(meta[metadataKey], &sig); err != nil || sig.V != signatureV1 {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "malformed signature")
	}
	want, err := hex.DecodeString(sig.Sig)
	if err != nil {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "malformed signature")
	}
	valid := false
	for _, key := range keys {
		got, _ := hex.DecodeString(mac(key, bind, sig.At, sig.Msg))
		if hmac.Equal(got, want) {
			valid = true
		}
	}
	if !valid {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "signature does not match")
	}
	if maxAge > 0 && time.Since(time.Unix(sig.At, 0)) > maxAge {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "signature expired")
	}
	raw, err := base64.StdEncoding.DecodeString(sig.Msg)
	if err != nil {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "malformed payload")
	}
	var msgs []ai.Message
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return nil, requestError(http.StatusBadRequest, ErrBadSignature, "malformed payload")
	}
	return msgs, nil
}

func stripProvider(msgs []ai.Message) []ai.Message {
	out := make([]ai.Message, len(msgs))
	for i, m := range msgs {
		out[i] = ai.Message{Role: m.Role, Content: stripParts(m.Content)}
	}
	return out
}

func stripParts(parts ai.Parts) ai.Parts {
	if parts == nil {
		return nil
	}
	out := make(ai.Parts, len(parts))
	for i, p := range parts {
		switch v := p.(type) {
		case ai.Text:
			v.Provider = nil
			v.Citations = stripCitations(v.Citations)
			out[i] = v
		case ai.Reasoning:
			v.Provider = nil
			out[i] = v
		case ai.File:
			v.Provider = nil
			out[i] = v
		case ai.ToolCall:
			v.Provider = nil
			out[i] = v
		case ai.ToolResult:
			v.Provider = nil
			v.Output.Parts = stripParts(v.Output.Parts)
			out[i] = v
		case ai.Custom:
			v.Provider = nil
			out[i] = v
		default:
			out[i] = p
		}
	}
	return out
}

func stripCitations(cs []ai.Citation) []ai.Citation {
	if cs == nil {
		return nil
	}
	out := make([]ai.Citation, len(cs))
	for i, c := range cs {
		c.Provider = nil
		out[i] = c
	}
	return out
}
