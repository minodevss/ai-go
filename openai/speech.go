package openai

import (
	"context"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	maxSpeechChars  = 4096
	minSpeechChunk  = 24
	speechReadChunk = 16 << 10
	maxSpeechBytes  = 256 << 20
	defaultVoice    = "alloy"
	customVoiceID   = "voice_"
)

// SpeechModel sends one request per sentence-sized chunk when streaming, so StreamSpeech accepts only the concatenable formats mp3 and pcm.
type SpeechModel struct {
	client *transport.Client
	id     string
}

type speechRequest struct {
	Model        string   `json:"model"`
	Input        string   `json:"input"`
	Voice        any      `json:"voice"`
	Instructions string   `json:"instructions,omitzero"`
	Format       string   `json:"response_format,omitzero"`
	Speed        *float64 `json:"speed,omitzero"`
}

type customVoice struct {
	ID string `json:"id"`
}

func (p *Provider) SpeechModel(id string) *SpeechModel {
	return &SpeechModel{client: p.client, id: id}
}

func (m *SpeechModel) Provider() string { return providerName }

func (m *SpeechModel) ModelID() string { return m.id }

func (m *SpeechModel) GenerateSpeech(ctx context.Context, c *ai.SpeechCall) (*ai.SpeechResponse, error) {
	if n := utf8.RuneCountInString(c.Text); n > maxSpeechChars {
		return nil, fmt.Errorf("openai: speech text has %d characters, the limit is %d; use StreamSpeech, which splits the text", n, maxSpeechChars)
	}
	resp, err := m.speak(ctx, c.Text, c.Voice, c.Format, c.Instructions, c.Speed, c.Headers)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := readLimited(resp.Body, maxSpeechBytes, "audio")
	if err != nil {
		return nil, err
	}
	return &ai.SpeechResponse{Audio: ai.File{MediaType: audioMediaType(c.Format), Data: data}, Header: resp.Header}, nil
}

func (m *SpeechModel) StreamSpeech(ctx context.Context, c *ai.SpeechStreamCall) (*ai.SpeechStreamResponse, error) {
	switch c.Format {
	case "", "mp3", "pcm":
	default:
		return nil, fmt.Errorf("openai: speech format %q cannot be streamed because every chunk is a separate file; use mp3 or pcm", c.Format)
	}
	audio := func(yield func([]byte, error) bool) {
		for chunk := range speechChunks(c.Text) {
			resp, err := m.speak(ctx, chunk, c.Voice, c.Format, c.Instructions, c.Speed, c.Headers)
			if err != nil {
				yield(nil, err)
				return
			}
			ok := copyAudio(resp.Body, yield)
			resp.Body.Close()
			if !ok {
				return
			}
		}
	}
	return &ai.SpeechStreamResponse{Audio: audio, MediaType: audioMediaType(c.Format)}, nil
}

func (m *SpeechModel) speak(ctx context.Context, text, voice, format, instructions string, speed *float64, headers http.Header) (*http.Response, error) {
	req := speechRequest{Model: m.id, Input: text, Voice: voiceOf(voice), Instructions: instructions, Format: format, Speed: speed}
	h := headers.Clone()
	if h == nil {
		h = http.Header{}
	}
	h.Set("Accept", "audio/*")
	return m.client.Do(ctx, transport.Request{Path: "/audio/speech", Body: req, Header: h})
}

func voiceOf(v string) any {
	switch {
	case v == "":
		return defaultVoice
	case strings.HasPrefix(v, customVoiceID):
		return customVoice{ID: v}
	default:
		return v
	}
}

func copyAudio(body io.Reader, yield func([]byte, error) bool) bool {
	buf := make([]byte, speechReadChunk)
	for {
		n, err := body.Read(buf)
		if n > 0 && !yield(append([]byte(nil), buf[:n]...), nil) {
			return false
		}
		if err == io.EOF {
			return true
		}
		if err != nil {
			yield(nil, fmt.Errorf("openai: read audio stream: %w", err))
			return false
		}
	}
}

func audioMediaType(format string) string {
	switch format {
	case "opus":
		return "audio/opus"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm"
	default:
		return "audio/mpeg"
	}
}

func speechChunks(text iter.Seq[string]) iter.Seq[string] {
	return func(yield func(string) bool) {
		var buf strings.Builder
		for piece := range text {
			buf.WriteString(piece)
			for {
				chunk, rest, ok := cutChunk(buf.String())
				if !ok {
					break
				}
				buf.Reset()
				buf.WriteString(rest)
				if !yield(chunk) {
					return
				}
			}
		}
		if rest := strings.TrimSpace(buf.String()); rest != "" {
			for _, part := range splitLong(rest) {
				if !yield(part) {
					return
				}
			}
		}
	}
}

func cutChunk(s string) (chunk, rest string, ok bool) {
	runes := 0
	for i, r := range s {
		runes++
		if runes > maxSpeechChars {
			return forcedCut(s)
		}
		if runes < minSpeechChunk || !isSentenceEnd(r) {
			continue
		}
		next := i + utf8.RuneLen(r)
		if r == '\n' {
			return strings.TrimSpace(s[:next]), s[next:], strings.TrimSpace(s[:next]) != ""
		}
		if next >= len(s) {
			return "", "", false
		}
		if isFullWidthEnd(r) {
			return strings.TrimSpace(s[:next]), s[next:], true
		}
		if after, _ := utf8.DecodeRuneInString(s[next:]); unicode.IsSpace(after) {
			return strings.TrimSpace(s[:next]), s[next:], true
		}
	}
	return "", "", false
}

func forcedCut(s string) (chunk, rest string, ok bool) {
	limit := 0
	for n := range s {
		if limit++; limit > maxSpeechChars {
			limit = n
			break
		}
	}
	head := s[:limit]
	if i := strings.LastIndexFunc(head, unicode.IsSpace); i > 0 {
		head = head[:i]
	}
	return strings.TrimSpace(head), s[len(head):], strings.TrimSpace(head) != ""
}

func splitLong(s string) []string {
	var out []string
	for utf8.RuneCountInString(s) > maxSpeechChars {
		chunk, rest, ok := forcedCut(s)
		if !ok {
			break
		}
		out = append(out, chunk)
		s = strings.TrimSpace(rest)
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}

func isSentenceEnd(r rune) bool {
	switch r {
	case '.', '!', '?', '\n':
		return true
	default:
		return isFullWidthEnd(r)
	}
}

func isFullWidthEnd(r rune) bool {
	switch r {
	case '。', '！', '？', '…':
		return true
	default:
		return false
	}
}

var (
	_ ai.SpeechModel       = (*SpeechModel)(nil)
	_ ai.SpeechStreamModel = (*SpeechModel)(nil)
)
