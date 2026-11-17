package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net"
	"net/url"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/emit"
)

type speechInit struct {
	Text          string                    `json:"text"`
	VoiceSettings *VoiceSettings            `json:"voice_settings,omitzero"`
	Generation    *generationConfig         `json:"generation_config,omitzero"`
	Dictionaries  []PronunciationDictionary `json:"pronunciation_dictionary_locators,omitzero"`
}

type generationConfig struct {
	ChunkLengthSchedule []int `json:"chunk_length_schedule"`
}

type speechText struct {
	Text string `json:"text"`
}

type speechMessage struct {
	Audio   *string `json:"audio"`
	IsFinal bool    `json:"isFinal"`
	Error   string  `json:"error"`
	Message string  `json:"message"`
}

func (m *SpeechModel) streamInput(ctx context.Context, c *ai.SpeechStreamCall, format outputFormat, opts SpeechOptions) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		yield = emit.New(yield)
		conn, err := dialWS(ctx, m.client, "/v1/text-to-speech/"+url.PathEscape(c.Voice)+"/stream-input"+queryString(m.inputQuery(format, opts)))
		if err != nil {
			yield(nil, err)
			return
		}
		ctx, cancel := context.WithCancel(ctx)
		sendErr := make(chan error, 1)
		defer func() {
			cancel()
			if err := conn.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				yield(nil, fmt.Errorf("elevenlabs: close speech stream: %w", err))
			}
		}()
		go func() {
			if err := writeSpeech(ctx, conn, c, opts); err != nil {
				sendErr <- err
				cancel()
			}
		}()
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				select {
				case serr := <-sendErr:
					yield(nil, serr)
				default:
					if websocket.CloseStatus(err) != websocket.StatusNormalClosure {
						yield(nil, fmt.Errorf("elevenlabs: read speech stream: %w", err))
					}
				}
				return
			}
			audio, done, err := decodeSpeechMessage(data)
			if err != nil {
				yield(nil, err)
				return
			}
			if len(audio) > 0 && !yield(audio, nil) {
				return
			}
			if done {
				return
			}
		}
	}
}

func (m *SpeechModel) inputQuery(format outputFormat, opts SpeechOptions) url.Values {
	q := url.Values{}
	set := func(k, v string) {
		if v != "" {
			q.Set(k, v)
		}
	}
	setBool := func(k string, v *bool) {
		if v != nil {
			q.Set(k, strconv.FormatBool(*v))
		}
	}
	set("model_id", m.id)
	set("output_format", format.query)
	set("language_code", opts.LanguageCode)
	set("apply_text_normalization", opts.ApplyTextNormalization)
	if opts.InactivityTimeout > 0 {
		q.Set("inactivity_timeout", strconv.Itoa(opts.InactivityTimeout))
	}
	if opts.Seed != nil {
		q.Set("seed", strconv.FormatInt(*opts.Seed, 10))
	}
	setBool("enable_logging", opts.EnableLogging)
	setBool("auto_mode", opts.AutoMode)
	setBool("sync_alignment", opts.SyncAlignment)
	setBool("enable_ssml_parsing", opts.EnableSSMLParsing)
	return q
}

func writeSpeech(ctx context.Context, conn *websocket.Conn, c *ai.SpeechStreamCall, opts SpeechOptions) error {
	init := speechInit{Text: " ", VoiceSettings: withSpeed(opts.VoiceSettings, c.Speed), Dictionaries: opts.PronunciationDictionaries}
	if len(opts.ChunkLengthSchedule) > 0 {
		init.Generation = &generationConfig{ChunkLengthSchedule: opts.ChunkLengthSchedule}
	}
	if err := writeJSON(ctx, conn, init); err != nil {
		return err
	}
	for chunk := range wordChunks(c.Text) {
		if err := writeJSON(ctx, conn, speechText{Text: chunk}); err != nil {
			return err
		}
	}
	return writeJSON(ctx, conn, speechText{})
}

func writeJSON(ctx context.Context, conn *websocket.Conn, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("elevenlabs: encode websocket message: %w", err)
	}
	if err := conn.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("elevenlabs: send websocket message: %w", err)
	}
	return nil
}

func decodeSpeechMessage(data []byte) (audio []byte, done bool, err error) {
	var m speechMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, false, fmt.Errorf("elevenlabs: decode speech message: %w", err)
	}
	if m.Error != "" || m.Message != "" {
		return nil, false, &ai.APIError{Provider: providerName, Code: m.Error, Message: m.Message, Body: string(data)}
	}
	if m.Audio != nil && *m.Audio != "" {
		audio, err = base64.StdEncoding.DecodeString(*m.Audio)
		if err != nil {
			return nil, false, fmt.Errorf("elevenlabs: decode speech audio: %w", err)
		}
	}
	return audio, m.IsFinal, nil
}

func wordChunks(text iter.Seq[string]) iter.Seq[string] {
	return func(yield func(string) bool) {
		var buf strings.Builder
		for piece := range text {
			buf.WriteString(piece)
			s := buf.String()
			i := strings.LastIndexFunc(s, unicode.IsSpace)
			if i < 0 {
				continue
			}
			_, size := utf8.DecodeRuneInString(s[i:])
			head, rest := s[:i], s[i+size:]
			buf.Reset()
			buf.WriteString(rest)
			if strings.TrimSpace(head) != "" && !yield(head+" ") {
				return
			}
		}
		if rest := strings.TrimSpace(buf.String()); rest != "" {
			yield(rest + " ")
		}
	}
}
