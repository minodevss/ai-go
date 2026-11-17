package elevenlabs

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/form"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

const maxTranscribeBytes = 256 << 20

// TranscriptionModel transcribes with Scribe models such as scribe_v2. Segments end at a speaker change or sentence-final punctuation, and the audio is read into memory up to 256 MB.
// StreamTranscribe takes raw PCM or mu-law audio, reads it until EOF and closes it if it is an io.Closer. The ids scribe_v2 and scribe_v3 map to their live model ids.
type TranscriptionModel struct {
	client *transport.Client
	id     string
}

type scribeResponse struct {
	LanguageCode string           `json:"language_code"`
	Text         string           `json:"text"`
	Words        []scribeWord     `json:"words"`
	Transcripts  []scribeResponse `json:"transcripts"`
}

type scribeWord struct {
	Text      string   `json:"text"`
	Type      string   `json:"type"`
	Start     *float64 `json:"start"`
	End       *float64 `json:"end"`
	SpeakerID string   `json:"speaker_id"`
}

func (p *Provider) TranscriptionModel(id string) *TranscriptionModel {
	return &TranscriptionModel{client: p.client, id: id}
}

func NewTranscriptionModel(id string) *TranscriptionModel { return New().TranscriptionModel(id) }

func (m *TranscriptionModel) Provider() string { return providerName }

func (m *TranscriptionModel) ModelID() string { return m.id }

func (m *TranscriptionModel) Transcribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	audio, err := io.ReadAll(io.LimitReader(c.Audio, maxTranscribeBytes+1))
	if err != nil {
		return nil, fmt.Errorf("elevenlabs: read audio: %w", err)
	}
	if len(audio) > maxTranscribeBytes {
		return nil, fmt.Errorf("elevenlabs: audio exceeds the %d MB limit", maxTranscribeBytes>>20)
	}
	f := form.New()
	f.Field("model_id", m.id)
	f.File("file", mediaio.AudioName(c.MediaType), c.MediaType, bytes.NewReader(audio))
	f.Field("language_code", c.Language)
	f.Flag("diarize", opts.Diarize)
	if opts.NumSpeakers > 0 {
		f.Field("num_speakers", strconv.Itoa(opts.NumSpeakers))
	}
	f.Field("timestamps_granularity", opts.TimestampsGranularity)
	f.Flag("tag_audio_events", opts.TagAudioEvents)
	for _, k := range opts.Keyterms {
		f.Field("keyterms", k)
	}
	f.Flag("no_verbatim", opts.NoVerbatim)
	f.Flag("use_multi_channel", opts.UseMultiChannel)
	if opts.Temperature != nil {
		f.Field("temperature", strconv.FormatFloat(*opts.Temperature, 'f', -1, 64))
	}
	if opts.Seed != nil {
		f.Field("seed", strconv.FormatInt(*opts.Seed, 10))
	}
	req, err := f.Request("/v1/speech-to-text")
	if err != nil {
		return nil, err
	}
	req.Header = c.Headers.Clone()
	if opts.EnableLogging != nil {
		q := url.Values{"enable_logging": {strconv.FormatBool(*opts.EnableLogging)}}
		req.Path += queryString(q)
	}
	var body scribeResponse
	header, err := m.client.JSON(ctx, req, &body)
	if err != nil {
		return nil, err
	}
	return responseOf(body, header), nil
}

func responseOf(b scribeResponse, header map[string][]string) *ai.TranscribeResponse {
	if len(b.Transcripts) == 0 {
		return &ai.TranscribeResponse{Text: b.Text, Segments: segmentsOf(b.Words, ""), Language: b.LanguageCode, Header: header}
	}
	out := &ai.TranscribeResponse{Header: header}
	texts := make([]string, 0, len(b.Transcripts))
	for i, t := range b.Transcripts {
		texts = append(texts, t.Text)
		out.Segments = append(out.Segments, segmentsOf(t.Words, "channel_"+strconv.Itoa(i))...)
		if out.Language == "" {
			out.Language = t.LanguageCode
		}
	}
	out.Text = strings.Join(texts, "\n")
	return out
}

func segmentsOf(words []scribeWord, fallbackSpeaker string) []ai.Segment {
	spaced := false
	for _, w := range words {
		if w.Type == "spacing" {
			spaced = true
			break
		}
	}
	var (
		out     []ai.Segment
		text    strings.Builder
		cur     ai.Segment
		started bool
	)
	flush := func() {
		if t := strings.TrimSpace(text.String()); t != "" {
			cur.Text = t
			out = append(out, cur)
		}
		text.Reset()
		cur = ai.Segment{}
		started = false
	}
	for _, w := range words {
		if w.Type == "spacing" {
			if started {
				text.WriteString(w.Text)
			}
			continue
		}
		speaker := w.SpeakerID
		if speaker == "" {
			speaker = fallbackSpeaker
		}
		if started && cur.Speaker != speaker {
			flush()
		}
		if !started {
			started = true
			cur.Speaker = speaker
			if w.Start != nil {
				cur.Start = seconds(*w.Start)
			}
		} else if !spaced {
			text.WriteByte(' ')
		}
		text.WriteString(w.Text)
		if w.End != nil {
			cur.End = seconds(*w.End)
		}
		if w.Type == "word" && endsSentence(w.Text) {
			flush()
		}
	}
	flush()
	return out
}

func endsSentence(s string) bool {
	s = strings.TrimRight(s, "\"')]”’」』 ")
	if s == "" {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?':
		return true
	}
	return strings.HasSuffix(s, "。") || strings.HasSuffix(s, "！") || strings.HasSuffix(s, "？")
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

var _ ai.TranscriptionModel = (*TranscriptionModel)(nil)
