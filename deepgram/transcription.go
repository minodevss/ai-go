package deepgram

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/transport"
)

// TranscriptionModel transcribes with the /v1/listen endpoint, sending the audio as the request body for Transcribe and over a live WebSocket for StreamTranscribe.
// StreamTranscribe reads Audio until EOF and closes it if it is an io.Closer; a reader that neither ends nor closes keeps one goroutine blocked.
type TranscriptionModel struct {
	client *transport.Client
	id     string
}

type listenResponse struct {
	Results struct {
		Channels   []listenChannel `json:"channels"`
		Utterances []utterance     `json:"utterances"`
	} `json:"results"`
}

type listenChannel struct {
	DetectedLanguage string `json:"detected_language"`
	Alternatives     []struct {
		Transcript string   `json:"transcript"`
		Languages  []string `json:"languages"`
		Words      []word   `json:"words"`
		Paragraphs *struct {
			Paragraphs []paragraph `json:"paragraphs"`
		} `json:"paragraphs"`
	} `json:"alternatives"`
}

type word struct {
	Word       string  `json:"word"`
	Punctuated string  `json:"punctuated_word"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Speaker    *int    `json:"speaker"`
}

type utterance struct {
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Channel    int     `json:"channel"`
	Transcript string  `json:"transcript"`
	Speaker    *int    `json:"speaker"`
}

type paragraph struct {
	Speaker   *int    `json:"speaker"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Sentences []struct {
		Text string `json:"text"`
	} `json:"sentences"`
}

func (m *TranscriptionModel) Provider() string { return providerName }

func (m *TranscriptionModel) ModelID() string { return m.id }

func (m *TranscriptionModel) Transcribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	q := m.commonQuery(c, opts)
	q.setBool("utterances", orTrue(opts.Utterances))
	q.setFlag("paragraphs", opts.Paragraphs)
	q.setFloat("utt_split", opts.UttSplit)
	q.setFlag("multichannel", opts.Multichannel)
	q.addAll("search", opts.Search)
	q.set("encoding", opts.Encoding)
	q.setInt("sample_rate", opts.SampleRate)
	contentType := c.MediaType
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	var body listenResponse
	header, err := m.client.JSON(ctx, transport.Request{Path: "/listen?" + q.Encode(), Raw: c.Audio, ContentType: contentType, Header: c.Headers}, &body)
	if err != nil {
		return nil, err
	}
	return responseOf(body, header), nil
}

func (m *TranscriptionModel) commonQuery(c *ai.TranscribeCall, opts TranscriptionOptions) query {
	q := newQuery()
	q.set("model", m.id)
	if opts.DetectLanguage {
		q.setFlag("detect_language", true)
	} else {
		q.set("language", c.Language)
	}
	q.setBool("smart_format", orTrue(opts.SmartFormat))
	q.setBool("punctuate", opts.Punctuate)
	switch {
	case opts.DiarizeModel != "":
		q.set("diarize_model", opts.DiarizeModel)
	case opts.Diarize:
		q.set("diarize_model", "latest")
	}
	q.setFlag("filler_words", opts.FillerWords)
	q.setFlag("numerals", opts.Numerals)
	q.setFlag("profanity_filter", opts.ProfanityFilter)
	q.setFlag("dictation", opts.Dictation)
	q.addAll("keyterm", opts.Keyterms)
	q.addAll("redact", opts.Redact)
	q.addAll("replace", opts.Replace)
	q.addAll("tag", opts.Tags)
	q.set("version", opts.Version)
	q.setFlag("mip_opt_out", opts.MIPOptOut)
	return q
}

func orTrue(b *bool) *bool {
	if b == nil {
		return new(true)
	}
	return b
}

func responseOf(body listenResponse, header http.Header) *ai.TranscribeResponse {
	resp := &ai.TranscribeResponse{Header: header}
	var texts []string
	for _, ch := range body.Results.Channels {
		if len(ch.Alternatives) == 0 {
			continue
		}
		alt := ch.Alternatives[0]
		if alt.Transcript != "" {
			texts = append(texts, alt.Transcript)
		}
		if resp.Language == "" {
			resp.Language = ch.DetectedLanguage
		}
		if resp.Language == "" && len(alt.Languages) > 0 {
			resp.Language = alt.Languages[0]
		}
	}
	resp.Text = strings.Join(texts, "\n")
	resp.Segments = segmentsOf(body)
	return resp
}

func segmentsOf(body listenResponse) []ai.Segment {
	if len(body.Results.Utterances) > 0 {
		out := make([]ai.Segment, 0, len(body.Results.Utterances))
		for _, u := range body.Results.Utterances {
			out = append(out, ai.Segment{Text: u.Transcript, Start: seconds(u.Start), End: seconds(u.End), Speaker: speakerOf(u.Speaker)})
		}
		return out
	}
	var out []ai.Segment
	for _, ch := range body.Results.Channels {
		if len(ch.Alternatives) == 0 {
			continue
		}
		alt := ch.Alternatives[0]
		if alt.Paragraphs != nil && len(alt.Paragraphs.Paragraphs) > 0 {
			for _, p := range alt.Paragraphs.Paragraphs {
				texts := make([]string, len(p.Sentences))
				for i, s := range p.Sentences {
					texts[i] = s.Text
				}
				out = append(out, ai.Segment{Text: strings.Join(texts, " "), Start: seconds(p.Start), End: seconds(p.End), Speaker: speakerOf(p.Speaker)})
			}
			continue
		}
		out = append(out, speakerRuns(alt.Words)...)
	}
	return out
}

func speakerRuns(words []word) []ai.Segment {
	var out []ai.Segment
	var texts []string
	var start, end float64
	var current *int
	flush := func() {
		if len(texts) == 0 {
			return
		}
		out = append(out, ai.Segment{Text: strings.Join(texts, " "), Start: seconds(start), End: seconds(end), Speaker: speakerOf(current)})
		texts = nil
	}
	hasSpeaker := false
	for _, w := range words {
		if w.Speaker != nil {
			hasSpeaker = true
			break
		}
	}
	if !hasSpeaker {
		return nil
	}
	for _, w := range words {
		if len(texts) > 0 && !sameSpeaker(current, w.Speaker) {
			flush()
		}
		if len(texts) == 0 {
			start, current = w.Start, w.Speaker
		}
		end = w.End
		texts = append(texts, wordText(w))
	}
	flush()
	return out
}

func wordText(w word) string {
	if w.Punctuated != "" {
		return w.Punctuated
	}
	return w.Word
}

func sameSpeaker(a, b *int) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func speakerOf(s *int) string {
	if s == nil {
		return ""
	}
	return strconv.Itoa(*s)
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

var _ ai.TranscriptionModel = (*TranscriptionModel)(nil)
