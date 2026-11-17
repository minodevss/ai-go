package assemblyai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/mediaio"
	"github.com/minodevss/ai-go/internal/retry"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	defaultPollInterval = 3 * time.Second
	maxPollFailures     = 3
)

// TranscriptionModel transcribes with the async API: Transcribe uploads, submits and polls, so ctx bounds the wait and an ended ctx returns an *ai.JobError with the transcript id.
// An empty model id leaves the speech model to AssemblyAI. StreamTranscribe uses Universal-Streaming, reads Audio until EOF and closes it if it is an io.Closer.
type TranscriptionModel struct {
	provider *Provider
	client   *transport.Client
	id       string
}

type uploadResponse struct {
	UploadURL string `json:"upload_url"`
}

type speakerOptions struct {
	Min int `json:"min_speakers_expected,omitzero"`
	Max int `json:"max_speakers_expected,omitzero"`
}

type submitRequest struct {
	AudioURL          string          `json:"audio_url"`
	SpeechModels      []string        `json:"speech_models,omitzero"`
	LanguageCode      string          `json:"language_code,omitzero"`
	LanguageDetection bool            `json:"language_detection,omitzero"`
	SpeakerLabels     bool            `json:"speaker_labels,omitzero"`
	SpeakersExpected  int             `json:"speakers_expected,omitzero"`
	SpeakerOptions    *speakerOptions `json:"speaker_options,omitzero"`
	Punctuate         *bool           `json:"punctuate,omitzero"`
	FormatText        *bool           `json:"format_text,omitzero"`
	Disfluencies      *bool           `json:"disfluencies,omitzero"`
	FilterProfanity   bool            `json:"filter_profanity,omitzero"`
	Multichannel      bool            `json:"multichannel,omitzero"`
	KeytermsPrompt    []string        `json:"keyterms_prompt,omitzero"`
	Prompt            string          `json:"prompt,omitzero"`
}

type transcriptBody struct {
	ID           string      `json:"id"`
	Status       string      `json:"status"`
	Text         string      `json:"text"`
	Error        string      `json:"error"`
	LanguageCode string      `json:"language_code"`
	Words        []wordBody  `json:"words"`
	Utterances   []utterance `json:"utterances"`
}

type wordBody struct {
	Text    string `json:"text"`
	Start   int64  `json:"start"`
	End     int64  `json:"end"`
	Speaker string `json:"speaker"`
}

type utterance struct {
	Text    string `json:"text"`
	Start   int64  `json:"start"`
	End     int64  `json:"end"`
	Speaker string `json:"speaker"`
}

func (p *Provider) TranscriptionModel(id string) *TranscriptionModel {
	return &TranscriptionModel{provider: p, client: p.client, id: id}
}

func NewTranscriptionModel(id string) *TranscriptionModel { return New().TranscriptionModel(id) }

func (m *TranscriptionModel) Provider() string { return providerName }

func (m *TranscriptionModel) ModelID() string { return m.id }

func (m *TranscriptionModel) Transcribe(ctx context.Context, c *ai.TranscribeCall) (*ai.TranscribeResponse, error) {
	opts, err := mediaio.DecodeOptions[TranscriptionOptions](providerName, c.Provider)
	if err != nil {
		return nil, err
	}
	var up uploadResponse
	upload := transport.Request{Path: "/v2/upload", Raw: c.Audio, ContentType: "application/octet-stream", Header: c.Headers.Clone()}
	if _, err := m.client.JSON(ctx, upload, &up); err != nil {
		return nil, err
	}
	var job transcriptBody
	if _, err := m.client.JSON(ctx, transport.Request{Path: "/v2/transcript", Body: m.submission(c, opts, up.UploadURL), Header: c.Headers.Clone()}, &job); err != nil {
		return nil, err
	}
	if job.ID == "" {
		return nil, errors.New("assemblyai: transcript response has no id")
	}
	body, header, err := m.await(ctx, job.ID, c.Headers, opts.PollInterval)
	if err != nil {
		return nil, &ai.JobError{Kind: "transcription", ID: job.ID, Err: err}
	}
	return &ai.TranscribeResponse{Text: body.Text, Segments: segmentsOf(body, opts.Words), Language: body.LanguageCode, Header: header}, nil
}

func (m *TranscriptionModel) submission(c *ai.TranscribeCall, o TranscriptionOptions, audioURL string) submitRequest {
	req := submitRequest{
		AudioURL:          audioURL,
		LanguageCode:      c.Language,
		LanguageDetection: o.LanguageDetection != nil && *o.LanguageDetection && c.Language == "",
		SpeakerLabels:     o.SpeakerLabels,
		SpeakersExpected:  o.SpeakersExpected,
		Punctuate:         o.Punctuate,
		FormatText:        o.FormatText,
		Disfluencies:      o.Disfluencies,
		FilterProfanity:   o.FilterProfanity,
		Multichannel:      o.Multichannel,
		KeytermsPrompt:    o.KeytermsPrompt,
		Prompt:            o.Prompt,
	}
	if m.id != "" {
		req.SpeechModels = []string{m.id}
	}
	if o.MinSpeakers != 0 || o.MaxSpeakers != 0 {
		req.SpeakerOptions = &speakerOptions{Min: o.MinSpeakers, Max: o.MaxSpeakers}
	}
	return req
}

func (m *TranscriptionModel) await(ctx context.Context, id string, headers http.Header, interval time.Duration) (*transcriptBody, http.Header, error) {
	if interval <= 0 {
		interval = defaultPollInterval
	}
	req := transport.Request{Method: http.MethodGet, Path: "/v2/transcript/" + url.PathEscape(id), Header: headers.Clone()}
	failures := 0
	for {
		var body transcriptBody
		header, err := m.client.JSON(ctx, req, &body)
		switch {
		case err == nil:
			failures = 0
			switch body.Status {
			case "completed":
				return &body, header, nil
			case "error":
				return nil, nil, fmt.Errorf("assemblyai: transcript failed: %s", body.Error)
			}
		case ctx.Err() != nil || !retryable(err) || failures >= maxPollFailures:
			return nil, nil, settled(err)
		default:
			failures++
		}
		if err := retry.Sleep(ctx, interval); err != nil {
			return nil, nil, err
		}
	}
}

func retryable(err error) bool {
	apiErr, ok := errors.AsType[*ai.APIError](err)
	return ok && apiErr.Retryable()
}

func settled(err error) error {
	if retryable(err) {
		return errors.New(err.Error())
	}
	return err
}

func segmentsOf(b *transcriptBody, perWord bool) []ai.Segment {
	if perWord && len(b.Words) > 0 {
		out := make([]ai.Segment, len(b.Words))
		for i, w := range b.Words {
			out[i] = ai.Segment{Text: w.Text, Start: millis(w.Start), End: millis(w.End), Speaker: w.Speaker}
		}
		return out
	}
	if !perWord && len(b.Utterances) > 0 {
		out := make([]ai.Segment, len(b.Utterances))
		for i, u := range b.Utterances {
			out[i] = ai.Segment{Text: u.Text, Start: millis(u.Start), End: millis(u.End), Speaker: u.Speaker}
		}
		return out
	}
	if b.Text == "" {
		return nil
	}
	whole := ai.Segment{Text: b.Text}
	if len(b.Words) > 0 {
		whole.Start, whole.End = millis(b.Words[0].Start), millis(b.Words[len(b.Words)-1].End)
	}
	return []ai.Segment{whole}
}

func millis(ms int64) time.Duration { return time.Duration(ms) * time.Millisecond }

var (
	_ ai.TranscriptionModel       = (*TranscriptionModel)(nil)
	_ ai.TranscriptionStreamModel = (*TranscriptionModel)(nil)
)
