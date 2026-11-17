package deepgram

import (
	"net/url"
	"strconv"
	"time"
)

// TranscriptionOptions are transcription settings passed with ai.Providers. SmartFormat and Utterances default to true, and Diarize uses the latest diarizer unless DiarizeModel names one.
// The fields from Encoding down apply to StreamTranscribe only.
type TranscriptionOptions struct {
	SmartFormat     *bool    `json:"smartFormat,omitzero"`
	Punctuate       *bool    `json:"punctuate,omitzero"`
	Utterances      *bool    `json:"utterances,omitzero"`
	UttSplit        *float64 `json:"uttSplit,omitzero"`
	Paragraphs      bool     `json:"paragraphs,omitzero"`
	Diarize         bool     `json:"diarize,omitzero"`
	DiarizeModel    string   `json:"diarizeModel,omitzero"`
	FillerWords     bool     `json:"fillerWords,omitzero"`
	Numerals        bool     `json:"numerals,omitzero"`
	ProfanityFilter bool     `json:"profanityFilter,omitzero"`
	Dictation       bool     `json:"dictation,omitzero"`
	DetectLanguage  bool     `json:"detectLanguage,omitzero"`
	Multichannel    bool     `json:"multichannel,omitzero"`
	Keyterms        []string `json:"keyterms,omitzero"`
	Redact          []string `json:"redact,omitzero"`
	Search          []string `json:"search,omitzero"`
	Replace         []string `json:"replace,omitzero"`
	Tags            []string `json:"tags,omitzero"`
	Version         string   `json:"version,omitzero"`
	MIPOptOut       bool     `json:"mipOptOut,omitzero"`

	Encoding           string        `json:"encoding,omitzero"`
	SampleRate         int           `json:"sampleRate,omitzero"`
	Channels           int           `json:"channels,omitzero"`
	InterimResults     *bool         `json:"interimResults,omitzero"`
	Endpointing        *int          `json:"endpointing,omitzero"`
	DisableEndpointing bool          `json:"disableEndpointing,omitzero"`
	UtteranceEndMS     int           `json:"utteranceEndMs,omitzero"`
	VADEvents          bool          `json:"vadEvents,omitzero"`
	KeepAlive          time.Duration `json:"keepAlive,omitzero"`
}

func (TranscriptionOptions) ProviderName() string { return providerName }

// SpeechOptions are text-to-speech settings passed with ai.Providers. They override the encoding and container chosen from SpeechRequest.Format.
type SpeechOptions struct {
	Encoding   string   `json:"encoding,omitzero"`
	Container  string   `json:"container,omitzero"`
	SampleRate int      `json:"sampleRate,omitzero"`
	BitRate    int      `json:"bitRate,omitzero"`
	Tags       []string `json:"tags,omitzero"`
	MIPOptOut  bool     `json:"mipOptOut,omitzero"`
}

func (SpeechOptions) ProviderName() string { return providerName }

type query struct{ url.Values }

func newQuery() query { return query{url.Values{}} }

func (q query) set(key, value string) {
	if value != "" {
		q.Set(key, value)
	}
}

func (q query) setInt(key string, value int) {
	if value != 0 {
		q.Set(key, strconv.Itoa(value))
	}
}

func (q query) setFlag(key string, value bool) {
	if value {
		q.Set(key, "true")
	}
}

func (q query) setBool(key string, value *bool) {
	if value != nil {
		q.Set(key, strconv.FormatBool(*value))
	}
}

func (q query) setFloat(key string, value *float64) {
	if value != nil {
		q.Set(key, strconv.FormatFloat(*value, 'f', -1, 64))
	}
}

func (q query) addAll(key string, values []string) {
	for _, v := range values {
		q.Add(key, v)
	}
}
