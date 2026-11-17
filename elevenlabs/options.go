package elevenlabs

type StreamTransport string

const (
	// StreamWebSocket sends text while it arrives, over the stream-input WebSocket. It is the default.
	StreamWebSocket StreamTransport = "websocket"
	// StreamHTTP collects all text first and then streams audio from the /stream endpoint.
	StreamHTTP StreamTransport = "http"
)

// VoiceSettings overrides the stored settings of a voice for one request. Unset fields keep the voice's values; Speed is also set by ai.SpeechRequest.Speed.
type VoiceSettings struct {
	Stability       *float64 `json:"stability,omitzero"`
	SimilarityBoost *float64 `json:"similarity_boost,omitzero"`
	Style           *float64 `json:"style,omitzero"`
	UseSpeakerBoost *bool    `json:"use_speaker_boost,omitzero"`
	Speed           *float64 `json:"speed,omitzero"`
}

type PronunciationDictionary struct {
	ID        string `json:"pronunciation_dictionary_id"`
	VersionID string `json:"version_id,omitzero"`
}

// SpeechOptions are text-to-speech settings passed with ai.Providers. ChunkLengthSchedule, InactivityTimeout, AutoMode, SyncAlignment and EnableSSMLParsing apply to the WebSocket transport only.
type SpeechOptions struct {
	LanguageCode              string                    `json:"languageCode,omitzero"`
	VoiceSettings             *VoiceSettings            `json:"voiceSettings,omitzero"`
	Seed                      *int64                    `json:"seed,omitzero"`
	PreviousText              string                    `json:"previousText,omitzero"`
	NextText                  string                    `json:"nextText,omitzero"`
	ApplyTextNormalization    string                    `json:"applyTextNormalization,omitzero"`
	PronunciationDictionaries []PronunciationDictionary `json:"pronunciationDictionaries,omitzero"`
	EnableLogging             *bool                     `json:"enableLogging,omitzero"`
	StreamTransport           StreamTransport           `json:"streamTransport,omitzero"`
	ChunkLengthSchedule       []int                     `json:"chunkLengthSchedule,omitzero"`
	InactivityTimeout         int                       `json:"inactivityTimeout,omitzero"`
	AutoMode                  *bool                     `json:"autoMode,omitzero"`
	SyncAlignment             *bool                     `json:"syncAlignment,omitzero"`
	EnableSSMLParsing         *bool                     `json:"enableSsmlParsing,omitzero"`
}

func (SpeechOptions) ProviderName() string { return providerName }

// TranscriptionOptions are speech-to-text settings passed with ai.Providers. AudioFormat, CommitStrategy and IncludeTimestamps apply to StreamTranscribe only;
// Diarize, NoVerbatim and EnableLogging apply to both calls, and the other fields to Transcribe only.
type TranscriptionOptions struct {
	Diarize               *bool    `json:"diarize,omitzero"`
	NumSpeakers           int      `json:"numSpeakers,omitzero"`
	TimestampsGranularity string   `json:"timestampsGranularity,omitzero"`
	TagAudioEvents        *bool    `json:"tagAudioEvents,omitzero"`
	Keyterms              []string `json:"keyterms,omitzero"`
	NoVerbatim            *bool    `json:"noVerbatim,omitzero"`
	Temperature           *float64 `json:"temperature,omitzero"`
	Seed                  *int64   `json:"seed,omitzero"`
	UseMultiChannel       *bool    `json:"useMultiChannel,omitzero"`
	EnableLogging         *bool    `json:"enableLogging,omitzero"`
	AudioFormat           string   `json:"audioFormat,omitzero"`
	CommitStrategy        string   `json:"commitStrategy,omitzero"`
	IncludeTimestamps     *bool    `json:"includeTimestamps,omitzero"`
}

func (TranscriptionOptions) ProviderName() string { return providerName }
