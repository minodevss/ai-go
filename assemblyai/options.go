package assemblyai

import "time"

// TranscriptionOptions are transcription settings passed with ai.Providers. Words returns one segment per word and PollInterval (default 3s) paces polling; both apply to Transcribe only.
// Mode through InactivityTimeout apply to StreamTranscribe only; their durations are sent in milliseconds, InactivityTimeout in seconds.
type TranscriptionOptions struct {
	SpeakerLabels       bool          `json:"speakerLabels,omitzero"`
	SpeakersExpected    int           `json:"speakersExpected,omitzero"`
	MinSpeakers         int           `json:"minSpeakers,omitzero"`
	MaxSpeakers         int           `json:"maxSpeakers,omitzero"`
	LanguageDetection   *bool         `json:"languageDetection,omitzero"`
	Punctuate           *bool         `json:"punctuate,omitzero"`
	FormatText          *bool         `json:"formatText,omitzero"`
	Disfluencies        *bool         `json:"disfluencies,omitzero"`
	FilterProfanity     bool          `json:"filterProfanity,omitzero"`
	Multichannel        bool          `json:"multichannel,omitzero"`
	KeytermsPrompt      []string      `json:"keytermsPrompt,omitzero"`
	Prompt              string        `json:"prompt,omitzero"`
	Words               bool          `json:"words,omitzero"`
	PollInterval        time.Duration `json:"pollInterval,omitzero"`
	Mode                string        `json:"mode,omitzero"`
	MinTurnSilence      time.Duration `json:"minTurnSilence,omitzero"`
	MaxTurnSilence      time.Duration `json:"maxTurnSilence,omitzero"`
	VADThreshold        *float64      `json:"vadThreshold,omitzero"`
	LanguageCodes       []string      `json:"languageCodes,omitzero"`
	IncludePartialTurns *bool         `json:"includePartialTurns,omitzero"`
	InactivityTimeout   time.Duration `json:"inactivityTimeout,omitzero"`
}

func (TranscriptionOptions) ProviderName() string { return providerName }
