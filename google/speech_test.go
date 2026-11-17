package google_test

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func TestSpeechModel_geminiThreePointEight(t *testing.T) {
	tests := []struct {
		name      string
		req       ai.SpeechRequest
		wantBody  string
		wantMedia string
	}{
		{
			name:      "default wav",
			req:       ai.SpeechRequest{Text: "Have a wonderful day!", Instructions: "cheerful and friendly"},
			wantBody:  `{"contents":[{"role":"user","parts":[{"text":"Have a wonderful day!","speech_metadata":{"style":"cheerful and friendly"}}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"voice":"Kore"}}}}`,
			wantMedia: "audio/wav",
		},
		{
			name:      "raw pcm voice id",
			req:       ai.SpeechRequest{Text: "hi", Voice: "voice_abc", Format: "pcm", Provider: ai.Providers(google.SpeechOptions{SampleRate: 16000})},
			wantBody:  `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"voice":"voice_abc"}},"responseFormat":{"audio":{"mimeType":"AUDIO_L16","sampleRate":16000}}}}`,
			wantMedia: "audio/wav",
		},
		{
			name: "two speakers",
			req: ai.SpeechRequest{Text: "conversation", Provider: ai.Providers(google.SpeechOptions{
				Speakers: []google.SpeechSpeaker{{Name: "Joe", Voice: "Puck"}, {Name: "Jane", Voice: "Kore"}},
				Turns:    []google.SpeechTurn{{Speaker: "Joe", Text: "How is it going?", Style: "cheerful"}, {Speaker: "Jane", Text: "Not bad."}},
			})},
			wantBody:  `{"contents":[{"role":"user","parts":[{"text":"How is it going?","speech_metadata":{"speaker":"Joe","style":"cheerful"}},{"text":"Not bad.","speech_metadata":{"speaker":"Jane"}}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"multiSpeakerVoiceConfig":{"speakerVoiceConfigs":[{"speaker":"Joe","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}},{"speaker":"Jane","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}]}}}}`,
			wantMedia: "audio/wav",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(fixture(t, "tts_wav.json")))
			res, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("gemini-3.8-flash-tts"), tt.req)
			if err != nil {
				t.Fatalf("GenerateSpeech() error = %v", err)
			}
			if string(res.Audio.Data) != "RIFF" || res.Audio.MediaType != tt.wantMedia {
				t.Errorf("Audio = %q %q, want the decoded wav", res.Audio.MediaType, res.Audio.Data)
			}
			got := srv.last(t)
			if got.Path != "/v1beta/models/gemini-3.8-flash-tts:generateContent" {
				t.Errorf("path = %q", got.Path)
			}
			equalJSON(t, tt.wantBody, got.Body)
		})
	}
}

func TestSpeechModel_legacyModelWrapsPCMIntoWAV(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "tts_legacy_pcm.json")))
	m := srv.provider().SpeechModel("gemini-2.5-flash-preview-tts")
	res, err := ai.GenerateSpeech(t.Context(), m, ai.SpeechRequest{Text: "Say cheerfully: have a nice day", Voice: "Puck", Speed: new(1.2)})
	if err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	data := res.Audio.Data
	if res.Audio.MediaType != "audio/wav" || len(data) != 48 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" || string(data[44:]) != "\x01\x02\x03\x04" {
		t.Fatalf("Audio = %q %x, want a 48-byte wav", res.Audio.MediaType, data)
	}
	if rate := binary.LittleEndian.Uint32(data[24:]); rate != 24000 {
		t.Errorf("sample rate = %d, want 24000", rate)
	}
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "speed") {
		t.Errorf("Warnings = %+v, want the speed warning", res.Warnings)
	}
	equalJSON(t, `{"contents":[{"role":"user","parts":[{"text":"Say cheerfully: have a nice day"}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}}}}`, srv.last(t).Body)
}

func TestSpeechModel_legacyPCMFormatStaysRaw(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "tts_legacy_pcm.json")))
	res, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel("gemini-3.1-flash-tts-preview"), ai.SpeechRequest{Text: "hi", Format: "pcm", Instructions: "softly"})
	if err != nil {
		t.Fatalf("GenerateSpeech() error = %v", err)
	}
	if res.Audio.MediaType != "audio/L16;codec=pcm;rate=24000" || string(res.Audio.Data) != "\x01\x02\x03\x04" {
		t.Errorf("Audio = %q %x, want raw pcm", res.Audio.MediaType, res.Audio.Data)
	}
	equalJSON(t, `{"contents":[{"role":"user","parts":[{"text":"softly: hi"}]}],"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}}}`, srv.last(t).Body)
}

func TestSpeechModel_errors(t *testing.T) {
	tests := []struct {
		name  string
		model string
		reply string
		req   ai.SpeechRequest
		want  string
	}{
		{name: "no audio", model: "gemini-3.8-flash-tts", reply: fixture(t, "tts_empty.json"), req: ai.SpeechRequest{Text: "hi"}, want: "no audio (other)"},
		{name: "unknown format", model: "gemini-3.8-flash-tts", reply: fixture(t, "tts_wav.json"), req: ai.SpeechRequest{Text: "hi", Format: "mp3"}, want: "unknown speech format"},
		{name: "legacy format", model: "gemini-2.5-pro-preview-tts", reply: fixture(t, "tts_wav.json"), req: ai.SpeechRequest{Text: "hi", Format: "mulaw"}, want: "PCM or WAV"},
		{name: "turns without speakers", model: "gemini-3.8-flash-tts", reply: fixture(t, "tts_wav.json"), req: ai.SpeechRequest{Text: "conversation", Provider: ai.Providers(google.SpeechOptions{Turns: []google.SpeechTurn{{Speaker: "a", Text: "b"}}})}, want: "need speakers"},
		{name: "three speakers", model: "gemini-3.8-flash-tts", reply: fixture(t, "tts_wav.json"), req: ai.SpeechRequest{Text: "hi", Provider: ai.Providers(google.SpeechOptions{Speakers: make([]google.SpeechSpeaker, 3)})}, want: "exactly 2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.reply))
			tt.req.MaxRetries = new(0)
			_, err := ai.GenerateSpeech(t.Context(), srv.provider().SpeechModel(tt.model), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateSpeech() error = %v, want %q", err, tt.want)
			}
		})
	}
}
