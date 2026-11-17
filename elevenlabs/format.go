package elevenlabs

import (
	"errors"
	"fmt"
	"strings"
)

var errNoVoice = errors.New("elevenlabs: speech: voice is required, pass a voice id")

type outputFormat struct {
	query     string
	mediaType string
}

func parseFormat(format string) (outputFormat, error) {
	if format == "" {
		return outputFormat{mediaType: "audio/mpeg"}, nil
	}
	codec, _, _ := strings.Cut(format, "_")
	switch codec {
	case "mp3":
		return outputFormat{query: format, mediaType: "audio/mpeg"}, nil
	case "opus":
		return outputFormat{query: format, mediaType: "audio/opus"}, nil
	case "pcm":
		return outputFormat{query: format, mediaType: "audio/pcm"}, nil
	case "wav":
		return outputFormat{query: format, mediaType: "audio/wav"}, nil
	case "ulaw":
		return outputFormat{query: format, mediaType: "audio/basic"}, nil
	case "alaw":
		return outputFormat{query: format, mediaType: "audio/x-alaw-basic"}, nil
	default:
		return outputFormat{}, fmt.Errorf("elevenlabs: unknown output format %q, use codec_samplerate[_bitrate] such as mp3_44100_128", format)
	}
}
