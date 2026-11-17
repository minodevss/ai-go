package openai

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func chunksOf(pieces ...string) []string {
	var out []string
	for c := range speechChunks(func(yield func(string) bool) {
		for _, p := range pieces {
			if !yield(p) {
				return
			}
		}
	}) {
		out = append(out, c)
	}
	return out
}

func TestSpeechChunks(t *testing.T) {
	tests := []struct {
		name   string
		pieces []string
		want   []string
	}{
		{name: "split across pieces", pieces: []string{"Hello there, my dear fr", "iend. How are you today? Fine."}, want: []string{"Hello there, my dear friend.", "How are you today? Fine."}},
		{name: "short sentences are merged until the minimum", pieces: []string{"Hi. Ok. This one is long enough to cut. Next."}, want: []string{"Hi. Ok. This one is long enough to cut.", "Next."}},
		{name: "decimal points do not end a sentence", pieces: []string{"Pi is 3.14159 and e is 2.71828 and more. Done."}, want: []string{"Pi is 3.14159 and e is 2.71828 and more.", "Done."}},
		{name: "terminator at the end waits for more text", pieces: []string{"This sentence is complete.", " And this continues."}, want: []string{"This sentence is complete.", "And this continues."}},
		{name: "cjk punctuation", pieces: []string{"오늘 날씨는 맑고 하늘은 높고 구름 한 점 없이 아주 좋습니다。 내일은 비가 옵니다。"}, want: []string{"오늘 날씨는 맑고 하늘은 높고 구름 한 점 없이 아주 좋습니다。", "내일은 비가 옵니다。"}},
		{name: "newline ends a chunk", pieces: []string{"A title line that is long enough\nbody text here"}, want: []string{"A title line that is long enough", "body text here"}},
		{name: "blank input", pieces: []string{" ", "\n\t"}, want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := chunksOf(tt.pieces...)
			if strings.Join(got, "|") != strings.Join(tt.want, "|") {
				t.Errorf("speechChunks(%q) = %q, want %q", tt.pieces, got, tt.want)
			}
		})
	}
}

func TestSpeechChunks_neverExceedTheLimit(t *testing.T) {
	inputs := []string{
		strings.Repeat("a", 10000),
		strings.Repeat("word ", 3000),
		strings.Repeat("가", 9000),
		strings.Repeat("Sentence number one. ", 600),
	}
	for _, in := range inputs {
		var rebuilt strings.Builder
		for _, c := range chunksOf(in) {
			if n := utf8.RuneCountInString(c); n > maxSpeechChars || n == 0 {
				t.Fatalf("chunk of %d runes for input starting %.20q", n, in)
			}
			if !utf8.ValidString(c) {
				t.Fatalf("chunk is not valid UTF-8 for input starting %.20q", in)
			}
			rebuilt.WriteString(c)
		}
		if strings.Join(strings.Fields(rebuilt.String()), "") != strings.Join(strings.Fields(in), "") {
			t.Errorf("chunks lost or changed text for input starting %.20q", in)
		}
	}
}

func FuzzSpeechChunks(f *testing.F) {
	for _, seed := range []string{"", "Hello. World.", "a b c", strings.Repeat("x", 5000), "가나다。라마바。", "1.2.3. 4", "\n\n\n"} {
		f.Add(seed, 7)
	}
	f.Fuzz(func(t *testing.T, text string, split int) {
		if !utf8.ValidString(text) {
			t.Skip()
		}
		cut := 0
		if len(text) > 0 {
			cut = split % (len(text) + 1)
			if cut < 0 {
				cut = -cut
			}
			for cut < len(text) && !utf8.RuneStart(text[cut]) {
				cut++
			}
		}
		var rebuilt strings.Builder
		for _, c := range chunksOf(text[:cut], text[cut:]) {
			if n := utf8.RuneCountInString(c); n > maxSpeechChars || strings.TrimSpace(c) == "" {
				t.Fatalf("bad chunk of %d runes: %q", n, c)
			}
			rebuilt.WriteString(c)
		}
		if strings.Join(strings.Fields(rebuilt.String()), "") != strings.Join(strings.Fields(text), "") {
			t.Fatalf("chunks changed the text: %q -> %q", text, rebuilt.String())
		}
	})
}
