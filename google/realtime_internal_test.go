package google

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
)

func FuzzDecodeLiveMessage(f *testing.F) {
	file, err := os.Open(filepath.Join("testdata", "realtime", "server_messages.jsonl"))
	if err != nil {
		f.Fatalf("open fixture: %v", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		f.Add(scanner.Bytes())
	}
	if err := scanner.Err(); err != nil {
		f.Fatalf("read fixture: %v", err)
	}
	for _, seed := range []string{``, `{}`, `null`, `[`, `{"serverContent":null}`, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"data":"!!"}}]}}}`, `{"goAway":{"timeLeft":"soon"}}`, `{"toolCall":{"functionCalls":[{}]}}`, `{"usageMetadata":"x"}`, `{"serverContent":{"turnComplete":true,"inputTranscription":{"text":"a"}}}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var s liveState
		events, err := decodeLiveMessage(data, &s)
		if err != nil && len(events) != 0 {
			t.Fatalf("decodeLiveMessage returned events %v together with error %v", events, err)
		}
	})
}
