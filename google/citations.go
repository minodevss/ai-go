package google

import (
	"encoding/json"
	"strings"

	"github.com/minodevss/ai-go/internal/pdata"

	"github.com/minodevss/ai-go"
)

type grounding struct {
	Chunks   []groundingChunk   `json:"groundingChunks"`
	Supports []groundingSupport `json:"groundingSupports"`
}

type groundingChunk struct {
	Web              *chunkSource `json:"web"`
	RetrievedContext *chunkSource `json:"retrievedContext"`
	Maps             *chunkSource `json:"maps"`
}

type chunkSource struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

type groundingSupport struct {
	Segment struct {
		StartIndex int    `json:"startIndex"`
		EndIndex   int    `json:"endIndex"`
		Text       string `json:"text"`
	} `json:"segment"`
	ChunkIndices []int     `json:"groundingChunkIndices"`
	Scores       []float64 `json:"confidenceScores"`
}

func (c groundingChunk) source() (chunkSource, bool) {
	for _, s := range []*chunkSource{c.Web, c.RetrievedContext, c.Maps} {
		if s != nil {
			return *s, true
		}
	}
	return chunkSource{}, false
}

func (b *backend) citations(texts []string, raw json.RawMessage) [][]ai.Citation {
	out := make([][]ai.Citation, len(texts))
	if len(raw) == 0 {
		return out
	}
	var g grounding
	if err := json.Unmarshal(raw, &g); err != nil {
		return out
	}
	full := strings.Join(texts, "")
	for _, s := range g.Supports {
		start, end, ok := locate(full, s.Segment.StartIndex, s.Segment.EndIndex, s.Segment.Text)
		if !ok {
			continue
		}
		part, offset := partAt(texts, start)
		end = min(end, offset+len(texts[part]))
		for _, idx := range s.ChunkIndices {
			if idx < 0 || idx >= len(g.Chunks) {
				continue
			}
			src, ok := g.Chunks[idx].source()
			if !ok {
				continue
			}
			out[part] = append(out[part], ai.Citation{
				Kind:      "url",
				URL:       src.URI,
				Title:     src.Title,
				CitedText: full[start:end],
				Start:     start - offset,
				End:       end - offset,
				Provider:  pdata.Encode(b.name, citationData{ChunkIndex: idx, Confidence: s.Scores}),
			})
		}
	}
	return out
}

func locate(full string, start, end int, text string) (int, int, bool) {
	if start >= 0 && end > start && end <= len(full) && (text == "" || full[start:end] == text) {
		return start, end, true
	}
	if text == "" {
		return 0, 0, false
	}
	idx := strings.Index(full, text)
	if idx < 0 {
		return 0, 0, false
	}
	return idx, idx + len(text), true
}

func partAt(texts []string, pos int) (int, int) {
	offset := 0
	for i, t := range texts {
		if pos < offset+len(t) {
			return i, offset
		}
		offset += len(t)
	}
	return len(texts) - 1, offset - len(texts[len(texts)-1])
}
