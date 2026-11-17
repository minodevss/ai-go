package ai

import (
	"slices"
	"strings"
)

type collector struct {
	content  Parts
	blocks   map[string]*block
	lastText int
	end      *StepEnd
}

type block struct {
	index int
	text  strings.Builder
}

func newCollector() *collector {
	return &collector{blocks: map[string]*block{}, lastText: -1}
}

func blockKey(kind, id string) string { return kind + "\x00" + id }

func (c *collector) open(kind, id string, part Part) *block {
	key := blockKey(kind, id)
	if b, ok := c.blocks[key]; ok {
		return b
	}
	b := &block{index: len(c.content)}
	c.content = append(c.content, part)
	c.blocks[key] = b
	if kind == "text" {
		c.lastText = b.index
	}
	return b
}

func (c *collector) add(p StreamPart) {
	switch v := p.(type) {
	case TextStart:
		c.open("text", v.ID, Text{})
	case TextDelta:
		b := c.open("text", v.ID, Text{})
		b.text.WriteString(v.Text)
		t, ok := c.content[b.index].(Text)
		if ok {
			t.Text = b.text.String()
			c.content[b.index] = t
		}
	case TextEnd:
		b := c.open("text", v.ID, Text{})
		if t, ok := c.content[b.index].(Text); ok {
			t.Provider = v.Provider
			c.content[b.index] = t
		}
		c.lastText = b.index
		delete(c.blocks, blockKey("text", v.ID))
	case ReasoningStart:
		c.open("reasoning", v.ID, Reasoning{})
	case ReasoningDelta:
		b := c.open("reasoning", v.ID, Reasoning{})
		b.text.WriteString(v.Text)
		if r, ok := c.content[b.index].(Reasoning); ok {
			r.Text = b.text.String()
			c.content[b.index] = r
		}
	case ReasoningEnd:
		b := c.open("reasoning", v.ID, Reasoning{})
		if r, ok := c.content[b.index].(Reasoning); ok {
			r.Provider = v.Provider
			c.content[b.index] = r
		}
		delete(c.blocks, blockKey("reasoning", v.ID))
	case ToolCall:
		c.content = append(c.content, v)
	case ToolResult:
		c.content = append(c.content, v)
	case File:
		c.content = append(c.content, v)
	case Custom:
		c.content = append(c.content, v)
	case Citation:
		c.cite(v)
	case StepEnd:
		c.end = &v
	default:
	}
}

func (c *collector) cite(v Citation) {
	if c.lastText < 0 {
		return
	}
	if t, ok := c.content[c.lastText].(Text); ok {
		t.Citations = append(slices.Clone(t.Citations), v)
		c.content[c.lastText] = t
	}
}

func (c *collector) parts() Parts {
	out := make(Parts, 0, len(c.content))
	for _, p := range c.content {
		switch v := p.(type) {
		case Text:
			if v.Text == "" && len(v.Provider) == 0 && len(v.Citations) == 0 {
				continue
			}
		case Reasoning:
			if v.Text == "" && len(v.Provider) == 0 {
				continue
			}
		default:
		}
		out = append(out, p)
	}
	return out
}
