package partialjson

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
)

const DefaultMaxDepth = 64

var ErrTooDeep = errors.New("partialjson: nesting too deep")

type mode int

const (
	modeValue mode = iota
	modeFirstKey
	modeKey
	modeColon
	modeAfterValue
	modeString
	modeNumber
	modeLiteral
	modeDone
)

type Parser struct {
	maxDepth int
	buf      []byte
	stack    []byte
	mode     mode
	keyStr   bool
	escape   int
	escStart int
	tokStart int
	num      numState
	literal  string
	safe     int
	err      error
}

func New(maxDepth int) *Parser {
	if maxDepth <= 0 {
		maxDepth = DefaultMaxDepth
	}
	return &Parser{maxDepth: maxDepth}
}

func (p *Parser) Write(chunk string) error {
	if p.err != nil {
		return p.err
	}
	for i := range len(chunk) {
		if err := p.step(chunk[i]); err != nil {
			p.err = err
			return err
		}
	}
	return nil
}

func (p *Parser) Len() int { return len(p.buf) }

func (p *Parser) Complete() bool {
	if p.err != nil {
		return false
	}
	if p.mode == modeNumber && len(p.stack) == 0 {
		return p.num.complete()
	}
	return p.mode == modeDone
}

func (p *Parser) Partial() []byte {
	if p.err != nil {
		return nil
	}
	cut := p.safe
	tail := ""
	switch p.mode {
	case modeString:
		if !p.keyStr {
			cut = p.stringEnd()
			tail = `"`
		}
	case modeNumber:
		if drop := p.num.trim(); drop >= 0 {
			cut = len(p.buf) - drop
		}
	case modeDone:
		cut = len(p.buf)
	case modeValue, modeFirstKey, modeKey, modeColon, modeAfterValue, modeLiteral:
	}
	if cut == 0 && tail == "" {
		return nil
	}
	out := make([]byte, 0, cut+len(tail)+len(p.stack))
	out = append(out, p.buf[:cut]...)
	out = append(out, tail...)
	for _, open := range slices.Backward(p.stack) {
		if open == '{' {
			out = append(out, '}')
		} else {
			out = append(out, ']')
		}
	}
	return bytes.Trim(out, " \t\r\n")
}

func (p *Parser) stringEnd() int {
	switch {
	case p.escape == 1:
		return len(p.buf) - 1
	case p.escape > 1:
		return p.escStart
	default:
		return len(p.buf)
	}
}

func (p *Parser) step(c byte) error {
	switch p.mode {
	case modeString:
		return p.inString(c)
	case modeNumber:
		if next, ok := p.num.next(c); ok {
			p.buf = append(p.buf, c)
			p.num = next
			return nil
		}
		if !p.num.complete() {
			return unexpected(c)
		}
		p.valueDone()
		return p.step(c)
	case modeLiteral:
		return p.inLiteral(c)
	case modeValue, modeFirstKey, modeKey, modeColon, modeAfterValue, modeDone:
	}
	if isSpace(c) {
		p.buf = append(p.buf, c)
		return nil
	}
	switch p.mode {
	case modeValue:
		return p.beginValue(c)
	case modeFirstKey:
		if c == '}' {
			return p.close(c, '{')
		}
		return p.beginKey(c)
	case modeKey:
		return p.beginKey(c)
	case modeColon:
		if c != ':' {
			return unexpected(c)
		}
		p.buf = append(p.buf, c)
		p.mode = modeValue
		return nil
	case modeAfterValue:
		return p.afterValue(c)
	case modeDone:
		return unexpected(c)
	case modeString, modeNumber, modeLiteral:
	}
	return unexpected(c)
}

func (p *Parser) beginKey(c byte) error {
	if c != '"' {
		return unexpected(c)
	}
	p.buf = append(p.buf, c)
	p.mode = modeString
	p.keyStr = true
	return nil
}

func (p *Parser) beginValue(c byte) error {
	switch {
	case c == '{' || c == '[':
		if len(p.stack) >= p.maxDepth {
			return ErrTooDeep
		}
		p.buf = append(p.buf, c)
		p.stack = append(p.stack, c)
		p.safe = len(p.buf)
		if c == '{' {
			p.mode = modeFirstKey
		} else {
			p.mode = modeValue
		}
	case c == ']' && p.inArray() && p.afterOpen():
		return p.close(c, '[')
	case c == '"':
		p.buf = append(p.buf, c)
		p.mode = modeString
		p.keyStr = false
	case c == '-' || (c >= '0' && c <= '9'):
		p.buf = append(p.buf, c)
		p.mode = modeNumber
		p.num = numStart(c)
	case c == 't' || c == 'f' || c == 'n':
		p.buf = append(p.buf, c)
		p.mode = modeLiteral
		p.tokStart = len(p.buf) - 1
		p.literal = map[byte]string{'t': "true", 'f': "false", 'n': "null"}[c]
	default:
		return unexpected(c)
	}
	return nil
}

func (p *Parser) afterOpen() bool {
	return p.safe > 0 && p.safe == len(bytes.TrimRight(p.buf, " \t\r\n")) && p.buf[p.safe-1] == '['
}

func (p *Parser) inArray() bool {
	return len(p.stack) > 0 && p.stack[len(p.stack)-1] == '['
}

func (p *Parser) afterValue(c byte) error {
	if len(p.stack) == 0 {
		return unexpected(c)
	}
	top := p.stack[len(p.stack)-1]
	switch c {
	case ',':
		p.buf = append(p.buf, c)
		if top == '{' {
			p.mode = modeKey
		} else {
			p.mode = modeValue
		}
		return nil
	case '}':
		return p.close(c, '{')
	case ']':
		return p.close(c, '[')
	default:
		return unexpected(c)
	}
}

func (p *Parser) close(c, want byte) error {
	if len(p.stack) == 0 || p.stack[len(p.stack)-1] != want {
		return unexpected(c)
	}
	p.buf = append(p.buf, c)
	p.stack = p.stack[:len(p.stack)-1]
	p.valueDone()
	return nil
}

func (p *Parser) valueDone() {
	p.safe = len(p.buf)
	if len(p.stack) == 0 {
		p.mode = modeDone
		return
	}
	p.mode = modeAfterValue
}

func (p *Parser) inString(c byte) error {
	switch {
	case p.escape == 1:
		if !isEscapeByte(c) {
			return unexpected(c)
		}
		p.buf = append(p.buf, c)
		if c == 'u' {
			p.escape = 2
		} else {
			p.escape = 0
		}
	case p.escape >= 2:
		if !isHex(c) {
			return unexpected(c)
		}
		p.buf = append(p.buf, c)
		p.escape++
		if p.escape == 6 {
			p.escape = 0
		}
	case c == '\\':
		p.escStart = len(p.buf)
		p.buf = append(p.buf, c)
		p.escape = 1
	case c == '"':
		p.buf = append(p.buf, c)
		if p.keyStr {
			p.keyStr = false
			p.mode = modeColon
			return nil
		}
		p.valueDone()
	case c < 0x20:
		return unexpected(c)
	default:
		p.buf = append(p.buf, c)
	}
	return nil
}

func (p *Parser) inLiteral(c byte) error {
	n := len(p.buf) - p.tokStart
	if c != p.literal[n] {
		return unexpected(c)
	}
	p.buf = append(p.buf, c)
	if n+1 == len(p.literal) {
		p.valueDone()
	}
	return nil
}

type numState int

const (
	numSign numState = iota + 1
	numZero
	numInt
	numDot
	numFrac
	numExp
	numExpSign
	numExpDigits
)

func numStart(c byte) numState {
	switch c {
	case '-':
		return numSign
	case '0':
		return numZero
	default:
		return numInt
	}
}

func (s numState) next(c byte) (numState, bool) {
	digit := c >= '0' && c <= '9'
	switch s {
	case numSign:
		if c == '0' {
			return numZero, true
		}
		if digit {
			return numInt, true
		}
	case numZero:
		return s.afterInt(c, false)
	case numInt:
		return s.afterInt(c, digit)
	case numDot, numFrac:
		if digit {
			return numFrac, true
		}
		if s == numFrac && (c == 'e' || c == 'E') {
			return numExp, true
		}
	case numExp:
		if c == '+' || c == '-' {
			return numExpSign, true
		}
		if digit {
			return numExpDigits, true
		}
	case numExpSign, numExpDigits:
		if digit {
			return numExpDigits, true
		}
	}
	return s, false
}

func (s numState) afterInt(c byte, digit bool) (numState, bool) {
	switch {
	case digit:
		return numInt, true
	case c == '.':
		return numDot, true
	case c == 'e' || c == 'E':
		return numExp, true
	default:
		return s, false
	}
}

func (s numState) complete() bool {
	return s == numZero || s == numInt || s == numFrac || s == numExpDigits
}

func (s numState) trim() int {
	switch s {
	case numDot, numExp:
		return 1
	case numExpSign:
		return 2
	case numSign:
		return -1
	default:
		return 0
	}
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isHex(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

func isEscapeByte(c byte) bool {
	switch c {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u':
		return true
	default:
		return false
	}
}

func unexpected(c byte) error {
	return fmt.Errorf("partialjson: unexpected %q", c)
}
