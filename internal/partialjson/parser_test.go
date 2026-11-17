package partialjson_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go/internal/partialjson"
)

func feed(t *testing.T, chunks ...string) *partialjson.Parser {
	t.Helper()
	p := partialjson.New(0)
	for _, c := range chunks {
		if err := p.Write(c); err != nil {
			t.Fatalf("Write(%q) error = %v", c, err)
		}
	}
	return p
}

func TestParser_partial(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "empty", in: "", want: ""},
		{name: "whitespace only", in: "  \n", want: ""},
		{name: "open object", in: `{`, want: `{}`},
		{name: "open array", in: `[`, want: `[]`},
		{name: "key in progress", in: `{"na`, want: `{}`},
		{name: "key without colon", in: `{"a"`, want: `{}`},
		{name: "key with colon", in: `{"a":`, want: `{}`},
		{name: "value after complete pair", in: `{"a":1,"b":`, want: `{"a":1}`},
		{name: "second key in progress", in: `{"a":1,"b`, want: `{"a":1}`},
		{name: "trailing comma object", in: `{"a":1,`, want: `{"a":1}`},
		{name: "string value in progress", in: `{"a":"x`, want: `{"a":"x"}`},
		{name: "empty string value in progress", in: `{"a":"`, want: `{"a":""}`},
		{name: "dangling escape", in: `{"a":"x\`, want: `{"a":"x"}`},
		{name: "partial unicode escape", in: `{"a":"x\u00`, want: `{"a":"x"}`},
		{name: "complete unicode escape", in: `{"a":"é`, want: `{"a":"é"}`},
		{name: "escaped quote", in: `{"a":"say \"hi`, want: `{"a":"say \"hi"}`},
		{name: "partial true", in: `{"a":tru`, want: `{}`},
		{name: "partial null in array", in: `[1,nu`, want: `[1]`},
		{name: "complete literal", in: `{"a":true,"b":nul`, want: `{"a":true}`},
		{name: "number in progress", in: `[1,2`, want: `[1,2]`},
		{name: "number after colon", in: `{"a":12`, want: `{"a":12}`},
		{name: "lone minus", in: `[1,-`, want: `[1]`},
		{name: "trailing dot", in: `[1.`, want: `[1]`},
		{name: "trailing exponent", in: `[1e`, want: `[1]`},
		{name: "exponent sign", in: `[1e+`, want: `[1]`},
		{name: "fraction", in: `[1.5`, want: `[1.5]`},
		{name: "nested open", in: `{"a":[1,{"b":`, want: `{"a":[1,{}]}`},
		{name: "nested arrays", in: `[[],[1,`, want: `[[],[1]]`},
		{name: "top level string", in: `"abc`, want: `"abc"`},
		{name: "top level number", in: `12`, want: `12`},
		{name: "top level lone minus", in: `-`, want: ``},
		{name: "complete object", in: `{"a":[1,2],"b":{"c":null}}`, want: `{"a":[1,2],"b":{"c":null}}`},
		{name: "complete with trailing space", in: ` {"a":1}  `, want: `{"a":1}`},
		{name: "empty containers", in: `{"a":[],"b":{}`, want: `{"a":[],"b":{}}`},
		{name: "array of strings", in: `["a","b`, want: `["a","b"]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := feed(t, tt.in)
			got := string(p.Partial())
			if got != tt.want {
				t.Errorf("Partial(%q) = %q, want %q", tt.in, got, tt.want)
			}
			if got != "" && !json.Valid([]byte(got)) {
				t.Errorf("Partial(%q) = %q is not valid JSON", tt.in, got)
			}
		})
	}
}

func TestParser_complete(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{in: `{"a":1}`, want: true},
		{in: `{"a":1} `, want: true},
		{in: `[1,2]`, want: true},
		{in: `"x"`, want: true},
		{in: `42`, want: true},
		{in: `true`, want: true},
		{in: `{"a":1`, want: false},
		{in: `[1,2`, want: false},
		{in: `"x`, want: false},
		{in: `4.`, want: false},
		{in: ``, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			if got := feed(t, tt.in).Complete(); got != tt.want {
				t.Errorf("Complete(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestParser_invalid(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{name: "bare key", in: `{a`},
		{name: "missing comma", in: `[1 2]`},
		{name: "trailing comma object", in: `{"a":1,}`},
		{name: "trailing comma array", in: `[1,]`},
		{name: "leading zero", in: `01`},
		{name: "double minus", in: `--1`},
		{name: "double dot", in: `1.2.3`},
		{name: "bad literal", in: `nul1`},
		{name: "control char in string", in: "\"a\nb\""},
		{name: "mismatched close", in: `[1}`},
		{name: "close without open", in: `]`},
		{name: "trailing garbage", in: `{} x`},
		{name: "second value", in: `1 2`},
		{name: "bad escape", in: `"\x"`},
		{name: "bad unicode escape", in: `"\u00zz"`},
		{name: "colon in array", in: `[1:2]`},
		{name: "missing colon", in: `{"a" 1}`},
		{name: "empty key value", in: `{"a":,}`},
		{name: "number then letter", in: `[1a]`},
		{name: "exponent without digits", in: `[1e]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := partialjson.New(0)
			if err := p.Write(tt.in); err == nil {
				t.Fatalf("Write(%q) error = nil, want an error", tt.in)
			}
			if p.Partial() != nil || p.Complete() {
				t.Errorf("failed parser still reports output: Partial = %q, Complete = %v", p.Partial(), p.Complete())
			}
			if err := p.Write("1"); err == nil {
				t.Error("Write after an error succeeded, want the sticky error")
			}
		})
	}
}

func TestParser_chunkingDoesNotMatter(t *testing.T) {
	doc := `{"name":"Alé\"x","tags":["a","b",{"deep":[1,2.5e3,-4]}],"ok":true,"none":null}`
	for cut := 0; cut <= len(doc); cut++ {
		whole := feed(t, doc[:cut])
		split := feed(t, doc[:cut/2], doc[cut/2:cut])
		bytewise := partialjson.New(0)
		for i := range cut {
			if err := bytewise.Write(doc[i : i+1]); err != nil {
				t.Fatalf("Write(%d) error = %v", i, err)
			}
		}
		if diff := cmp.Diff(string(whole.Partial()), string(split.Partial())); diff != "" {
			t.Errorf("cut %d: split differs from whole (-whole +split):\n%s", cut, diff)
		}
		if diff := cmp.Diff(string(whole.Partial()), string(bytewise.Partial())); diff != "" {
			t.Errorf("cut %d: bytewise differs from whole (-whole +bytewise):\n%s", cut, diff)
		}
	}
}

func TestParser_everyPrefixOfValidJSONIsValid(t *testing.T) {
	doc := `{"a":[1,-2.5e-3,"x\\y",{"b":[true,false,null,[]]}],"c":{"d":""}}`
	for i := range len(doc) + 1 {
		p := feed(t, doc[:i])
		if out := p.Partial(); out != nil && !json.Valid(out) {
			t.Errorf("prefix %q gave invalid partial %q", doc[:i], out)
		}
	}
}

func TestParser_depthCap(t *testing.T) {
	tests := []struct {
		name     string
		depth    int
		maxDepth int
		wantErr  bool
	}{
		{name: "default allows 64", depth: 64, maxDepth: 0},
		{name: "default rejects 65", depth: 65, maxDepth: 0, wantErr: true},
		{name: "custom cap allows", depth: 3, maxDepth: 3},
		{name: "custom cap rejects", depth: 4, maxDepth: 3, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := partialjson.New(tt.maxDepth).Write(strings.Repeat("[", tt.depth))
			if (err != nil) != tt.wantErr {
				t.Fatalf("Write() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr && !errors.Is(err, partialjson.ErrTooDeep) {
				t.Errorf("error = %v, want %v", err, partialjson.ErrTooDeep)
			}
		})
	}
}

func TestParser_hugeInputWithinDepthCap(t *testing.T) {
	var b strings.Builder
	b.WriteString("[")
	for i := range 20000 {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`{"k":"value","n":123}`)
	}
	p := feed(t, b.String())
	if out := p.Partial(); !json.Valid(out) {
		t.Error("partial of a large document is not valid JSON")
	}
}

func BenchmarkParser_write(b *testing.B) {
	doc := strings.Repeat(`{"key":"value","n":[1,2,3]},`, 2000)
	b.SetBytes(int64(len(doc)))
	for b.Loop() {
		p := partialjson.New(0)
		if err := p.Write("[" + doc[:len(doc)-1] + "]"); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzParser(f *testing.F) {
	for _, seed := range []string{
		`{"a":[1,2,{"b":"cé"}],"d":true}`,
		`[[[[[[1.5e+3]]]]]]`,
		`{"a":"x\`,
		`{"a":tru`,
		`-0.0e-0`,
		`"😀"`,
		`{"a":1,}`,
		`[1 2]`,
		`{"k":[{"k":[{"k":[]}]}]}`,
	} {
		f.Add([]byte(seed), uint8(3))
	}
	f.Fuzz(func(t *testing.T, data []byte, step uint8) {
		size := int(step%8) + 1
		chunked := partialjson.New(8)
		var chunkErr error
		for i := 0; i < len(data) && chunkErr == nil; i += size {
			chunkErr = chunked.Write(string(data[i:min(i+size, len(data))]))
		}
		whole := partialjson.New(8)
		wholeErr := whole.Write(string(data))
		if (chunkErr != nil) != (wholeErr != nil) {
			t.Fatalf("chunked error %v, whole error %v", chunkErr, wholeErr)
		}
		if wholeErr != nil {
			return
		}
		got, want := chunked.Partial(), whole.Partial()
		if string(got) != string(want) {
			t.Fatalf("chunked %q != whole %q", got, want)
		}
		if want != nil && !json.Valid(want) {
			t.Fatalf("Partial(%q) = %q is not valid JSON", data, want)
		}
		if json.Valid(data) {
			if !whole.Complete() {
				t.Fatalf("Complete() = false for valid JSON %q", data)
			}
			var a, b any
			if json.Unmarshal(data, &a) != nil {
				return
			}
			if err := json.Unmarshal(want, &b); err != nil {
				t.Fatalf("Unmarshal(partial) error = %v", err)
			}
			if !cmp.Equal(a, b) {
				t.Fatalf("complete document changed: %q -> %q", data, want)
			}
		}
	})
}
