package ai_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type sentiment string

const (
	positive sentiment = "positive"
	negative sentiment = "negative"
)

func TestGenerateObject(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		want    person
		wantErr bool
	}{
		{name: "valid", text: `{"name":"Alice","age":30}`, want: person{Name: "Alice", Age: 30}},
		{name: "surrounding whitespace", text: "  {\"name\":\"Bob\",\"age\":1}\n", want: person{Name: "Bob", Age: 1}},
		{name: "wrong type", text: `{"name":"Alice","age":"thirty"}`, wantErr: true},
		{name: "missing field", text: `{"name":"Alice"}`, wantErr: true},
		{name: "extra field", text: `{"name":"Alice","age":3,"x":1}`, wantErr: true},
		{name: "not JSON", text: `Sure! Here is the person.`, wantErr: true},
		{name: "truncated", text: `{"name":"Al`, wantErr: true},
		{name: "empty", text: ``, wantErr: true},
		{name: "array instead of object", text: `[]`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewModel(aitest.Turn(aitest.Text(tt.text)))
			got, res, err := ai.GenerateObject[person](t.Context(), m, ai.TextRequest{})
			if tt.wantErr {
				outErr, ok := errors.AsType[*ai.OutputError](err)
				if !ok || outErr.Text != tt.text {
					t.Fatalf("error = %v, want an OutputError carrying the text %q", err, tt.text)
				}
				if res == nil || res.Text != tt.text {
					t.Errorf("result missing on output error: %+v", res)
				}
				return
			}
			if err != nil {
				t.Fatalf("GenerateObject() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("GenerateObject() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestGenerateObject_sendsSchemaToModel(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"A","age":1}`)))
	if _, _, err := ai.GenerateObject[person](t.Context(), m, ai.TextRequest{}); err != nil {
		t.Fatalf("GenerateObject() error = %v", err)
	}
	spec := m.Calls()[0].Output
	if spec == nil {
		t.Fatal("call has no Output spec")
	}
	var s map[string]any
	mustUnmarshal(t, spec.Schema, &s)
	if s["type"] != "object" || !cmp.Equal(s["required"], []any{"name", "age"}) {
		t.Errorf("schema = %s, want an object schema requiring name and age", spec.Schema)
	}
}

func TestGenerateObject_doesNotMutateRequest(t *testing.T) {
	req := ai.TextRequest{}
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"A","age":1}`)))
	if _, _, err := ai.GenerateObject[person](t.Context(), m, req); err != nil {
		t.Fatalf("GenerateObject() error = %v", err)
	}
	if req.Output != nil {
		t.Error("GenerateObject set Output on the caller's request")
	}
}

func TestGenerateObject_wrapsNonObjectRoots(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":[{"name":"A","age":1},{"name":"B","age":2}]}`)))
	got, _, err := ai.GenerateObject[[]person](t.Context(), m, ai.TextRequest{})
	if err != nil {
		t.Fatalf("GenerateObject() error = %v", err)
	}
	if diff := cmp.Diff([]person{{Name: "A", Age: 1}, {Name: "B", Age: 2}}, got); diff != "" {
		t.Errorf("GenerateObject() mismatch (-want +got):\n%s", diff)
	}
	var s map[string]any
	mustUnmarshal(t, m.Calls()[0].Output.Schema, &s)
	if s["type"] != "object" {
		t.Errorf("sent schema type = %v, want an object wrapper", s["type"])
	}
}

func TestGenerateObject_validatesWrappedElements(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":[{"name":"A","age":"x"}]}`)))
	_, _, err := ai.GenerateObject[[]person](t.Context(), m, ai.TextRequest{})
	if _, ok := errors.AsType[*ai.OutputError](err); !ok {
		t.Errorf("error = %v, want an OutputError", err)
	}
}

func TestGenerateObject_loopErrorSkipsDecoding(t *testing.T) {
	boom := errors.New("boom")
	m := aitest.NewModel(aitest.Turn(aitest.Err(boom)))
	_, res, err := ai.GenerateObject[person](t.Context(), m, ai.TextRequest{MaxRetries: new(0)})
	if !errors.Is(err, boom) || res == nil {
		t.Errorf("GenerateObject() = result %v, error %v; want the model error and a result", res, err)
	}
}

func TestGenerateObject_withToolsFirst(t *testing.T) {
	m := aitest.NewModel(toolTurn("search", `{"query":"q"}`), aitest.Turn(aitest.Text(`{"name":"Z","age":9}`)))
	got, res, err := ai.GenerateObject[person](t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})
	if err != nil || got.Name != "Z" || len(res.Steps) != 2 {
		t.Errorf("GenerateObject() = %+v, steps %d, error %v", got, len(res.Steps), err)
	}
}

func TestGenerateChoice(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		want    sentiment
		wantErr bool
	}{
		{name: "positive", text: `{"result":"positive"}`, want: positive},
		{name: "negative", text: `{"result":"negative"}`, want: negative},
		{name: "not a choice", text: `{"result":"meh"}`, wantErr: true},
		{name: "wrong type", text: `{"result":3}`, wantErr: true},
		{name: "missing result", text: `{}`, wantErr: true},
		{name: "bare string", text: `"positive"`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewModel(aitest.Turn(aitest.Text(tt.text)))
			got, _, err := ai.GenerateChoice(t.Context(), m, ai.TextRequest{}, positive, negative)
			if tt.wantErr {
				if _, ok := errors.AsType[*ai.OutputError](err); !ok {
					t.Fatalf("error = %v, want an OutputError", err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("GenerateChoice() = %q, %v; want %q", got, err, tt.want)
			}
		})
	}
}

func TestGenerateChoice_sendsEnum(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"positive"}`)))
	if _, _, err := ai.GenerateChoice(t.Context(), m, ai.TextRequest{}, positive, negative); err != nil {
		t.Fatalf("GenerateChoice() error = %v", err)
	}
	var s struct {
		Properties struct {
			Result struct {
				Enum []string `json:"enum"`
			} `json:"result"`
		} `json:"properties"`
	}
	mustUnmarshal(t, m.Calls()[0].Output.Schema, &s)
	if diff := cmp.Diff([]string{"positive", "negative"}, s.Properties.Result.Enum); diff != "" {
		t.Errorf("enum mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateChoice_needsChoices(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"x"}`)))
	if _, _, err := ai.GenerateChoice[sentiment](t.Context(), m, ai.TextRequest{}); err == nil {
		t.Error("GenerateChoice() with no choices error = nil, want an error")
	}
}

func TestGenerateJSON(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		want    string
		wantErr bool
	}{
		{name: "object", text: `{"a":1}`, want: `{"a":1}`},
		{name: "array with whitespace", text: " [1, 2] \n", want: `[1, 2]`},
		{name: "scalar", text: `42`, want: `42`},
		{name: "invalid", text: `{a:1}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := aitest.NewModel(aitest.Turn(aitest.Text(tt.text)))
			got, _, err := ai.GenerateJSON(t.Context(), m, ai.TextRequest{})
			if tt.wantErr {
				if _, ok := errors.AsType[*ai.OutputError](err); !ok {
					t.Fatalf("error = %v, want an OutputError", err)
				}
				return
			}
			if err != nil || string(got) != tt.want {
				t.Errorf("GenerateJSON() = %s, %v; want %s", got, err, tt.want)
			}
			if spec := m.Calls()[0].Output; spec == nil || len(spec.Schema) != 0 {
				t.Errorf("call Output = %+v, want a schema-less JSON spec", spec)
			}
		})
	}
}

func TestDecode_viaRequestOutput(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"Al","age":2}`)), aitest.Turn(aitest.Text(`{"result":"negative"}`)))

	obj := mustGenerate(t, m, ai.TextRequest{Output: ai.ObjectOutput[person]()})
	if got, err := ai.DecodeObject[person](obj); err != nil || got.Name != "Al" {
		t.Errorf("DecodeObject() = %+v, %v", got, err)
	}
	choice := mustGenerate(t, m, ai.TextRequest{Output: ai.ChoiceOutput(positive, negative)})
	if got, err := ai.DecodeChoice[sentiment](choice); err != nil || got != negative {
		t.Errorf("DecodeChoice() = %q, %v", got, err)
	}
}

func TestDecodeChoice_requiresChoiceOutput(t *testing.T) {
	plain := mustGenerate(t, aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"positive"}`))), ai.TextRequest{})
	if _, err := ai.DecodeChoice[sentiment](plain); err == nil {
		t.Error("DecodeChoice on a result without ChoiceOutput error = nil, want an error")
	}
	if _, err := ai.DecodeChoice[sentiment](nil); err == nil {
		t.Error("DecodeChoice(nil) error = nil, want an error")
	}
	if _, err := ai.DecodeObject[person](nil); err == nil {
		t.Error("DecodeObject(nil) error = nil, want an error")
	}
}

func TestOutputError_unwraps(t *testing.T) {
	cause := errors.New("cause")
	err := &ai.OutputError{Text: "t", Err: cause}
	if !errors.Is(err, cause) || err.Error() != "ai: invalid structured output: cause" {
		t.Errorf("OutputError = %q, unwraps = %v", err.Error(), errors.Is(err, cause))
	}
}

func TestStreamObject_partials(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"na`, `me":"Al`, `ice","age":3`, `0}`)))
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{})

	got := slices.Collect(s.Partials())
	want := []person{{}, {Name: "Al"}, {Name: "Alice", Age: 3}, {Name: "Alice", Age: 30}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Partials mismatch (-want +got):\n%s", diff)
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v", err)
	}
	final, res, err := s.Result()
	if err != nil || final != (person{Name: "Alice", Age: 30}) || res.Text != `{"name":"Alice","age":30}` {
		t.Errorf("Result() = %+v, %+v, %v", final, res, err)
	}
}

func TestStreamObject_elements(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"result":[{"name":"A","age":1},`, `{"name":"B"`, `,"age":2},{"name":"C","age":3}]}`)))
	s := ai.StreamObject[[]person](t.Context(), m, ai.TextRequest{})

	var got []person
	for p := range ai.Elements(s) {
		got = append(got, p)
	}
	if diff := cmp.Diff([]person{{Name: "A", Age: 1}, {Name: "B", Age: 2}, {Name: "C", Age: 3}}, got); diff != "" {
		t.Errorf("Elements mismatch (-want +got):\n%s", diff)
	}
	if err := s.Err(); err != nil {
		t.Errorf("Err() = %v", err)
	}
}

func TestStreamObject_elementsStopEarly(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"result":[{"name":"A","age":1},`, `{"name":"B","age":2},{"name":"C","age":3}]}`)))
	s := ai.StreamObject[[]person](t.Context(), m, ai.TextRequest{})
	for range ai.Elements(s) {
		break
	}
	if err := s.Err(); err == nil {
		t.Error("Err() = nil after stopping early, want context.Canceled")
	}
}

func TestStreamObject_invalidFinalOutput(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"name":"Al`, `ice","age":"old"}`)))
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{})
	for range s.Partials() {
	}
	if _, ok := errors.AsType[*ai.OutputError](s.Err()); !ok {
		t.Errorf("Err() = %v, want an OutputError", s.Err())
	}
}

func TestStreamObject_partsAndResult(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"A","age":1}`)))
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{})
	var deltas int
	for p := range s.Parts() {
		if _, ok := p.(ai.TextDelta); ok {
			deltas++
		}
	}
	got, _, err := s.Result()
	if err != nil || got.Name != "A" || deltas != 1 {
		t.Errorf("Result() = %+v, %v after %d deltas", got, err, deltas)
	}
}

func TestStreamObject_singleUse(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"A","age":1}`)))
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{})
	for range s.Partials() {
	}
	for range s.Partials() {
	}
	if !errors.Is(s.Err(), ai.ErrStreamConsumed) {
		t.Errorf("Err() = %v, want %v", s.Err(), ai.ErrStreamConsumed)
	}
}

func TestStreamObject_resetsBetweenSteps(t *testing.T) {
	m := aitest.NewModel(
		aitest.Turn(aitest.Text(`{"name":"draft`), aitest.ToolCall("search", `{"query":"q"}`)),
		aitest.Turn(aitest.TextChunks(`{"name":"Fin`, `al","age":4}`)),
	)
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{Tools: []ai.Tool{searchTool()}})
	var last person
	for p := range s.Partials() {
		last = p
	}
	if last != (person{Name: "Final", Age: 4}) {
		t.Errorf("last partial = %+v, want the final step's object", last)
	}
}

func TestOutputSpecs_encodeAsJSON(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"A","age":1}`)))
	if _, _, err := ai.GenerateObject[person](t.Context(), m, ai.TextRequest{}); err != nil {
		t.Fatalf("GenerateObject() error = %v", err)
	}
	spec := m.Calls()[0].Output
	if spec.Name == "" || !json.Valid(spec.Schema) {
		t.Errorf("spec = %+v, want a name and valid schema JSON", spec)
	}
}

func manyDeltas(n int) []string {
	chunks := make([]string, 0, n+2)
	chunks = append(chunks, `{"name":"`)
	for range n {
		chunks = append(chunks, "abcdefghij")
	}
	return append(chunks, `","age":1}`)
}

func TestStreamObject_partialsAreSparseOnLongStreams(t *testing.T) {
	m := aitest.NewModel(aitest.Turn(aitest.TextChunks(manyDeltas(20000)...)))
	s := ai.StreamObject[person](t.Context(), m, ai.TextRequest{})
	var count int
	var last person
	for p := range s.Partials() {
		count++
		last = p
	}
	if count > 400 {
		t.Errorf("partials = %d for 20000 deltas, want a number logarithmic in the stream size", count)
	}
	if last.Age != 1 || len(last.Name) != 200000 {
		t.Errorf("last partial = age %d, name length %d, want the complete object", last.Age, len(last.Name))
	}
	if _, _, err := s.Result(); err != nil {
		t.Errorf("Result() error = %v", err)
	}
}

func BenchmarkStreamObject_partials(b *testing.B) {
	for _, n := range []int{1000, 4000, 16000} {
		chunks := manyDeltas(n)
		b.Run(fmt.Sprintf("deltas=%d", n), func(b *testing.B) {
			for b.Loop() {
				m := aitest.NewModel(aitest.Turn(aitest.TextChunks(chunks...)))
				s := ai.StreamObject[person](b.Context(), m, ai.TextRequest{})
				for range s.Partials() {
				}
			}
			b.SetBytes(int64(n) * 10)
		})
	}
}
