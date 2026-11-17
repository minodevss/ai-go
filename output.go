package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/minodevss/ai-go/internal/schema"
)

const resultKey = "result"

type Output interface {
	spec() (outputSpec, error)
}

type outputKind int

const (
	outputObject outputKind = iota + 1
	outputChoice
	outputJSON
)

type outputSpec struct {
	kind    outputKind
	schema  json.RawMessage
	choices []string
}

type objectOutput[T any] struct{}

type choiceOutput[T ~string] struct{ choices []T }

type jsonOutput struct{}

func ObjectOutput[T any]() Output { return objectOutput[T]{} }

func ChoiceOutput[T ~string](choices ...T) Output {
	return choiceOutput[T]{choices: slices.Clone(choices)}
}

func JSONOutput() Output { return jsonOutput{} }

func (objectOutput[T]) spec() (outputSpec, error) {
	raw, err := schema.For[T]()
	if err != nil {
		return outputSpec{}, fmt.Errorf("ai: output schema: %w", err)
	}
	wrapped, err := wrapIfNeeded(raw)
	if err != nil {
		return outputSpec{}, err
	}
	return outputSpec{kind: outputObject, schema: wrapped}, nil
}

func (o choiceOutput[T]) spec() (outputSpec, error) {
	if len(o.choices) == 0 {
		return outputSpec{}, errors.New("ai: choice output needs at least one choice")
	}
	choices := make([]string, len(o.choices))
	for i, c := range o.choices {
		choices[i] = string(c)
	}
	inner, err := json.Marshal(map[string]any{"type": "string", "enum": choices})
	if err != nil {
		return outputSpec{}, fmt.Errorf("ai: output schema: %w", err)
	}
	wrapped, err := wrapSchema(inner)
	if err != nil {
		return outputSpec{}, err
	}
	return outputSpec{kind: outputChoice, schema: wrapped, choices: choices}, nil
}

func (jsonOutput) spec() (outputSpec, error) { return outputSpec{kind: outputJSON}, nil }

func (s outputSpec) call() *OutputSpec {
	return &OutputSpec{Name: "response", Description: "The response as JSON.", Schema: slices.Clone(s.schema)}
}

func wrapIfNeeded(raw json.RawMessage) (json.RawMessage, error) {
	if rootIsObject(raw) {
		return raw, nil
	}
	return wrapSchema(raw)
}

func wrapSchema(inner json.RawMessage) (json.RawMessage, error) {
	out, err := json.Marshal(map[string]any{
		"type":                 "object",
		"properties":           map[string]json.RawMessage{resultKey: inner},
		"required":             []string{resultKey},
		"additionalProperties": false,
	})
	if err != nil {
		return nil, fmt.Errorf("ai: output schema: %w", err)
	}
	return out, nil
}

func rootIsObject(raw json.RawMessage) bool {
	var root struct {
		Type json.RawMessage `json:"type"`
	}
	if err := json.Unmarshal(raw, &root); err != nil {
		return false
	}
	return string(root.Type) == `"object"`
}

func unwrapResult(doc []byte) (json.RawMessage, error) {
	var wrapper map[string]json.RawMessage
	if err := json.Unmarshal(doc, &wrapper); err != nil {
		return nil, err
	}
	inner, ok := wrapper[resultKey]
	if !ok {
		return nil, errors.New("ai: missing \"result\" field")
	}
	return inner, nil
}

func DecodeObject[T any](res *TextResult) (T, error) {
	var v T
	if res == nil {
		return v, errors.New("ai: decode object: nil result")
	}
	raw, err := schema.For[T]()
	if err != nil {
		return v, fmt.Errorf("ai: output schema: %w", err)
	}
	validator, err := schema.Compile(raw)
	if err != nil {
		return v, fmt.Errorf("ai: output schema: %w", err)
	}
	doc := json.RawMessage(strings.TrimSpace(res.Text))
	if !rootIsObject(raw) {
		doc, err = unwrapResult(doc)
		if err != nil {
			return *new(T), &OutputError{Text: res.Text, Err: err}
		}
	}
	if err := decodeInto(validator, doc, &v); err != nil {
		return *new(T), &OutputError{Text: res.Text, Err: err}
	}
	return v, nil
}

func decodeInto(validator *schema.Validator, doc []byte, dst any) error {
	if err := validator.Validate(doc); err != nil {
		return err
	}
	return json.Unmarshal(doc, dst)
}

func DecodeChoice[T ~string](res *TextResult) (T, error) {
	var zero T
	if res == nil {
		return zero, errors.New("ai: decode choice: nil result")
	}
	choice, ok := res.output.(choiceOutput[T])
	if !ok {
		return zero, errors.New("ai: decode choice: result was not produced with ChoiceOutput of this type")
	}
	inner, err := unwrapResult([]byte(strings.TrimSpace(res.Text)))
	if err != nil {
		return zero, &OutputError{Text: res.Text, Err: err}
	}
	var s T
	if err := json.Unmarshal(inner, &s); err != nil {
		return zero, &OutputError{Text: res.Text, Err: err}
	}
	if !slices.Contains(choice.choices, s) {
		return zero, &OutputError{Text: res.Text, Err: fmt.Errorf("ai: %q is not one of the choices", string(s))}
	}
	return s, nil
}

func GenerateObject[T any](ctx context.Context, m LanguageModel, req TextRequest) (T, *TextResult, error) {
	req.Output = ObjectOutput[T]()
	res, err := GenerateText(ctx, m, req)
	if err != nil {
		return *new(T), res, err
	}
	v, err := DecodeObject[T](res)
	return v, res, err
}

func GenerateChoice[T ~string](ctx context.Context, m LanguageModel, req TextRequest, choices ...T) (T, *TextResult, error) {
	req.Output = ChoiceOutput(choices...)
	res, err := GenerateText(ctx, m, req)
	if err != nil {
		return *new(T), res, err
	}
	v, err := DecodeChoice[T](res)
	return v, res, err
}

func GenerateJSON(ctx context.Context, m LanguageModel, req TextRequest) (json.RawMessage, *TextResult, error) {
	req.Output = JSONOutput()
	res, err := GenerateText(ctx, m, req)
	if err != nil {
		return nil, res, err
	}
	text := strings.TrimSpace(res.Text)
	if !json.Valid([]byte(text)) {
		return nil, res, &OutputError{Text: res.Text, Err: errors.New("ai: output is not valid JSON")}
	}
	return json.RawMessage(text), res, nil
}
