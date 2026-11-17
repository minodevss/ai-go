// Package openaiwire holds the JSON helpers shared by the OpenAI-style Responses, Chat Completions and Embeddings adapters.
package openaiwire

import (
	"cmp"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"strings"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/schemafit"
	"github.com/minodevss/ai-go/internal/transport"
)

const (
	MissingResult     = "error: this tool call has no result"
	defaultSchemaName = "response"
)

var validCallID = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type SchemaFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitzero"`
	Description string          `json:"description,omitzero"`
	Schema      json.RawMessage `json:"schema,omitzero"`
	Strict      *bool           `json:"strict,omitzero"`
}

type annotation struct {
	Type       string `json:"type"`
	URL        string `json:"url"`
	Title      string `json:"title"`
	Filename   string `json:"filename"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	Nested     *struct {
		URL        string `json:"url"`
		Title      string `json:"title"`
		StartIndex int    `json:"start_index"`
		EndIndex   int    `json:"end_index"`
	} `json:"url_citation"`
}

type usageBody struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	InputDetails     struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	PromptDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	OutputDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	CompletionDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func StrictRules(provider string) schemafit.Rules {
	return schemafit.Rules{
		Provider: provider,
		Unsupported: []string{
			"$schema", "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
			"minLength", "maxLength", "uniqueItems", "contains", "minContains", "maxContains",
			"minProperties", "maxProperties", "propertyNames", "patternProperties",
			"unevaluatedProperties", "unevaluatedItems", "prefixItems", "default", "examples",
		},
		Renamed: map[string]string{"oneOf": "anyOf"},
		Formats: []string{"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"},
		Strict:  true,
	}
}

func Instructions(texts []ai.Text) string {
	parts := make([]string, 0, len(texts))
	for _, t := range texts {
		parts = append(parts, t.Text)
	}
	return strings.Join(parts, "\n\n")
}

func Arguments(in json.RawMessage) string {
	if len(in) == 0 {
		return "{}"
	}
	var s string
	if err := json.Unmarshal(in, &s); err == nil && !json.Valid([]byte(s)) {
		return s
	}
	return string(in)
}

func ToolInput(arguments string) json.RawMessage {
	if strings.TrimSpace(arguments) == "" {
		return json.RawMessage("{}")
	}
	if json.Valid([]byte(arguments)) {
		return json.RawMessage(arguments)
	}
	quoted, err := transport.EncodeJSON(arguments)
	if err != nil {
		return json.RawMessage("{}")
	}
	return quoted
}

func DataURL(mediaType string, data []byte) string {
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func CallID(id string, maxLen int) string {
	if id != "" && len(id) <= maxLen && validCallID.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return "call_" + hex.EncodeToString(sum[:])[:maxLen-len("call_")-8]
}

func Usage(raw json.RawMessage) ai.Usage {
	if len(raw) == 0 || string(raw) == "null" {
		return ai.Usage{}
	}
	var u usageBody
	if err := json.Unmarshal(raw, &u); err != nil {
		return ai.Usage{Raw: raw}
	}
	return ai.Usage{
		Input:     u.InputTokens + u.PromptTokens,
		Output:    u.OutputTokens + u.CompletionTokens,
		CacheRead: u.InputDetails.CachedTokens + u.PromptDetails.CachedTokens,
		Reasoning: u.OutputDetails.ReasoningTokens + u.CompletionDetails.ReasoningTokens,
		Raw:       raw,
	}
}

func JSONSchema(provider string, spec *ai.OutputSpec) (SchemaFormat, []ai.Warning) {
	name := schemaName(spec.Name)
	fit, err := schemafit.Fit(spec.Schema, StrictRules(provider))
	if err != nil {
		return SchemaFormat{Type: "json_schema", Name: name, Description: spec.Description, Schema: spec.Schema, Strict: new(false)}, []ai.Warning{{Message: err.Error()}}
	}
	warnings := fit.Warnings
	if !fit.Strict {
		warnings = append(warnings, ai.Warning{Message: provider + ": structured output is not strict because the schema is not strict-compatible; the model may not follow it exactly"})
	}
	return SchemaFormat{Type: "json_schema", Name: name, Description: spec.Description, Schema: fit.Schema, Strict: &fit.Strict}, warnings
}

func schemaName(name string) string {
	cleaned := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' {
			return r
		}
		return -1
	}, name)
	if cleaned == "" {
		return defaultSchemaName
	}
	return cleaned[:min(len(cleaned), 64)]
}

func Citations(provider string, annotations []json.RawMessage) []ai.Citation {
	var out []ai.Citation
	for _, raw := range annotations {
		var a annotation
		if err := json.Unmarshal(raw, &a); err != nil {
			continue
		}
		if n := a.Nested; n != nil {
			a.URL, a.Title, a.StartIndex, a.EndIndex = n.URL, n.Title, n.StartIndex, n.EndIndex
		}
		out = append(out, ai.Citation{Kind: citationKind(a.Type), URL: a.URL, Title: cmp.Or(a.Title, a.Filename), Start: a.StartIndex, End: a.EndIndex, Provider: ai.ProviderData{provider: raw}})
	}
	return out
}

func citationKind(t string) string {
	switch t {
	case "url_citation":
		return "url"
	case "file_citation":
		return "file"
	case "container_file_citation":
		return "container-file"
	case "file_path":
		return "file-path"
	default:
		return t
	}
}

type FileRef struct {
	FileID string `json:"fileId,omitzero"`
	Detail string `json:"detail,omitzero"`
}

func EncodeFileRef(r FileRef) json.RawMessage {
	raw, err := transport.EncodeJSON(r)
	if err != nil {
		panic("openaiwire: encode file ref: " + err.Error())
	}
	return raw
}

func FileRefOf(provider string, d ai.ProviderData) FileRef {
	var out FileRef
	if raw, ok := d[provider]; ok {
		if err := json.Unmarshal(raw, &out); err != nil {
			return FileRef{}
		}
	}
	return out
}
