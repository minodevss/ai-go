# ai-go

A Go SDK for AI models: text generation and streaming, tools, structured output, embeddings, media, realtime, and a test kit. Providers are implemented directly on `net/http`.

## Install

```sh
go get github.com/minodevss/ai-go
```

Requires Go 1.26.

## Generate text

```go
res, err := ai.GenerateText(ctx, openai.Model("gpt-5-mini"), ai.TextRequest{
	Instructions: ai.Texts("You are a support agent."),
	Messages:     []ai.Message{ai.User("Where is my order?")},
})
if err != nil {
	return err
}
fmt.Println(res.Text)
```

## Stream text

```go
s := ai.StreamText(ctx, anthropic.Model("claude-opus-5-5"), ai.TextRequest{
	Messages: []ai.Message{ai.User("Write a haiku about Go.")},
})
for text := range s.Text() {
	fmt.Print(text)
}
if err := s.Err(); err != nil {
	return err
}
```

## Providers

| Package | Provider |
|---|---|
| `openai` | OpenAI |
| `anthropic` | Anthropic |
| `google` | Google AI Studio and Vertex AI |
| `bedrock` | Amazon Bedrock |
| `cohere` | Cohere |
| `ollama` | Ollama |
| `voyage`, `jina` | Voyage AI, Jina |
| `deepgram`, `assemblyai`, `elevenlabs` | Speech and transcription |
| `fal`, `replicate`, `bfl`, `luma` | Image and video |
| `openaicompat` | Azure OpenAI, Mistral, DeepSeek, Perplexity, OpenRouter, xAI, Groq, Together, Fireworks, Cerebras, vLLM |

Separate modules: `aiotel` (OpenTelemetry), `aimcp` (MCP tools), `uiredis` (Redis resume store for `ui`).

## Documentation

See [docs/design.md](docs/design.md) for the API reference and [docs/style/go.md](docs/style/go.md) for code style.
