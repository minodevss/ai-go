# Go AI SDK — design v3 (2026-11-17)

Review history: v1 → 5 agents (evidence audit, Go API red-team, provider losslessness, realistic programs, modules/tests) → v2 → 3 agents (contested-decision challenge, security review, full compile of the API with 7 programs under `run/scratch-compile/`) → v3. Decision logs at the end.

## Principles
- Goal: the best AI SDK for Go. Vercel AI SDK (npm `ai` v7) is a source of features and lessons, not a spec to mirror.
- Lossless: appending `res.Messages` to history always makes the next same-provider turn succeed. When `Provider[name]` is present it is the replay source of truth; typed fields are a portable projection.
- Safe by default: anything from a browser, an MCP server, or a tool output is untrusted; side effects need an explicit decision.
- Decisions by Go developers' need + official Go guidance. Implementation cost is never a reason. Exclusions only for: no need / outside identity / redundant / complexity pushed onto every user.
- Min Go 1.26 (two latest majors supported); raised only in minor releases.

## 1. Scope
In: text generate/stream, tools + loop, structured output, embeddings (text + multimodal), rerank, image, video, speech (incl. streaming), transcription (incl. streaming), realtime, batch, files, MCP adapter, middleware, rate limiting, fallback, registry, OpenTelemetry adapter, useChat-compatible UI stream incl. resume, test kit.
Out: Vercel Gateway default provider, durable workflow engine, CLI-agent harnesses, RAG pipelines, vector DBs, prompt templates, chains, frontend hooks, WebRTC client.

## 2. Providers
Own adapters on `net/http` + `encoding/json`; no official SDKs. Internal kit: SSE, AWS event-stream, SigV4, JSON, retry, limiter hook, schema transforms. WebSocket via `coder/websocket` (zero deps).

## 3. Messages
```go
type Role string // "user" | "assistant" | "tool"

type Message struct {
    Role     Role
    Content  Parts
    Provider ProviderData // message-level replay state (e.g. Anthropic container id)
}

type Parts []Part               // UnmarshalJSON with "type" discriminator; normalizes *T to T
type Part interface{ isPart() } // sealed; parts are values

type Text struct       { Text string; Citations []Citation; Provider ProviderData }
type Reasoning struct  { Text string; Provider ProviderData }
type File struct       { MediaType, Filename string; Data []byte; URL string; Citations bool; Provider ProviderData }
type ToolCall struct   { ID, Name string; Input json.RawMessage; ProviderExecuted bool; Provider ProviderData }
type ToolResult struct { CallID, Name string; Output ToolOutput; IsError, ProviderExecuted bool; Provider ProviderData }
type ToolApprovalRequest struct  { ApprovalID, CallID, Reason string; InputHash [32]byte }
type ToolApprovalResponse struct { ApprovalID, CallID string; Approved bool; Reason string }
type Custom struct     { Kind string; Data json.RawMessage; Provider ProviderData }

type Citation struct { Kind, URL, Title, CitedText string; DocIndex *int; Start, End int; Provider ProviderData }
type ToolOutput struct { Text string; JSON json.RawMessage; Parts Parts } // exactly one set
type ProviderData map[string]json.RawMessage
func (d ProviderData) With(key string, v any) ProviderData // copy
func ProviderValue[T any](d ProviderData, key string) (T, bool)
```
- `File.Citations` marks a document as citable (Anthropic native; others via adapter strategy).
- Adapter rules (contract-tested): never reorder/merge parts; keep signature-carrying empty parts; keep item ids and call ids; preserve unknown fields in `Provider`; drop model-bound reasoning the target model rejects.
- Approval parts never reach providers; the core resolves them.
- Cross-provider: reasoning without the target's key is dropped; foreign provider-executed calls/results become Text summaries; call ids mapped to the target format; calls without results get synthetic error results.

## 4. Call shape
```go
res, err := ai.GenerateText(ctx, model, ai.TextRequest{
    Instructions: ai.Texts("You are a support agent."),
    Messages:     []ai.Message{ai.User("Where is my order?")},
})
```
- Model is a required argument; options in a request struct shared by Generate/Stream. `new(expr)` for optional values where 0 is meaningful (`*float64`, `*int`).
- No Prompt field. Helpers: `ai.User`, `ai.UserParts`, `ai.Assistant`, `ai.Texts`.
- `Instructions []Text` so system blocks can carry cache markers (`anthropic.CachedText(s)`); `anthropic.Options{CacheControl}` for top-level automatic caching (not on Bedrock). No system role in messages; per-step instruction changes via `PrepareStep`.
- Requests are never mutated by the library; `req.Clone()` and `req.WithMessages(msgs)` (returns a clone) for reuse as a stored "agent"; `StepRequest` holds cloned slices.
- Provider options: `Provider: ai.Providers(anthropic.Options{...}, openai.Options{...})`, where each options type implements `ai.ProviderOptions{ ProviderName() string }`.
- Models: `p := openai.New(opts...)`; `p.Model(id)` returns the `*openai.LanguageModel`, and `p.EmbeddingModel(id)`, `p.RerankModel`, `p.ImageModel`, `p.VideoModel`, `p.SpeechModel`, `p.TranscriptionModel`, `p.RealtimeModel` return the concrete `EmbeddingModel`, `RerankModel`, `ImageModel`, `VideoModel`, `SpeechModel`, `TranscriptionModel`, `RealtimeModel` types of that package (OpenAI Chat Completions is `ChatModel`). `openai.Model(id)` uses a default provider configured from env; every other kind has a package-level `NewEmbeddingModel(id)`, `NewImageModel(id)`, ... (a func named `EmbeddingModel` would collide with the type).
- Provider option structs are `Options` (language model), `EmbeddingOptions`, `RerankOptions`, `ImageOptions`, `VideoOptions`, `SpeechOptions`, `TranscriptionOptions`, `RealtimeOptions`; the key they carry equals their `ProviderName()`. File references are `FileID(id)` plus `FileIDOf(ai.File) (string, bool)` in openai, anthropic and google. Token sources are `type TokenSource func(ctx) (string, error)` in google and openaicompat. Bedrock uses `WithAPIKey` and `WithBaseURL`; `anthropic.CacheTTL` and `bedrock.CacheTTL` are typed.
```go
type TextRequest struct {
    Instructions       []Text
    Messages           []Message
    Tools              []Tool
    ToolChoice         ToolChoice // zero = auto; ai.ToolRequired(), ai.ToolNone(), ai.ToolNamed(name)
    MaxToolConcurrency int        // 0 = 8
    Output             Output     // ai.ObjectOutput[T](), ai.ChoiceOutput(...), ai.JSONOutput()
    MaxSteps           int        // 0 = 20; always enforced
    StopWhen           []StopCondition // extra conditions, OR'd
    PrepareStep        func(ctx context.Context, s StepState, step *StepRequest) error
    OnStepEnd          func(ctx context.Context, s Step) error // error aborts
    MaxOutputTokens    int
    Temperature        *float64
    TopP               *float64
    TopK               int
    PresencePenalty    *float64
    FrequencyPenalty   *float64
    Seed               *int
    StopSequences      []string
    Reasoning          ReasoningEffort // ai.ReasoningNone ... ai.ReasoningMax; "" = provider default
    MaxRetries         *int            // default 2
    Timeouts           Timeouts
    Headers            http.Header
    Provider           ProviderData
    Telemetry          Telemetry // nil = off
}
```

## 5. Streaming
```go
s := ai.StreamText(ctx, model, req) // request sent on first consumption
for p := range s.Parts() {
    switch p := p.(type) {
    case ai.TextDelta: fmt.Print(p.Text)
    case ai.ToolCall:  log.Println(p.Name)
    }
}
res, err := s.Result()
```
- Why not `iter.Seq2[Part, error]`: a one-variable range compiles and drops the error (verified); Go has no convention yet (golang/go#71901, open). `Result()` returns the error with the aggregate.
- `s.Text()` text deltas only; `s.Err()` terminal error.
- Single-use enforced (`ai.ErrStreamConsumed`, `ai.ErrStreamActive`); not for concurrent use.
- `Result()` always returns completed steps (`Messages`, `Steps`) together with the error; the failed step's partial output is excluded. Callers append and call again to continue.
- `ToolCall`, `ToolResult`, `ToolApprovalRequest`, `File` implement both `Part` and `StreamPart`; `Citation` implements `StreamPart` only. Stream-only: `StreamStart`, `StepStart`, `TextStart/TextDelta/TextEnd`, `ReasoningStart/Delta/End`, `ToolInputStart/Delta/End`, `ToolFailed`, `StepEnd`, `StreamFinish`. Only start/finish carry the `Stream` prefix because bare `Start`/`Finish` say nothing; the rest are self-describing.

## 6. Tools, loop, approval
```go
type Tool struct {
    Name, Description string
    InputSchema       json.RawMessage
    Strict            bool
    Execute           func(ctx context.Context, input json.RawMessage) (ToolOutput, error) // nil = client-side
    Approval          func(ctx context.Context, in ApprovalInput) Approval                   // nil = none
    Untrusted         bool // output may carry injected instructions (default true for MCP)
    Timeout           time.Duration
    Provider          ProviderData
}
func NewTool[In, Out any](name, desc string, fn func(context.Context, In) (Out, error)) Tool
func NewClientTool[In any](name, desc string) Tool
func (t Tool) WithApproval(f func(context.Context, ApprovalInput) Approval) Tool
func (t Tool) WithTimeout(d time.Duration) Tool

type ApprovalInput struct { Call ToolCall; Tainted bool } // Tainted: an Untrusted tool output is in context
type Decision int // DecisionAsk (zero, fail-closed), DecisionAllow, DecisionDeny
type Approval struct { Decision Decision; Reason string }
func Allow() Approval; func Deny(reason string) Approval; func AskUser(reason string) Approval
func AlwaysAsk(context.Context, ApprovalInput) Approval
```
- Schema from `In` via `google/jsonschema-go` (pinned, internal; `jsonschema:"..."` = descriptions, like the MCP Go SDK). Named types implementing `JSONSchema() json.RawMessage` override their schema everywhere (wired via `ForOptions.TypeSchemas`). `Out` must exist; `ToolOutput` passes through, anything else is JSON.
- Tool names must be unique (`ai.ErrDuplicateTool`).
- Loop: continues while the model calls tools or the provider pauses (`ai.FinishPause`, e.g. Anthropic `pause_turn`; the response is appended and the loop continues); `MaxSteps` always applies; `StopWhen` adds `ai.HasToolCall`, `ai.MaxTotalTokens`, custom `func(steps []Step) bool`; checked after the step's tools run. Cap hit → `ai.FinishStepLimit`; stopped for approval → `ai.FinishApprovalPending`.
- `Untrusted` defaults to false for `NewTool` (the SDK cannot know); tools that fetch outside content should set it. MCP tools default to true.
- Tool failures: only errors created with `ai.ToolErrorf`/wrapping `*ai.ToolError` show their message to the model; other errors and panics become "tool failed" (no secrets to model or client).
- Approval binding: the SDK generates a random 128-bit `ApprovalID` and an `InputHash`; on resume it executes the stored call input only, matches `ApprovalID` + `CallID`, accepts each approval once, re-runs `Tool.Approval` and schema validation (policy `Deny` beats client approval).
- No Agent type; a stored `TextRequest` (+ `Clone`) is the agent.
```go
type TextResult struct {
    Text         string
    Content      Parts
    Messages     []Message // new messages of this call (completed steps)
    Steps        []Step
    Usage        Usage
    FinishReason FinishReason // string type: FinishStop, FinishLength, FinishToolCalls, FinishContentFilter, FinishRefusal, FinishStepLimit, FinishApprovalPending, FinishError, FinishOther, FinishPause
    FinishRaw    string
    Warnings     []Warning
    Provider     ProviderData
}
type Usage struct {
    Input, Output         int // Input includes cache read/write; Output includes reasoning
    CacheRead, CacheWrite int
    Reasoning             int // 0 also means "not reported"
    Raw                   json.RawMessage
}
```

## 7. Structured output
```go
p, res, err := ai.GenerateObject[Person](ctx, model, req)
label, res, err := ai.GenerateChoice(ctx, model, req, Positive, Negative) // [T ~string]; pass typed constants
raw, res, err := ai.GenerateJSON(ctx, model, req)

s := ai.StreamObject[[]Item](ctx, model, req)
for item := range ai.Elements(s) { ... } // element type must be spelled []E
if err := s.Err(); err != nil { ... }

req.Output = ai.ObjectOutput[Person]()   // or ai.ChoiceOutput(Positive, Negative), ai.JSONOutput()
p, err := ai.DecodeObject[Person](res)   // ai.DecodeChoice[T](res)
```
- Same loop as GenerateText. Per-provider schema transform; dropped keywords → Warning. Output validated against the original schema → `*ai.OutputError{Text, Err}` on failure. No native schema mode → forced tool call. Arrays wrapped where required. `Partials()` is best effort (zero = not yet).

## 8. Errors, retries, timeouts
- `*ai.APIError{Provider, StatusCode, Code, Message, RequestID, RetryAfter, Body, ShouldRetry *bool, ContextTooLong bool}`; `Body` is never in `Error()`, warnings, or default telemetry. `Retryable()`; `Is` for `ai.ErrRateLimit`, `ai.ErrQuota`, `ai.ErrAuth`, `ai.ErrContextTooLong`.
- `ShouldRetry` carries an explicit retry instruction and wins over the status-based default: `internal/transport` sets it from an `x-should-retry: true|false` response header (sent by OpenAI and Anthropic; the Anthropic and OpenAI Go SDKs honor it, their error pages do not document it). Bedrock sets it for `ModelErrorException`/`ModelStreamErrorException` (HTTP 424) from the documented `originalStatusCode`: retry only when the upstream status was 408/409/429/5xx. AWS documents no retry guidance for 424, so without `originalStatusCode` it is not retried.
- `ContextTooLong` is set by a provider classifier (`transport.Config.Classify`, e.g. Bedrock "Input is too long"); `Is(ErrContextTooLong)` also keeps generic checks (code `context_length_exceeded`, 413, "maximum context length", "prompt is too long").
- `*ai.RetryError` (`Unwrap() []error`), `*ai.OutputError`, `*ai.ToolError`, `*ai.NoSuchToolError`, `*ai.NoSuchModelError`, `*ai.JobError{Kind, ID, Err}` (failed or abandoned async job; the id survives ctx), `ai.ErrDuplicateTool`, `ai.ErrInvalidApproval`, `ai.ErrUnsupported`, `ai.ErrStreamConsumed`, `ai.ErrStreamActive`.
- Retries: default 2, backoff + jitter, `retry-after-ms`/`retry-after` capped at 60s; 408/409/429 (not quota)/5xx/529 unless `ShouldRetry` says otherwise; only before the first streamed byte.
- Timeouts: total = ctx; `Timeouts{Step, FirstChunk, Chunk, Tool}`; `Tool.Timeout`. No `http.Client.Timeout` for streams.

## 9. Middleware, limits, fallback, registry, telemetry
- `ai.WrapModel(m, ai.Middleware{...})`, `ai.WrapEmbeddingModel`, `ai.WrapImageModel`. Wrappers implement `Unwrap()`; `ai.As[T any](model any) (T, bool)` finds capabilities.
- `As` searches depth-first and returns the first match (Fallback children in order).
- Built-ins: `ai.ExtractReasoning(tag)`, `ai.DefaultSettings(req)`.
- Rate limiting at the provider: `openai.New(openai.WithLimiter(l))`, `l` is `ai.Limiter{ Wait(context.Context) error }` (`x/time/rate.Limiter` fits; one limiter can be shared across providers). Applies to every model kind of that provider instance.
- `ai.Fallback(models...)`: next model on retryable/category errors before any output; `Unwrap() []LanguageModel`; batch needs a single concrete provider.
- `ai.Registry{Providers, Aliases}`; `reg.LanguageModel("anthropic:claude-opus-5-5")`.
- Telemetry per request (`TextRequest.Telemetry`, media requests, `realtime.Config.Telemetry`); no package-level global (Google Go best practices: "Libraries should not force their clients to use APIs that rely on global state").
```go
type Telemetry interface {
    StartStep(ctx context.Context, info StepInfo) context.Context
    FinishStep(ctx context.Context, info StepInfo, step Step, err error)
    StartTool(ctx context.Context, call ToolCall) context.Context
    FinishTool(ctx context.Context, call ToolCall, result ToolResult, took time.Duration, err error)
}
type StepInfo struct { Provider, ModelID string; Number int; Request *StepRequest } // Request nil for media calls
```
- `StepInfo.Request` and `FinishTool`'s `ToolResult` are shared with the loop, read-only and free to pass; the implementation decides whether to copy or record them. The step and tool contexts flow to the model call and the tool. Media calls report one step.
- `aiotel` module: OTel API, GenAI conventions (Development status), span in tool ctx, metadata only by default. `CaptureContent` adds `gen_ai.system_instructions`, `gen_ai.input.messages` (roles user/assistant/tool; parts `text`, `reasoning`, `tool_call`, `tool_call_response`, `uri`/`file` without bytes), `gen_ai.output.messages`, `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` (successful calls only), all as JSON strings; error messages are recorded only with it. `Telemetry.InvokeAgent(ctx, name)` opens an `invoke_agent` parent span.
- The library never logs.

## 10. Other model kinds
```go
r, err := ai.Embed(ctx, emb, ai.EmbedRequest{Texts: docs, InputType: ai.InputDocument, Dimensions: 1024, Concurrency: 4})
q, err := ai.Embed(ctx, emb, ai.EmbedRequest{Texts: []string{query}, InputType: ai.InputQuery})
// r.Embeddings [][]float32; Texts or Inputs []Parts (multimodal), exactly one set; auto-chunked
rr, err := ai.Rerank(ctx, rk, ai.RerankRequest{Query: q, Documents: texts, TopN: 8}) // rr.Ranking []Ranked{Index, Score}
img, err := ai.GenerateImage(ctx, m, ai.ImageRequest{Prompt, N, Size, AspectRatio, Seed, Files, Mask})
vid, err := ai.GenerateVideo(ctx, m, ai.VideoRequest{Prompt, Duration, AspectRatio, Seed, Files, PollInterval}) // core owns polling; ctx bounds it; *ai.JobError carries the job id
sp, err := ai.GenerateSpeech(ctx, m, ai.SpeechRequest{Text, Voice, Format})
ss := ai.StreamSpeech(ctx, m, ai.SpeechStreamRequest{Text: s.Text()}) // ss.Audio() iter.Seq[[]byte]; ss.Err()
tr, err := ai.Transcribe(ctx, m, ai.TranscribeRequest{Audio: r, MediaType})  // io.Reader
ts := ai.StreamTranscribe(ctx, m, ai.TranscribeStreamRequest{Audio: mic})    // ts.Segments(), ts.Err()
```
Video: `ai.VideoModel` is `StartVideo(ctx, *VideoCall) (*VideoJob, error)` + `PollVideo(ctx, *VideoJob) (*VideoPoll, error)`; providers never loop. `VideoJob{ID, Provider, Header}`; `VideoPoll{Done, Progress, Videos, Failure, Interval, Warnings, Header}` is "running" while `Done` and `Failure` are unset. The core retries Start and each Poll (`MaxRetries`), waits `PollInterval` (0 = 5s, grows by half to 30s, or the provider's `Interval`), and wraps failures in `*ai.JobError`.
Realtime (package `ai/realtime`):
```go
sess, err := realtime.Connect(ctx, model, realtime.Config{Instructions, Tools, Voice, InputAudio, OutputAudio, TurnDetection, Resume /* handle from a Resumable event */, Provider})
sess.SendAudio(ctx, pcm); sess.SendText(ctx, s); sess.Interrupt(ctx); sess.Approve(ctx, approvalID, ok)
sess.SendToolResult(ctx, callID, out, isError) // answers a ToolCall of a tool declared without Execute
for ev := range sess.Events() {} // iter.Seq[Event], rangeable once // realtime.AudioDelta, TextDelta, Transcript, ToolCall, ToolResult, ApprovalRequest, Interrupted, ToolCanceled, TurnEnd, GoAway, Resumable
sess.Err(); sess.Close()
secret, err := realtime.MintClientSecret(ctx, model, cfg, realtime.MintOptions{TTL: time.Minute}) // realtime.ClientSecret{Value, ExpiresAt}; locks model/instructions/tools/voice where the provider supports it
```
Tools run server-side; `Approval` honoured via `ApprovalRequest` events + `Approve`; provider cancellation cancels the tool ctx; input buffers bounded.
Batch:
```go
job, err := ai.StartBatch(ctx, model, []ai.BatchItem{{ID: t.ID, Request: req}}) // single-step requests; loops/streams rejected
st, err := job.Status(ctx); err = job.Wait(ctx, time.Minute)
for r := range job.Results(ctx) {} // BatchResult{ID, Result *TextResult, Err}; job.Err()
job = ai.ResumeBatch(model, id).WithOutput(t.ID, ai.ChoiceOutput("a", "b")) // WithOutput supplies the Output that DecodeObject/DecodeChoice need; StartBatch remembers each item's Output
err = job.Cancel(ctx)
```
Files: `ai.UploadFile(ctx, model, ai.FileUpload{...}) (ai.File, error)`, Get/List/Delete; ids in `File.Provider`.
MCP (`aimcp` module): `aimcp.Tools(ctx, session, aimcp.Prefix(name))` — prefix required (default `<server>_`), local tools win, descriptions/schemas pinned at first list (`aimcp.Refresh` to update), size caps. Every MCP tool defaults to `Approval = ai.AlwaysAsk` and `Untrusted = true`; annotations can relax this only with `aimcp.TrustHints()` (spec: annotations are untrusted unless the server is trusted).

## 11. UI stream (useChat protocol v1)
```go
req, err := ui.ReadRequest(r, ui.Limits{}) // {MaxBodyBytes 1 MiB, MaxMessages, MaxParts, MaxDepth, FilePolicy}; client file URLs off
msgs, err := ui.Merge(stored, req, tools)  // server-owned history (stored loaded by the app with its own authz)

ui.Respond(w, r, func(ctx context.Context) *ai.TextStream {
    return ai.StreamText(ctx, model, base.WithMessages(msgs)) // use this ctx, never r.Context(), or resume breaks
}, ui.Options{
    OnFinish: func(ctx context.Context, f ui.Finish) error { return save(ctx, f.New) }, // ui.Finish{New, UI, Aborted, FinishReason, Err}; New = completed steps
    History:  msgs,                       // history the stream continues; if it does not end in a user message, the last assistant message is continued
    Key:       key,                        // signs assistant messages (stateless chats)
    ChatID:    chatID, Owner: userID,      // optional: bound into the signature
    Expose:    ui.Expose{Provider, Usage, UsageRaw, Warnings}, // all off by default
    NewID:     nil,                        // message ids; default random
    Resume:   &ui.Resume{Store: store, ChatID: chatID, Owner: userID},
})
http.Handle("GET /chat/{id}/stream", ui.ResumeHandler(store, ownerOf)) // ownerOf func(*http.Request) (string, error), required; mismatch = 204
```
- `ui.Respond` owns the ctx: request ctx normally, detached with limits when `Resume` is set (`MaxDuration` 10 min, per-owner `MaxConcurrent`, store TTL, `MaxBytes`). `ui.Stop(ctx, store, chatID, owner)` cancels through the store. Store errors: `ErrBusy`, `ErrTooManyStreams`, `ErrStreamTooLarge`, `ErrNoStream`.
- `ui.Merge` accepts only the new user message and approval responses matching stored requests; `Provider` data comes only from stored messages.
- Server-owned history → `ui.Merge`. Stateless apps → `ui.ToMessages`: `ui.Options.Key` (≥32 bytes, else `ui.ErrWeakKey`) signs every emitted assistant message; `ui.ToMessages(req, tools, ui.VerifyKey(newest, older...), ui.MaxAge(d), ui.Chat(chatID, userID))` accepts client history only when signatures verify and are younger than `MaxAge` (default `ui.DefaultMaxAge`, 10 minutes; 0 selects the default, a negative value disables the check), which bounds the replay window of a pending approval. When `Options.ChatID`/`Owner` were set at signing, the signature covers them, so history cannot be replayed into another chat or user; `ui.Chat` must then pass the same values. Without a key it accepts user text/file parts only and rejects approval responses.
- `ui.ToAI(msgs []ui.Message)` and `ui.FromMessages([]ai.Message)` convert trusted, application-stored messages without verification; never feed them client input (`Merge`/`ToMessages` only). `ui.Message.WithoutProvider()` strips replay data.
- `ui.Limits` zero values: 1 MiB body, 200 messages, 100 parts, depth 32. Request problems are `*ui.RequestError{Status, Err}` wrapping `ErrInvalidRequest`, `ErrRequestTooBig`, `ErrApproval`, `ErrFile`, `ErrBadSignature`.
- Client file URLs rejected unless `ui.Limits.FilePolicy` allows them; provider file ids never accepted from the wire.
- Errors: generic text by default (`OnError` customises), incl. tool errors and panics. `Provider`, `Usage.Raw`, `Warnings` never sent to the client unless enabled.
- Aborted streams persist completed steps only; orphan calls get synthetic error results.
- `ui.Stream(w, r, func(ctx, uw *ui.Writer) error {...}, opts)` for custom data (`uw.Data(name, v, ui.Transient(), ui.DataID(id))`, `uw.Merge(textStream)`); `ui.RespondText(w, r, factory, opts)`. `ui.StreamStore` + memory store; Redis in module `uiredis`.

## 12. Modules, providers, versions
- Module `github.com/minodevss/ai-go`, package `ai` (goimports derives `ai` from `ai-go`; future SDKs: `ai-ts`, `ai-rs`).
- Core deps: stdlib, `google/jsonschema-go` (pinned), `coder/websocket`. Separate modules: `aiotel`, `aimcp`, `uiredis`. Bedrock/Vertex in core with `bedrock.Credentials` / `google.TokenSource` + static/env/bearer paths.
- Spec interfaces in `ai`; extended only by new interfaces checked by assertion. Provider stream: `Stream(ctx, *Call) (*StreamResponse, error)` with `Parts iter.Seq2[StreamPart, error]` (core is the only consumer; adapters write one generator).
- Dedicated providers: OpenAI (Responses + Chat Completions), Anthropic, Google (AI Studio + Vertex option), Bedrock (Converse), Cohere, Ollama, Voyage, Jina, Deepgram, AssemblyAI, ElevenLabs, fal, Replicate, Black Forest Labs, Luma. `openaicompat` presets: Azure OpenAI, Mistral, DeepSeek, Perplexity, OpenRouter, xAI, Groq, Together, Fireworks, Cerebras, vLLM.
- v0 until: ≥2 providers per model kind pass the contract suite; nightly live suite green 4 weeks; an out-of-tree adapter compiles; `gorelease` clean for 3 minors; persisted JSON frozen with a version field. Then `v1.0.0-rc.N`.

## 13. Tests
- Offline wire identity: recorded response → decode → re-encode as history → byte-equal replayed items.
- Contract suite per adapter (order, ids, signatures, usage arithmetic, schema warnings, cross-provider rules, stream aggregate == generate).
- `aitest/record` RoundTripper with redaction and `-update`; scheduled drift check.
- Live suite: cheapest models, tiny limits, spend-capped keys, nightly on main.
- Fuzz: SSE, event-stream, WebSocket frames, partial JSON (linear time, depth cap), Message JSON.
- Leak/race/fault tests; `gorelease`; `GOWORK=off`.
- UI both ways with the npm `ai` reader and real useChat requests (approval, resume, signed history); matrix of npm majors.
- Security tests: forged/replayed approvals, changed-input re-emission, cross-owner resume/stop, client Provider data, file URL SSRF, oversized bodies.
- `aitest`: `aitest.NewModel(aitest.Turn(aitest.Text(..), aitest.ToolCall(..)), aitest.Err(..))` with delays, and `aitest.Fail(errs...)` to make a model's first calls fail; `Calls()`; `ReadUIStream`; deterministic IDs; zero-backoff retries.

## Decision log v2 → v3
| Change | Source |
|---|---|
| Approval binding (random id, input hash, stored input only, single use, policy re-check) | security review F1; Vercel signs name+id+input for the same reason |
| Signed history for stateless UI; `Merge` takes Provider data only from storage | F2 |
| Client file URLs off by default; no wire file ids | F3 (SSRF, cross-tenant ids) |
| Resume/stop bound to an owner; required owner func | F4 |
| Detached generation limits; stop through the store | F5 |
| `ai.ToolError` opt-in visibility; `APIError.Body` excluded; UI hides Provider/Raw/Warnings | F6 |
| Unique tool names; required MCP prefix; pinned MCP descriptions; MCP tools ask by default | F7, F8 (MCP spec: annotations untrusted) |
| Realtime approvals + locked client secrets | F9 |
| `Tool.Untrusted` + `ApprovalInput.Tainted` | F10 |
| `MaxSteps` always enforced; tool concurrency default 8; request size limits | F11 + compile check #18 |
| Persist completed steps only on abort | F12 |
| `Result()` returns completed steps with the error | challenge round: only recovery path after removing mid-stream retry |
| `anthropic.Options{CacheControl}` + `anthropic.CachedText` | Anthropic docs: automatic caching exists but explicit system breakpoints still recommended; Bedrock lacks automatic |
| Limiter moved to provider option | compile: `ai.Limit` could not wrap non-language models |
| `As[T any](model any)` | compile #2 |
| `Elements` example keeps the stream and checks `Err` | compile #5 |
| `WithApproval`/`WithTimeout`, `Allow()/Deny()/AskUser()` helpers | compile #6 + vague names |
| `Parts` normalises pointers | compile #9 |
| Shared part types implement `StreamPart`; `Stream*` prefixes; realtime in `ai/realtime` | compile #10 |
| `FinishReason` string + `FinishRaw` | compile #11 |
| `ChoiceOutput`/`DecodeChoice`/`JSONOutput` | compile #13 |
| `Fallback` exposes children; batch needs a concrete provider | compile #14 |
| `ui.Respond` takes a stream factory (owns ctx) | compile #15 |
| `Finish.New`, cloned `StepRequest`, `ai.Embed` single function, named ToolChoice/Reasoning constants, realtime `Resume`, `aitest.Turn`, `File.Citations` | compile #16, #17, #19–21 |
| Final compile pass: `ToolFailed` (name clash with `*ai.ToolError`), `WithMessages`, `ai.Providers`/`ProviderOptions`, model constructors, ToolChoice funcs, exported `Decision`, `FinishApprovalPending`, `EmbedRequest.Texts`, `ai.Limiter`, `As` order, missing signatures | final compile + consistency agent |
| Stood after challenge: provider Seq2, `Instructions []Text`, no system role, per-request telemetry, per-tool approval, coder/websocket, removals, 20-step default, int Usage | challenge round with primary sources |
