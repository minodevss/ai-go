# Go style

Rules for all Go code in `github.com/minodevss/ai-go`. Where the Go sources disagree, this file picks one option, and that option is binding. Tooling enforces every rule it can. The rest are checked in review.

## Principles

- Code explains itself. Use names, types and structure, not comments.
- Use the standard library first. Every dependency needs a reason that holds up in review.
- Make the zero value useful, and make invalid states unrepresentable.
- Choose the obvious solution over the clever one. Use the shortest correct code, not the shortest code.

## Comments

Comments are banned. The only exceptions are the cases below.

| Allowed | Example |
|---|---|
| Directives the toolchain reads | `//go:build`, `//go:generate`, `//go:embed`, `//go:fix inline` |
| Example output checks | `// Output:` in `Example` functions |
| Deprecation notices | `// Deprecated: use X.` on a deprecated identifier |
| Doc comments on exported API that add what the signature cannot say | `// Parts can be ranged once; Result returns the error.` |
| Lint suppression with a reason | `//nolint:errcheck // body already fully read` |
| A why that code cannot express, with a source | `// Gemini 3 rejects parallel calls unless only the first carries the signature: <doc URL>` |

- **No comments that say what code does, and no section dividers.**
  - Do not keep commented-out code, TODOs or changelogs in code.
  - Rename, extract a function, or introduce a type instead.
- **Doc comments on exported identifiers follow Go convention, but only when they carry information.**
  - Write one when the name and signature do not tell the whole story: zero-value meaning, single use, concurrency, defaults, ownership, or what an error means.
  - Never restate the name (`// Text is a text part.`). An identifier that explains itself gets no comment.
  - A doc comment is a full sentence that starts with the identifier's name, and is one or two lines long.
  - A package comment is one line in the package's main file.
  - Every exported entry point also has a runnable `Example`. Examples are compiled and executed by `go test`, so they cannot rot.
- **Unexported code has no doc comments.**
- **A why-comment is one line, and it must cite an external fact.**
  - The fact is a provider doc, an issue or an RFC.
  - If the reason is internal, the code must say it instead, through a name or a type.

## Formatting and tooling

- `gofumpt` and `goimports` format all code. Import groups are the standard library, third-party packages, then `github.com/minodevss/ai-go`.
- There is no line-length limit. If a line is too long, refactor it. Do not break signatures, conditions, strings or URLs to fit a width.
- `golangci-lint` v2 with the configuration in the appendix must pass with zero issues. Every `//nolint` names its linter and gives a reason.
- Run `go fix ./...` until it is stable. CI runs `go fix -diff ./...`, and any diff fails the build.

## Naming

- **Packages:** short, lowercase, one word, and equal to the last path element.
  - Never use `util`, `common`, `helper`, `types` or `api`.
  - Providers are named after the provider: `openai`, `anthropic`.
- **No stutter.** Write `ai.Message`, `openai.Options`, `anthropic.New`. Never `ai.AIMessage` or `openai.OpenAIOptions`.
- **Initialisms keep one case:** `ID`, `URL`, `HTTP`, `API`, `JSON`, `MCP`, `apiKey`, `callID`.
- **Accessors have no `Get` prefix.** Getters are `Name()` and setters are `SetName()`.
- **Constructors:** `New` builds a package's main type. `NewX` builds any other type.
- **Errors:** sentinels are `ErrX`, and error types end in `Error` (`APIError`, `OutputError`).
- **Receivers:** one or two letters, the same in every method of a type, never `self` or `this`.
- **Length tracks scope.** Short names in small scopes, descriptive names at package level. No type names inside variable names: write `parts`, not `partSlice`.
- **Do not shadow predeclared identifiers** (`len`, `min`, `max`, `error`, `string`, `new`) or imported package names.
- **Tests and examples:** `TestType_Method`, `TestFunc`, `ExampleGenerateText`, `ExampleTool_WithApproval`.
  - A lowercase scenario suffix is allowed on tests: `TestType_scenario`, `TestFunc_scenario`.

## Packages and files

- One package owns one concept. Split a package only when the dependency arrow is one-way and both halves are useful alone.
- **Files are grouped by concept, not by one type per file.** For example: `message.go`, `stream.go`, `tool.go`, `loop.go`.
- **Declaration order within a file:**
  1. Types, consts and vars
  2. Constructors
  3. Methods, grouped by receiver
  4. Unexported helpers, in call order
- **`internal/` holds everything users must not import.** That includes the SSE parser, retry logic and schema transforms.
- **No `init` functions and no mutable package-level variables.**
  - Package-level values are sentinel errors and immutable tables only.
  - Providers are registered explicitly with `ai.Registry`, never through `init`.

## Types and values

- **Struct literals always use field names.** Write `var x T` for a zero value and `&T{}` for a pointer.
- **Optional values whose zero is meaningful are pointers set with `new(expr)`.** Values whose zero means "unset" stay plain.

```go
req := ai.TextRequest{
	Messages:        []ai.Message{ai.User("hi")},
	Temperature:     new(0.2),
	MaxOutputTokens: 1024,
}
```

- **Wire-facing enums are typed string constants.**

```go
type FinishReason string

const (
	FinishStop      FinishReason = "stop"
	FinishLength    FinishReason = "length"
	FinishToolCalls FinishReason = "tool-calls"
)
```

- **Integer enums start at `iota + 1`.** A zero that is a valid choice must be a deliberate default, like `DecisionAsk`.
- **Sealed interfaces use an unexported method.** Their implementations are values with value receivers.
- **Receivers are consistent per type.**
  - Use pointer receivers when the type mutates, holds a mutex, or is large. Use value receivers for small immutable values.
  - Never mix the two on one type.
- **Copy slices and maps at API boundaries.**
  - Copy anything you store that a caller still owns, and anything you return that you still own.
  - Request structs are never mutated by the library.
- **Use `time.Duration` and `time.Time` for time.** Never use an `int` for time.
- **Empty slices are `nil`.** Check them with `len(s) == 0`.

## Functions and APIs

- **`ctx context.Context` is the first parameter of any call that can block.**
  - Never store it in a struct or put it in an options struct.
  - Library code never calls `context.Background()`.
- **Required inputs are parameters. Optional inputs live in a request struct.**
  - New fields are added with a zero value that keeps the old behavior.
- **Constructors take variadic functional options.**

```go
type Option func(*options)

func WithAPIKey(key string) Option {
	return func(o *options) { o.apiKey = key }
}

func New(opts ...Option) *Provider {
	o := options{apiKey: os.Getenv("OPENAI_API_KEY")}
	for _, opt := range opts {
		opt(&o)
	}
	return &Provider{opts: o}
}
```

- **`error` is the last result.** Exported functions return the `error` interface, never a concrete error type.
- **Named results only when they document the meaning of results with the same type.**
  - No naked returns.
  - A deferred closure that sets a named error is the one place named results are required.
- **Handle errors first and return early.** No `else` after `return`. Keep the happy path at the left margin.
- **Boolean parameters that are unclear at the call site become a type or a field.**
- **Functions are synchronous.** Callers add concurrency. A function that starts a goroutine documents its stop path through its API, with `Close` or ctx.
- **Write `any`, not `interface{}`.**

## Errors

- **Error strings are lowercase, have no trailing punctuation, and are prefixed with the package name.**
  - Example: `"openai: decode stream: unexpected event"`.
  - Never use "failed to" or "error while".
- **Add only context the caller lacks.** Name the failed action, not the function.
- **Choose the verb by who should match the cause.**
  - Use `%w` when callers should match the cause: sentinels, typed errors, `ctx.Err()`.
  - Use `%v` for causes that are implementation details.
  - Wrapped errors are part of the API.
- **Match errors with `errors.Is` and `errors.AsType`.** Never compare error strings.

```go
if apiErr, ok := errors.AsType[*ai.APIError](err); ok && apiErr.Retryable() {
	return retry(ctx, apiErr.RetryAfter)
}
```

- **Handle each error once.** The library never logs. Errors, warnings and telemetry are its only outputs.
- **Never discard an error with `_`.**
  - The only exception is a call documented to always return nil.
  - Write that call with a why-comment that cites the source.
- **No panics on user input.**
  - Internal invariants may panic, with a message prefixed by the package name.
  - Recovered tool panics become tool errors.
- **Type assertions always use the comma-ok form.**

## Interfaces

- **The consumer defines the interface.**
  - Package `ai` defines `LanguageModel` and the other model interfaces.
  - Providers return concrete types.
- **Interfaces are small.**
  - Capabilities are added as new interfaces, checked with a type assertion, never as new methods on an existing interface.
- **Provider packages assert their contracts at compile time.**

```go
var _ ai.LanguageModel = (*Model)(nil)
```

- **No interfaces created only for mocking.** Tests use the `aitest` fakes or `httptest`.
- **No embedded types in exported structs.** Use a named field and delegate to it.

## Generics and iterators

- **Use generics only where the code is identical for every type argument,** or where they turn a runtime type assertion into a compile-time check: `NewTool[In, Out]`, `GenerateObject[T]`.
  - Use reflection for behavior that differs per type.
- **No exported generic methods.** They need Go 1.27, and the module minimum is 1.26.
- **User-facing streams are `iter.Seq[T]` with `Err()` and `Result()`.**
  - Ranging an `iter.Seq2[T, error]` with one variable silently drops the error.
  - `iter.Seq2[T, error]` is allowed only at the internal provider boundary, where the core is the only consumer.
- **Iterators stop as soon as `yield` returns false, and they release resources on every exit path.**

```go
func (s *Stream) Parts() iter.Seq[Part] {
	return func(yield func(Part) bool) {
		defer s.close()
		for s.next() {
			if !yield(s.part) {
				return
			}
		}
	}
}
```

- **Single-use iterators enforce single use.** A second consumer gets an error, never a silent empty loop.
- **Callers of `iter.Pull` always `defer stop()`.**

## Concurrency

- **Every goroutine has an owner and a stop path:** ctx cancellation, a closed channel, or `Close`.
- **Start groups with `wg.Go`.**
- **APIs expose iterators, not channels.**
  - Channel buffers inside the library are 0 or 1, unless a justified bound is required.
  - Parameters declare a channel's direction.
- **A mutex is a named field, `mu sync.Mutex`, never embedded.** Write `defer mu.Unlock()` right after `Lock`.
- **Use the typed atomics in `sync/atomic`:** `atomic.Bool`, `atomic.Int64`.
- **Tests run with `-race`.**

## JSON and HTTP

- **Use `encoding/json` v1 APIs, not v2.**
  - `encoding/json/v2` is Go 1.27 only.
  - v1 runs on the v2 engine in 1.27, so behavior matches.
- **Every serialized field has an explicit tag.**
  - Use `omitzero` for optional fields. Use `omitempty` only where an empty string or slice must be omitted.
- **Sum types serialize with a `"type"` discriminator.**
  - Decoders keep unknown fields as `json.RawMessage`. Nothing a provider sends is dropped.
- **Close every response body with `defer`.**
  - Stream lifetimes follow ctx. `http.Client.Timeout` is never set, because it cuts off long streams.
- **Use `http.NewRequestWithContext` for outbound requests.**
  - Build server routes with `ServeMux` method and wildcard patterns (`"GET /chat/{id}/stream"`).
  - Join hosts and ports with `net.JoinHostPort`.

## Dependencies and compatibility

- **`go.mod` declares `go 1.26.0`.**
  - The `stdversion` analyzer rejects symbols newer than that.
  - The floor rises only in minor releases, and only to the oldest Go release that is still supported.
- **Core dependencies:**
  - The standard library.
  - `github.com/google/jsonschema-go`, pinned and internal only. Its types never appear in the public API.
  - `github.com/coder/websocket`.
  - Any other dependency goes in its own module (`aiotel`, `aimcp`, `uiredis`).
- **Tools are not declared in `go.mod`.** There is no `tools.go` and no `tool` directive, because they would put lint dependencies into users' module graphs. CI installs pinned versions instead.
- **Every release runs `gorelease`.**
  - Before v1, breaking changes are allowed but listed in the release notes.

## Testing

- **Test with the standard `testing` package and `github.com/google/go-cmp/cmp`.**
  - go-cmp is already in the module graph through jsonschema-go.
  - No assertion libraries.
- **Test public API from `package x_test`.** Internal tests are only for unexported machinery.
- **Table tests use `tests` and `tt`, and every case is a named `t.Run` subtest.**
  - Expected values are `want…`. Actual values are `got…`.
  - The loop body does not branch on the case, except one error-expectation branch (`if tt.wantErr`).

```go
func TestParseFinishReason(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want ai.FinishReason
	}{
		{name: "anthropic end_turn", raw: "end_turn", want: ai.FinishStop},
		{name: "openai length", raw: "length", want: ai.FinishLength},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseFinishReason(tt.raw); got != tt.want {
				t.Errorf("parseFinishReason(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}
```

- **Failure messages follow two formats.**
  - Single values: `Func(args) = got, want want`.
  - Diffs: `Func(args) mismatch (-want +got):\n%s` with `cmp.Diff(want, got)`.
- **Choose `t.Error` or `t.Fatal` by whether the test can continue.**
  - Use `t.Error` to keep going.
  - Use `t.Fatal` only when continuing is meaningless.
  - Never call `t.Fatal` from a goroutine the test started.
- **Test helpers call `t.Helper()`.**
  - Use `t.Cleanup` for teardown and `t.Context()` for contexts.
- **Time-dependent code is tested inside `synctest.Test`. No real sleeps.**
- **Providers are tested against `httptest` servers that replay recorded fixtures from `testdata/`.**
  - A package-level `-update` flag re-records them.
- **Parsers and decoders have fuzz tests:** SSE, event-stream, WebSocket frames, partial JSON, `Message` JSON.
  - Their seed corpora are committed.
- **Benchmarks use `b.Loop()`.**

## CI gates

All of these must pass.

```sh
golangci-lint fmt --diff
go mod tidy -diff
go vet ./...
golangci-lint run
go fix -diff ./...
go test -race -shuffle=on ./...
govulncheck ./...
gorelease -base=latest
```

Workspaces are not committed. CI runs with `GOWORK=off`.

## Sources

- Effective Go: go.dev/doc/effective_go
- Go Code Review Comments: go.dev/wiki/CodeReviewComments
- Google Go Style Guide (guide, decisions, best practices): google.github.io/styleguide/go
- Uber Go Style Guide: github.com/uber-go/guide
- Go blog: package names; errors are values; Go 1.13 errors; context and structs; keeping modules compatible; when to use generics; range over function types; examples; gofix.
- Go release notes 1.21–1.27; go.dev/doc/comment; pkg.go.dev for iter, errors, testing/synctest, modernize; golangci-lint.run configuration docs.

## Where this file departs from the sources, and why

| Topic | Sources say | This file | Reason |
|---|---|---|---|
| Doc comments | go.dev/doc/comment: every exported name should have one | Only on exported names whose signature leaves something out, plus `Example` functions everywhere | Owner's no-comment rule, kept compatible with Go convention: no comment restates a name, and pkg.go.dev still explains every non-obvious behavior |
| Functional options | Uber: `Option` interface. Google: options struct or variadic | Variadic `func(*options)` for constructors, request structs for calls | Constructors are rarely configured, and calls carry many optional fields (Google's split). The func form is the smallest that keeps `options` unexported |
| Enum zero value | Uber: `iota + 1`. Google: no rule | `iota + 1` unless zero is a deliberate default | A forgotten field never silently becomes a valid choice |
| Naked returns | Allowed in short functions | Banned | One fewer style to read |
| Line length | Uber: soft 99. Google: none | None, refactor instead | gofumpt plus refactoring keeps lines readable without wrapping noise |
| Fallible iterators | No Go convention (golang/go#71901 open) | `Seq` + `Err`/`Result` for users, `Seq2` internal only | A one-variable range over `Seq2` drops errors silently |
| Import groups | Google: four groups. Uber: two | Three: std, third-party, this module | Local imports stay visibly separate (`goimports` `local-prefixes`) |

## Appendix: `.golangci.yml`

```yaml
version: "2"

run:
  go: "1.26"
  tests: true
  modules-download-mode: readonly

linters:
  default: none
  enable:
    - copyloopvar
    - errcheck
    - errname
    - errorlint
    - exhaustive
    - exptostd
    - govet
    - ineffassign
    - intrange
    - misspell
    - modernize
    - nilerr
    - noctx
    - nolintlint
    - perfsprint
    - predeclared
    - revive
    - staticcheck
    - testableexamples
    - thelper
    - unconvert
    - unparam
    - unused
    - usestdlibvars
    - usetesting
    - wastedassign
  settings:
    errcheck:
      check-type-assertions: true
    exhaustive:
      default-signifies-exhaustive: true
    govet:
      enable-all: true
      disable:
        - fieldalignment
        - shadow
    misspell:
      locale: US
    nolintlint:
      require-specific: true
      require-explanation: true
      allow-unused: false
    perfsprint:
      err-error: true
      strconcat: false
    revive:
      enable-default-rules: false
      rules:
        - name: blank-imports
        - name: context-as-argument
        - name: context-keys-type
        - name: dot-imports
        - name: empty-block
        - name: error-naming
        - name: error-return
        - name: error-strings
        - name: errorf
        - name: increment-decrement
        - name: indent-error-flow
        - name: range
        - name: receiver-naming
        - name: redefines-builtin-id
        - name: superfluous-else
        - name: time-naming
        - name: unexported-return
        - name: unreachable-code
        - name: unused-parameter
        - name: var-declaration
        - name: var-naming
    staticcheck:
      checks:
        - all
        - -ST1000
        - -ST1003
        - -ST1016
        - -ST1020
        - -ST1021
        - -ST1022
        - -ST1023
    unparam:
      check-exported: false
    usetesting:
      os-temp-dir: true
      context-background: true
      context-todo: true
  exclusions:
    generated: lax
    warn-unused: true
    presets:
      - std-error-handling
      - common-false-positives

formatters:
  enable:
    - gofumpt
    - goimports
  settings:
    gofumpt:
      module-path: github.com/minodevss/ai-go
    goimports:
      local-prefixes:
        - github.com/minodevss/ai-go
  exclusions:
    generated: lax

issues:
  max-issues-per-linter: 0
  max-same-issues: 0
```
