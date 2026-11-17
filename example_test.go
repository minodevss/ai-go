package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func ExampleGenerateText() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("Your order ships tomorrow.")))

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Instructions: ai.Texts("You are a support agent."),
		Messages:     []ai.Message{ai.User("Where is my order?")},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	fmt.Println(res.FinishReason)
	// Output:
	// Your order ships tomorrow.
	// stop
}

func ExampleGenerateText_tools() {
	type orderIn struct {
		ID string `json:"id" jsonschema:"order id"`
	}
	lookup := ai.NewTool("lookup_order", "Look up an order", func(_ context.Context, in orderIn) (string, error) {
		return "order " + in.ID + " ships tomorrow", nil
	})
	model := aitest.NewModel(
		aitest.Turn(aitest.ToolCall("lookup_order", `{"id":"42"}`)),
		aitest.Turn(aitest.Text("It ships tomorrow.")),
	)

	res, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("Where is order 42?")},
		Tools:    []ai.Tool{lookup},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Steps), "steps:", res.Text)
	// Output:
	// 2 steps: It ships tomorrow.
}

func ExampleGenerateText_stepLimit() {
	search := ai.NewTool("search", "Search", func(context.Context, struct{}) (string, error) { return "hit", nil })
	model := aitest.NewModel(
		aitest.Turn(aitest.ToolCall("search", `{}`)),
		aitest.Turn(aitest.ToolCall("search", `{}`)),
		aitest.Turn(aitest.ToolCall("search", `{}`)),
	)

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Tools: []ai.Tool{search}, MaxSteps: 2})
	fmt.Println(len(res.Steps), res.FinishReason)
	// Output:
	// 2 step-limit
}

func ExampleStreamText() {
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks("Hel", "lo, ", "world")))

	s := ai.StreamText(context.Background(), model, ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	for text := range s.Text() {
		fmt.Print(text)
	}
	fmt.Println()
	if err := s.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// Hello, world
}

func ExampleTextStream_Parts() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("done")))

	s := ai.StreamText(context.Background(), model, ai.TextRequest{})
	for p := range s.Parts() {
		switch p := p.(type) {
		case ai.TextDelta:
			fmt.Println("text:", p.Text)
		case ai.StreamFinish:
			fmt.Println("finish:", p.FinishReason)
		}
	}
	// Output:
	// text: done
	// finish: stop
}

func ExampleTextStream_Result() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("all at once"), aitest.Usage(5, 2)))

	res, err := ai.StreamText(context.Background(), model, ai.TextRequest{}).Result()
	fmt.Println(res.Text, res.Usage.Input, res.Usage.Output, err)
	// Output:
	// all at once 5 2 <nil>
}

func ExampleTextStream_Err() {
	s := ai.StreamText(context.Background(), aitest.NewModel(aitest.Turn(aitest.Text("x"))), ai.TextRequest{})
	for range s.Parts() {
	}
	for range s.Parts() {
	}
	fmt.Println(errors.Is(s.Err(), ai.ErrStreamConsumed))
	// Output:
	// true
}

func ExampleTextStream_Text() {
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks("a", "b")))
	for chunk := range ai.StreamText(context.Background(), model, ai.TextRequest{}).Text() {
		fmt.Print(chunk, "|")
	}
	fmt.Println()
	// Output:
	// a|b|
}

func ExampleNewTool() {
	type weatherIn struct {
		City string `json:"city" jsonschema:"city name"`
	}
	tool := ai.NewTool("weather", "Current weather", func(_ context.Context, in weatherIn) (string, error) {
		return "sunny in " + in.City, nil
	})
	out, err := tool.Execute(context.Background(), json.RawMessage(`{"city":"Daegu"}`))
	fmt.Println(string(out.JSON), err)
	// Output:
	// "sunny in Daegu" <nil>
}

func ExampleNewClientTool() {
	type answerIn struct {
		Text string `json:"text"`
	}
	final := ai.NewClientTool[answerIn]("final_answer", "Submit the answer")
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("final_answer", `{"text":"42"}`)))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Tools: []ai.Tool{final}})
	for _, p := range res.Content {
		if call, ok := p.(ai.ToolCall); ok {
			fmt.Println(call.Name, string(call.Input))
		}
	}
	// Output:
	// final_answer {"text":"42"}
}

func ExampleTool_WithApproval() {
	remove := ai.NewTool("remove", "Remove a record", func(context.Context, struct{}) (string, error) { return "removed", nil }).
		WithApproval(ai.AlwaysAsk)
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("remove", `{}`)))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Tools: []ai.Tool{remove}})
	fmt.Println(res.FinishReason)
	// Output:
	// approval-pending
}

func ExampleTool_WithTimeout() {
	tool := ai.NewTool("slow", "d", func(context.Context, struct{}) (int, error) { return 1, nil }).WithTimeout(0)
	fmt.Println(tool.Timeout)
	// Output:
	// 0s
}

func ExampleToolErrorf() {
	lookup := ai.NewTool("lookup", "Look up a city", func(context.Context, struct{}) (string, error) {
		return "", ai.ToolErrorf("unknown city %q", "Oz")
	})
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("lookup", `{}`)), aitest.Turn(aitest.Text("sorry")))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Tools: []ai.Tool{lookup}})
	for _, r := range resultsOf(res.Messages) {
		fmt.Println(r.Output.Text)
	}
	// Output:
	// unknown city "Oz"
}

func ExampleAllow() {
	policy := func(_ context.Context, in ai.ApprovalInput) ai.Approval {
		if in.Tainted {
			return ai.AskUser("output of an untrusted tool is in context")
		}
		return ai.Allow()
	}
	fmt.Println(policy(context.Background(), ai.ApprovalInput{}).Decision == ai.DecisionAllow)
	fmt.Println(policy(context.Background(), ai.ApprovalInput{Tainted: true}).Decision == ai.DecisionAsk)
	// Output:
	// true
	// true
}

func ExampleDeny() {
	policy := func(_ context.Context, in ai.ApprovalInput) ai.Approval {
		if in.Call.Name == "drop_table" {
			return ai.Deny("never allowed")
		}
		return ai.Allow()
	}
	a := policy(context.Background(), ai.ApprovalInput{Call: ai.ToolCall{Name: "drop_table"}})
	fmt.Println(a.Decision == ai.DecisionDeny, a.Reason)
	// Output:
	// true never allowed
}

func ExampleHasToolCall() {
	stop := ai.HasToolCall("final_answer")
	steps := []ai.Step{{ToolCalls: []ai.ToolCall{{Name: "search"}}}, {ToolCalls: []ai.ToolCall{{Name: "final_answer"}}}}
	fmt.Println(stop(steps[:1]), stop(steps))
	// Output:
	// false true
}

func ExampleMaxTotalTokens() {
	stop := ai.MaxTotalTokens(100)
	fmt.Println(stop([]ai.Step{{Usage: ai.Usage{Input: 60, Output: 30}}}), stop([]ai.Step{{Usage: ai.Usage{Input: 60, Output: 30}}, {Usage: ai.Usage{Output: 10}}}))
	// Output:
	// false true
}

func ExampleTextRequest_Clone() {
	agent := ai.TextRequest{Instructions: ai.Texts("You are a researcher."), Temperature: new(0.2)}
	turn := agent.Clone()
	turn.Messages = []ai.Message{ai.User("hi")}
	fmt.Println(len(agent.Messages), len(turn.Messages))
	// Output:
	// 0 1
}

func ExampleTextRequest_WithMessages() {
	agent := ai.TextRequest{Instructions: ai.Texts("You are a researcher.")}
	turn := agent.WithMessages([]ai.Message{ai.User("q1")})
	fmt.Println(len(agent.Messages), len(turn.Messages), turn.Instructions[0].Text)
	// Output:
	// 0 1 You are a researcher.
}

func ExampleGenerateObject() {
	type person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"Alice","age":30}`)))

	p, _, err := ai.GenerateObject[person](context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("Extract: Alice is 30.")},
	})
	fmt.Println(p.Name, p.Age, err)
	// Output:
	// Alice 30 <nil>
}

func ExampleGenerateObject_invalid() {
	type person struct {
		Name string `json:"name"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":42}`)))

	_, _, err := ai.GenerateObject[person](context.Background(), model, ai.TextRequest{})
	outErr, ok := errors.AsType[*ai.OutputError](err)
	fmt.Println(ok, outErr.Text)
	// Output:
	// true {"name":42}
}

func ExampleStreamObject() {
	type person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"name":"Al`, `ice","age":3`, `0}`)))

	s := ai.StreamObject[person](context.Background(), model, ai.TextRequest{})
	for p := range s.Partials() {
		fmt.Printf("%+v\n", p)
	}
	fmt.Println(s.Err())
	// Output:
	// {Name:Al Age:0}
	// {Name:Alice Age:3}
	// {Name:Alice Age:30}
	// <nil>
}

func ExampleElements() {
	type item struct {
		Title string `json:"title"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"result":[{"title":"a"},`, `{"title":"b"}]}`)))

	s := ai.StreamObject[[]item](context.Background(), model, ai.TextRequest{})
	for it := range ai.Elements(s) {
		fmt.Println(it.Title)
	}
	if err := s.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// a
	// b
}

func ExampleGenerateChoice() {
	type sentiment string
	const (
		positive sentiment = "positive"
		negative sentiment = "negative"
	)
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"negative"}`)))

	label, _, err := ai.GenerateChoice(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("The delivery was late.")},
	}, positive, negative)
	fmt.Println(label, err)
	// Output:
	// negative <nil>
}

func ExampleGenerateJSON() {
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"any":["json"]}`)))
	raw, _, err := ai.GenerateJSON(context.Background(), model, ai.TextRequest{})
	fmt.Println(string(raw), err)
	// Output:
	// {"any":["json"]} <nil>
}

func ExampleObjectOutput() {
	type verdict struct {
		OK bool `json:"ok"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"ok":true}`)))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Output: ai.ObjectOutput[verdict]()})
	v, err := ai.DecodeObject[verdict](res)
	fmt.Println(v.OK, err)
	// Output:
	// true <nil>
}

func ExampleChoiceOutput() {
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"yes"}`)))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Output: ai.ChoiceOutput("yes", "no")})
	answer, err := ai.DecodeChoice[string](res)
	fmt.Println(answer, err)
	// Output:
	// yes <nil>
}

func ExampleJSONOutput() {
	model := aitest.NewModel(aitest.Turn(aitest.Text(`[1,2,3]`)))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Output: ai.JSONOutput()})
	fmt.Println(res.Text, model.Calls()[0].Output.Name)
	// Output:
	// [1,2,3] response
}

func ExampleDecodeObject() {
	res := &ai.TextResult{Text: `{"n":7}`}
	v, err := ai.DecodeObject[struct {
		N int `json:"n"`
	}](res)
	fmt.Println(v.N, err)
	// Output:
	// 7 <nil>
}

func ExampleDecodeChoice() {
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"result":"b"}`)))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Output: ai.ChoiceOutput("a", "b")})
	choice, err := ai.DecodeChoice[string](res)
	fmt.Println(choice, err)
	// Output:
	// b <nil>
}

func ExampleEmbed() {
	model := aitest.NewEmbeddingModel(aitest.MaxBatch(2))

	res, err := ai.Embed(context.Background(), model, ai.EmbedRequest{
		Texts:       []string{"a", "bb", "ccc"},
		InputType:   ai.InputDocument,
		Concurrency: 2,
	})
	fmt.Println(len(res.Embeddings), len(model.Calls()), err)
	// Output:
	// 3 2 <nil>
}

func ExampleRerank() {
	res, err := ai.Rerank(context.Background(), aitest.NewRerankModel(), ai.RerankRequest{
		Query:     "go generics",
		Documents: []string{"baking bread", "go generics tutorial", "go concurrency"},
		TopN:      2,
	})
	for _, r := range res.Ranking {
		fmt.Println(r.Index, r.Score)
	}
	fmt.Println(err)
	// Output:
	// 1 1
	// 2 0.5
	// <nil>
}

func ExampleWrapModel() {
	logging := ai.Middleware{TransformRequest: func(_ context.Context, c *ai.Call) error {
		c.Instructions = append(c.Instructions, ai.Text{Text: "Answer briefly."})
		return nil
	}}
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))

	_, _ = ai.GenerateText(context.Background(), ai.WrapModel(inner, logging), ai.TextRequest{})
	fmt.Println(inner.Calls()[0].Instructions[0].Text)
	// Output:
	// Answer briefly.
}

func ExampleAs() {
	inner := aitest.NewModel()
	wrapped := ai.WrapModel(ai.Fallback(inner), ai.Middleware{})

	found, ok := ai.As[*aitest.Model](wrapped)
	fmt.Println(ok, found == inner)
	// Output:
	// true true
}

func ExampleExtractReasoning() {
	model := ai.WrapModel(
		aitest.NewModel(aitest.Turn(aitest.Text("<think>2+2 is 4</think>The answer is 4."))),
		ai.ExtractReasoning("think"),
	)
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	for _, p := range res.Content {
		switch p := p.(type) {
		case ai.Reasoning:
			fmt.Println("reasoning:", p.Text)
		case ai.Text:
			fmt.Println("text:", p.Text)
		}
	}
	// Output:
	// reasoning: 2+2 is 4
	// text: The answer is 4.
}

func ExampleDefaultSettings() {
	inner := aitest.NewModel(aitest.Turn(aitest.Text("ok")))
	model := ai.WrapModel(inner, ai.DefaultSettings(ai.TextRequest{Temperature: new(0.2), MaxOutputTokens: 256}))

	_, _ = ai.GenerateText(context.Background(), model, ai.TextRequest{})
	call := inner.Calls()[0]
	fmt.Println(*call.Temperature, call.MaxOutputTokens)
	// Output:
	// 0.2 256
}

func ExampleFallback() {
	primary := aitest.NewModel(aitest.Turn(aitest.Err(&ai.APIError{StatusCode: http.StatusServiceUnavailable})))
	backup := aitest.NewModel(aitest.Turn(aitest.Text("from the backup")))

	res, err := ai.GenerateText(context.Background(), ai.Fallback(primary, backup), ai.TextRequest{MaxRetries: new(0)})
	fmt.Println(res.Text, err)
	// Output:
	// from the backup <nil>
}

type exampleProvider struct{ model ai.LanguageModel }

func (p exampleProvider) LanguageModel(string) (ai.LanguageModel, error) { return p.model, nil }

func ExampleRegistry_LanguageModel() {
	reg := ai.Registry{
		Providers: map[string]ai.Provider{"anthropic": exampleProvider{model: aitest.NewModel()}},
		Aliases:   map[string]string{"smart": "anthropic:claude-opus-5-5"},
	}
	m, err := reg.LanguageModel("smart")
	fmt.Println(m.ModelID(), err)

	_, err = reg.LanguageModel("openai:gpt")
	_, isMissing := errors.AsType[*ai.NoSuchModelError](err)
	fmt.Println(isMissing)
	// Output:
	// scripted <nil>
	// true
}

func ExampleProviderData_WithRaw() {
	base := ai.ProviderData{}
	derived := base.WithRaw("anthropic", json.RawMessage(`{"container":"c1"}`))
	v, _ := ai.ProviderValue[map[string]string](derived, "anthropic")
	fmt.Println(len(base), v["container"])
	// Output:
	// 0 c1
}

func ExampleProviderData_WithValue() {
	derived, err := ai.ProviderData{}.WithValue("anthropic", map[string]string{"container": "c1"})
	v, _ := ai.ProviderValue[map[string]string](derived, "anthropic")
	fmt.Println(err, v["container"])
	// Output:
	// <nil> c1
}

func ExampleProviderValue() {
	data := ai.ProviderData{}.WithRaw("openai", json.RawMessage(`{"itemId":"rs_1"}`))
	v, ok := ai.ProviderValue[struct {
		ItemID string `json:"itemId"`
	}](data, "openai")
	fmt.Println(v.ItemID, ok)
	// Output:
	// rs_1 true
}

func ExampleProviders() {
	data := ai.Providers(exampleOptions{})
	_, ok := ai.ProviderValue[exampleOptions](data, "example")
	fmt.Println(ok)
	// Output:
	// true
}

type exampleOptions struct {
	Tier string `json:"tier"`
}

func (exampleOptions) ProviderName() string { return "example" }

func ExampleUser() {
	fmt.Println(ai.User("hi").Role, len(ai.User("hi").Content))
	// Output:
	// user 1
}

func ExampleUserParts() {
	msg := ai.UserParts(ai.Text{Text: "what is in this image?"}, &ai.File{MediaType: "image/png", Data: []byte{0x89}})
	_, isValue := msg.Content[1].(ai.File)
	fmt.Println(len(msg.Content), isValue)
	// Output:
	// 2 true
}

func ExampleAssistant() {
	fmt.Println(ai.Assistant("hello").Role)
	// Output:
	// assistant
}

func ExampleTexts() {
	fmt.Println(len(ai.Texts("rules", "style guide")))
	// Output:
	// 2
}

func ExampleParts() {
	raw := `[{"type":"text","text":"hi"},{"type":"tool-call","id":"c1","name":"search","input":{"q":"go"}}]`
	var parts ai.Parts
	if err := json.Unmarshal([]byte(raw), &parts); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%T %T\n", parts[0], parts[1])
	// Output:
	// ai.Text ai.ToolCall
}

func ExampleToolRequired() {
	fmt.Println(ai.ToolRequired().Mode(), ai.ToolNone().Mode(), ai.ToolNamed("search").Name(), ai.ToolChoice{}.Mode())
	// Output:
	// required none search auto
}

func ExampleAPIError() {
	err := &ai.APIError{Provider: "openai", StatusCode: http.StatusTooManyRequests, Code: "rate_limit_exceeded", Message: "slow down", Body: "{raw}"}
	fmt.Println(err)
	fmt.Println(errors.Is(err, ai.ErrRateLimit), err.Retryable())
	// Output:
	// ai: openai: status 429 (rate_limit_exceeded): slow down
	// true true
}

type exampleModelProvider struct {
	exampleProvider
	embedding ai.EmbeddingModel
	rerank    ai.RerankModel
}

func (p exampleModelProvider) EmbeddingModel(string) (ai.EmbeddingModel, error) {
	return p.embedding, nil
}

func (p exampleModelProvider) RerankModel(string) (ai.RerankModel, error) { return p.rerank, nil }

func exampleRegistry() ai.Registry {
	return ai.Registry{
		Providers: map[string]ai.Provider{"example": exampleModelProvider{
			exampleProvider: exampleProvider{model: aitest.NewModel()},
			embedding:       aitest.NewEmbeddingModel(),
			rerank:          aitest.NewRerankModel(),
		}},
		Aliases: map[string]string{"embedder": "example:embed-1"},
	}
}

func ExampleRegistry_EmbeddingModel() {
	reg := exampleRegistry()
	m, err := reg.EmbeddingModel("embedder")
	fmt.Println(m.ModelID(), err)

	_, err = reg.EmbeddingModel("missing:model")
	_, isMissing := errors.AsType[*ai.NoSuchModelError](err)
	fmt.Println(isMissing)
	// Output:
	// scripted-embedding <nil>
	// true
}

func ExampleRegistry_RerankModel() {
	reg := exampleRegistry()
	m, err := reg.RerankModel("example:rerank-1")
	fmt.Println(m.ModelID(), err)

	_, err = reg.RerankModel("missing:model")
	_, isMissing := errors.AsType[*ai.NoSuchModelError](err)
	fmt.Println(isMissing)
	// Output:
	// scripted-rerank <nil>
	// true
}

func ExampleToolNone() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("no tools needed")))
	_, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages:   []ai.Message{ai.User("Just answer.")},
		Tools:      []ai.Tool{ai.NewClientTool[struct{}]("search", "Search the web")},
		ToolChoice: ai.ToolNone(),
	})
	fmt.Println(model.Calls()[0].ToolChoice.Mode(), err)
	// Output:
	// none <nil>
}

func ExampleToolNamed() {
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("search", `{}`)))
	_, err := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages:   []ai.Message{ai.User("Find it.")},
		Tools:      []ai.Tool{ai.NewClientTool[struct{}]("search", "Search the web")},
		ToolChoice: ai.ToolNamed("search"),
	})
	choice := model.Calls()[0].ToolChoice
	fmt.Println(choice.Mode(), choice.Name(), err)
	// Output:
	// named search <nil>
}

func ExampleAskUser() {
	policy := func(_ context.Context, in ai.ApprovalInput) ai.Approval {
		return ai.AskUser("deleting " + in.Call.Name + " cannot be undone")
	}
	remove := ai.NewTool("remove", "Remove a record", func(context.Context, struct{}) (string, error) { return "removed", nil }).
		WithApproval(policy)
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("remove", `{}`)))

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{Tools: []ai.Tool{remove}})
	fmt.Println(res.FinishReason)
	// Output:
	// approval-pending
}

func ExampleAlwaysAsk() {
	approval := ai.AlwaysAsk(context.Background(), ai.ApprovalInput{})
	fmt.Println(approval.Decision == ai.DecisionAsk)
	// Output:
	// true
}

func ExampleAPIError_Retryable() {
	rateLimited := &ai.APIError{Provider: "openai", StatusCode: http.StatusTooManyRequests}
	badRequest := &ai.APIError{Provider: "openai", StatusCode: http.StatusBadRequest}
	fmt.Println(rateLimited.Retryable(), badRequest.Retryable())
	// Output:
	// true false
}

func ExampleObjectStream_Partials() {
	type person struct {
		Name string `json:"name"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks(`{"name":"Al`, `ice"}`)))

	s := ai.StreamObject[person](context.Background(), model, ai.TextRequest{})
	for p := range s.Partials() {
		fmt.Println(p.Name)
	}
	// Output:
	// Al
	// Alice
}

func ExampleObjectStream_Result() {
	type person struct {
		Name string `json:"name"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"Alice"}`)))

	s := ai.StreamObject[person](context.Background(), model, ai.TextRequest{})
	v, res, err := s.Result()
	fmt.Println(v.Name, res.Text, err)
	// Output:
	// Alice {"name":"Alice"} <nil>
}

func ExampleObjectStream_Parts() {
	type person struct {
		Name string `json:"name"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`{"name":"Alice"}`)))

	s := ai.StreamObject[person](context.Background(), model, ai.TextRequest{})
	for part := range s.Parts() {
		if delta, ok := part.(ai.TextDelta); ok {
			fmt.Println(delta.Text)
		}
	}
	// Output:
	// {"name":"Alice"}
}

func ExampleObjectStream_Err() {
	type person struct {
		Name string `json:"name"`
	}
	model := aitest.NewModel(aitest.Turn(aitest.Text(`not json`)))

	s := ai.StreamObject[person](context.Background(), model, ai.TextRequest{})
	for range s.Partials() {
	}
	_, isOutput := errors.AsType[*ai.OutputError](s.Err())
	fmt.Println(isOutput)
	// Output:
	// true
}

func ExampleToolChoice_Mode() {
	fmt.Println(ai.ToolChoice{}.Mode(), ai.ToolRequired().Mode())
	// Output:
	// auto required
}

func ExampleToolChoice_Name() {
	fmt.Println(ai.ToolNamed("search").Name())
	// Output:
	// search
}
