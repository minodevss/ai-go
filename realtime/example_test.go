package realtime_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/realtime"
)

type doorInput struct {
	Name string `json:"name" jsonschema:"door name"`
}

func ExampleConnect() {
	model := aitest.NewRealtimeModel()
	open := ai.NewTool("open_door", "Opens a door", func(_ context.Context, in doorInput) (string, error) {
		return in.Name + " is open", nil
	})
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{Instructions: "You run the building.", Tools: []ai.Tool{open}})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	go func() {
		_ = model.Emit(ctx, realtime.TextDelta{Text: "Opening."})
		_ = model.Emit(ctx, realtime.ToolCall{ID: "c1", Name: "open_door", Input: json.RawMessage(`{"name":"front"}`)})
	}()
	for ev := range sess.Events() {
		switch e := ev.(type) {
		case realtime.TextDelta:
			fmt.Println("text:", e.Text)
		case realtime.ToolResult:
			fmt.Println("tool:", string(e.Output.JSON))
			return
		}
	}
	// Output:
	// text: Opening.
	// tool: "front is open"
}

func ExampleSession_Approve() {
	model := aitest.NewRealtimeModel()
	pay := ai.NewTool("pay", "Pays an invoice", func(_ context.Context, in doorInput) (string, error) {
		return "paid " + in.Name, nil
	}).WithApproval(ai.AlwaysAsk)
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{Tools: []ai.Tool{pay}})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	go func() {
		_ = model.Emit(ctx, realtime.ToolCall{ID: "c1", Name: "pay", Input: json.RawMessage(`{"name":"invoice-7"}`)})
	}()
	for ev := range sess.Events() {
		switch e := ev.(type) {
		case realtime.ApprovalRequest:
			fmt.Println("approval needed for", e.Name, string(e.Input))
			if err := sess.Approve(ctx, e.ApprovalID, true); err != nil {
				fmt.Println(err)
				return
			}
		case realtime.ToolResult:
			fmt.Println("result:", string(e.Output.JSON))
			return
		}
	}
	// Output:
	// approval needed for pay {"name":"invoice-7"}
	// result: "paid invoice-7"
}

func ExampleSession_SendAudio() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{MaxInputBytes: 4})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()
	fmt.Println(sess.SendAudio(ctx, []byte{1, 2, 3, 4}))
	fmt.Println(sess.SendAudio(ctx, []byte{1, 2, 3, 4, 5}) != nil)
	// Output:
	// <nil>
	// true
}

func ExampleMintClientSecret() {
	model := aitest.NewRealtimeModel()
	secret, err := realtime.MintClientSecret(context.Background(), model, realtime.Config{Instructions: "Greet the visitor.", Voice: "marin"}, realtime.MintOptions{TTL: 90 * time.Second})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(secret.Value, secret.Locked, model.MintRequests()[0].TTL)
	// Output: ek_1 true 1m30s
}

func ExampleSession_SendText() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()
	fmt.Println(sess.SendText(ctx, "Hello"))
	fmt.Println(model.Sent()[0].Kind, model.Sent()[0].Text)
	// Output:
	// <nil>
	// text Hello
}

func ExampleSession_Interrupt() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()
	fmt.Println(sess.Interrupt(ctx))
	fmt.Println(model.Sent()[0].Kind)
	// Output:
	// <nil>
	// interrupt
}

func ExampleSession_SendToolResult() {
	model := aitest.NewRealtimeModel()
	lookup := ai.NewClientTool[doorInput]("lookup", "Looks up a door on the client")
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{Tools: []ai.Tool{lookup}})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	go func() {
		_ = model.Emit(ctx, realtime.ToolCall{ID: "c1", Name: "lookup", Input: json.RawMessage(`{"name":"front"}`)})
	}()
	for ev := range sess.Events() {
		if call, ok := ev.(realtime.ToolCall); ok {
			fmt.Println(sess.SendToolResult(ctx, call.ID, ai.ToolOutput{Text: "unlocked"}, false))
			break
		}
	}
	sent := model.Sent()[0]
	fmt.Println(sent.Kind, sent.CallID, sent.Output.Text)
	// Output:
	// <nil>
	// tool-result c1 unlocked
}

func ExampleSession_Events() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	go func() {
		_ = model.Emit(ctx, realtime.TextDelta{Text: "Hi"})
	}()
	for ev := range sess.Events() {
		if delta, ok := ev.(realtime.TextDelta); ok {
			fmt.Println(delta.Text)
			break
		}
	}
	for range sess.Events() {
	}
	fmt.Println(sess.Err())
	// Output:
	// Hi
	// realtime: events already consumed
}

func ExampleSession_Err() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	go func() {
		_ = model.FailStream(ctx, errors.New("connection lost"))
	}()
	for range sess.Events() {
	}
	fmt.Println(sess.Err())
	// Output:
	// connection lost
}

func ExampleSession_Close() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(sess.Close(), sess.Close())
	fmt.Println(errors.Is(sess.SendText(ctx, "late"), realtime.ErrClosed), sess.Err())
	// Output:
	// <nil> <nil>
	// true <nil>
}
