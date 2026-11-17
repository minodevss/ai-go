package aitest_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
	"github.com/minodevss/ai-go/realtime"
)

func ExampleNewModel() {
	model := aitest.NewModel(
		aitest.Turn(aitest.ToolCall("search", `{"q":"go"}`)),
		aitest.Turn(aitest.Reasoning("combine the hits"), aitest.Text("Go is a language.")),
	)
	search := ai.NewTool("search", "Search", func(context.Context, struct {
		Q string `json:"q"`
	},
	) (string, error) {
		return "hit", nil
	})

	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{
		Messages: []ai.Message{ai.User("What is Go?")},
		Tools:    []ai.Tool{search},
	})
	fmt.Println(res.Text, len(model.Calls()))
	// Output:
	// Go is a language. 2
}

func ExampleNewEmbeddingModel() {
	model := aitest.NewEmbeddingModel(aitest.MaxBatch(2))
	res, _ := ai.Embed(context.Background(), model, ai.EmbedRequest{Texts: []string{"a", "bb", "ccc"}})
	fmt.Println(len(res.Embeddings), len(model.Calls()))
	// Output:
	// 3 2
}

func ExampleNewRerankModel() {
	model := aitest.NewRerankModel()
	res, _ := ai.Rerank(context.Background(), model, ai.RerankRequest{Query: "go", Documents: []string{"rust", "go"}})
	fmt.Println(res.Ranking[0].Index)
	// Output:
	// 1
}

func ExampleTurn() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("hi"), aitest.Finish(ai.FinishLength), aitest.Usage(3, 1)))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	fmt.Println(res.FinishReason, res.Usage.Input)
	// Output:
	// length 3
}

func ExampleNewBatchModel() {
	model := aitest.NewBatchModel(
		aitest.Turn(aitest.Text("first")), aitest.Turn(aitest.Text("second")),
		aitest.States(ai.BatchRunning, ai.BatchCompleted),
	)
	ctx := context.Background()
	job, _ := ai.StartBatch(ctx, model, []ai.BatchItem{
		{ID: "a", Request: ai.TextRequest{Messages: []ai.Message{ai.User("1")}}},
		{ID: "b", Request: ai.TextRequest{Messages: []ai.Message{ai.User("2")}}},
	})
	_ = job.Wait(ctx, time.Millisecond)
	for r := range job.Results(ctx) {
		fmt.Println(r.ID, r.Result.Text)
	}
	fmt.Println(len(model.Submitted()))
	// Output:
	// b second
	// a first
	// 2
}

func ExampleFailItems() {
	model := aitest.NewBatchModel(
		aitest.Turn(aitest.Text("ok")),
		aitest.FailItems("b"),
	)
	ctx := context.Background()
	job, _ := ai.StartBatch(ctx, model, []ai.BatchItem{
		{ID: "a", Request: ai.TextRequest{Messages: []ai.Message{ai.User("1")}}},
		{ID: "b", Request: ai.TextRequest{Messages: []ai.Message{ai.User("2")}}},
	})
	_ = job.Wait(ctx, time.Millisecond)
	failed := map[string]bool{}
	for r := range job.Results(ctx) {
		failed[r.ID] = r.Err != nil
	}
	fmt.Println(failed["a"], failed["b"])
	// Output:
	// false true
}

func ExampleBatchModel_Canceled() {
	model := aitest.NewBatchModel()
	ctx := context.Background()
	job, _ := ai.StartBatch(ctx, model, []ai.BatchItem{{ID: "a", Request: ai.TextRequest{Messages: []ai.Message{ai.User("1")}}}})
	_ = job.Cancel(ctx)
	fmt.Println(model.Canceled())
	// Output:
	// true
}

func ExampleNewImageModel() {
	model := aitest.NewImageModel(aitest.Fail(errors.New("rate limited")))
	ctx := context.Background()
	_, err := ai.GenerateImage(ctx, model, ai.ImageRequest{Prompt: "a cat", MaxRetries: new(0)})
	fmt.Println(err != nil)
	res, _ := ai.GenerateImage(ctx, model, ai.ImageRequest{Prompt: "a cat", N: 2})
	fmt.Println(len(res.Images), res.Images[0].MediaType, len(model.Calls()))
	// Output:
	// true
	// 2 image/png 2
}

func ExampleNewVideoModel() {
	clip := ai.File{MediaType: "video/mp4", Data: []byte("mp4")}
	model := aitest.NewVideoModel(
		aitest.VideoProgress(40),
		aitest.VideoPollHint(aitest.VideoProgress(80), time.Millisecond),
		aitest.VideoDone(clip),
	)
	res, _ := ai.GenerateVideo(context.Background(), model, ai.VideoRequest{Prompt: "waves", PollInterval: time.Millisecond})
	fmt.Println(res.ID, len(res.Videos), len(model.Started()), len(model.PollTimes()))
	// Output:
	// video_1 1 1 3
}

func ExampleVideoFailed() {
	model := aitest.NewVideoModel(aitest.VideoFailed(errors.New("content policy")))
	_, err := ai.GenerateVideo(context.Background(), model, ai.VideoRequest{Prompt: "x", PollInterval: time.Millisecond})
	fmt.Println(err != nil)
	// Output:
	// true
}

func ExampleVideoPollError() {
	model := aitest.NewVideoModel(aitest.VideoPollError(errors.New("network down")), aitest.VideoDone())
	_, err := ai.GenerateVideo(context.Background(), model, ai.VideoRequest{Prompt: "x", PollInterval: time.Millisecond, MaxRetries: new(0)})
	fmt.Println(err != nil)
	// Output:
	// true
}

func ExampleNewSpeechModel() {
	model := aitest.NewSpeechModel()
	ctx := context.Background()
	res, _ := ai.GenerateSpeech(ctx, model, ai.SpeechRequest{Text: "hello"})
	fmt.Println(string(res.Audio.Data))

	stream := ai.StreamSpeech(ctx, model, ai.SpeechStreamRequest{Text: slices.Values([]string{"a", "b"})})
	for chunk := range stream.Audio() {
		fmt.Print(string(chunk))
	}
	fmt.Println()
	fmt.Println(model.StreamedText(), len(model.Calls()))
	// Output:
	// hello
	// ab
	// [a b] 1
}

func ExampleNewTranscriptionModel() {
	model := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "hello"}, ai.Segment{Text: "world"}))
	ctx := context.Background()
	res, _ := ai.Transcribe(ctx, model, ai.TranscribeRequest{Audio: strings.NewReader("pcm")})
	fmt.Println(res.Text)

	stream := ai.StreamTranscribe(ctx, model, ai.TranscribeStreamRequest{Audio: strings.NewReader("pcm")})
	for s := range stream.Segments() {
		fmt.Println(s.Text)
	}
	fmt.Println(len(model.Audio()))
	// Output:
	// hello world
	// hello
	// world
	// 2
}

func ExampleNewRealtimeModel() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, err := realtime.Connect(ctx, model, realtime.Config{Instructions: "be brief"})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer sess.Close()

	_ = sess.SendText(ctx, "hi")
	go func() { _ = model.Emit(ctx, realtime.TextDelta{Text: "hello"}) }()
	for ev := range sess.Events() {
		if d, ok := ev.(realtime.TextDelta); ok {
			fmt.Println(d.Text)
			break
		}
	}
	fmt.Println(model.Sent()[0].Kind, model.Sent()[0].Text, model.Config().Instructions)
	// Output:
	// hello
	// text hi be brief
}

func ExampleRealtimeModel_FailStream() {
	model := aitest.NewRealtimeModel()
	ctx := context.Background()
	sess, _ := realtime.Connect(ctx, model, realtime.Config{})
	defer sess.Close()

	go func() { _ = model.FailStream(ctx, errors.New("connection lost")) }()
	for range sess.Events() {
	}
	fmt.Println(sess.Err() != nil)
	// Output:
	// true
}

func ExampleRealtimeModel_MintRequests() {
	model := aitest.NewRealtimeModel()
	secret, _ := realtime.MintClientSecret(context.Background(), model, realtime.Config{}, realtime.MintOptions{TTL: time.Minute})
	fmt.Println(secret.Value, len(model.MintRequests()))
	// Output:
	// ek_1 1
}

func ExampleNewFiles() {
	files := aitest.NewFiles()
	ctx := context.Background()
	up, _ := ai.UploadFile(ctx, files, ai.FileUpload{Name: "a.txt", MediaType: "text/plain", Data: strings.NewReader("data")})
	id := aitest.FileID(up)
	data, ok := files.Data(id)
	fmt.Println(id, string(data), ok)
	// Output:
	// file_1 data true
}

func ExampleFileID() {
	files := aitest.NewFiles()
	f, _ := files.UploadFile(context.Background(), &ai.FileUploadCall{Name: "a.txt", Data: strings.NewReader("x")})
	fmt.Println(aitest.FileID(*f))
	// Output:
	// file_1
}

func ExampleText() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("hello")))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	fmt.Println(res.Text)
	// Output:
	// hello
}

func ExampleTextChunks() {
	model := aitest.NewModel(aitest.Turn(aitest.TextChunks("hel", "lo")))
	stream := ai.StreamText(context.Background(), model, ai.TextRequest{})
	for chunk := range stream.Text() {
		fmt.Print(chunk, "|")
	}
	fmt.Println()
	// Output:
	// hel|lo|
}

func ExampleReasoning() {
	model := aitest.NewModel(aitest.Turn(aitest.Reasoning("think"), aitest.Text("done")))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	reasoning, _ := res.Content[0].(ai.Reasoning)
	fmt.Println(reasoning.Text, res.Text)
	// Output:
	// think done
}

func ExampleToolCall() {
	model := aitest.NewModel(aitest.Turn(aitest.ToolCall("lookup", `{"q":"go"}`)))
	res, _ := model.Generate(context.Background(), &ai.Call{})
	call, _ := res.Content[0].(ai.ToolCall)
	fmt.Println(res.FinishReason, call.Name)
	// Output:
	// tool-calls lookup
}

func ExampleErr() {
	model := aitest.NewModel(aitest.Turn(aitest.Err(errors.New("overloaded"))))
	_, err := ai.GenerateText(context.Background(), model, ai.TextRequest{MaxRetries: new(0)})
	fmt.Println(err != nil)
	// Output:
	// true
}

func ExampleDelay() {
	model := aitest.NewModel(aitest.Turn(aitest.Delay(time.Hour), aitest.Text("late")))
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	_, err := ai.GenerateText(ctx, model, ai.TextRequest{})
	fmt.Println(err != nil)
	// Output:
	// true
}

func ExamplePart() {
	model := aitest.NewModel(aitest.Turn(aitest.Part(ai.Text{Text: "from a part"})))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	fmt.Println(res.Text)
	// Output:
	// from a part
}

func ExampleStreamPart() {
	model := aitest.NewModel(aitest.Turn(aitest.StreamPart(ai.TextDelta{ID: "t", Text: "raw"})))
	stream := ai.StreamText(context.Background(), model, ai.TextRequest{})
	for p := range stream.Parts() {
		if d, ok := p.(ai.TextDelta); ok {
			fmt.Println(d.Text)
		}
	}
	// Output:
	// raw
}

func ExampleFinish() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("cut"), aitest.Finish(ai.FinishLength)))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	fmt.Println(res.FinishReason)
	// Output:
	// length
}

func ExampleUsage() {
	model := aitest.NewModel(aitest.Turn(aitest.Text("x"), aitest.Usage(7, 2)))
	res, _ := ai.GenerateText(context.Background(), model, ai.TextRequest{})
	fmt.Println(res.Usage.Input, res.Usage.Output)
	// Output:
	// 7 2
}

func ExampleMaxBatch() {
	model := aitest.NewEmbeddingModel(aitest.MaxBatch(1))
	_, _ = ai.Embed(context.Background(), model, ai.EmbedRequest{Texts: []string{"a", "b"}})
	fmt.Println(len(model.Calls()))
	// Output:
	// 2
}

func ExampleEmbedFunc() {
	model := aitest.NewEmbeddingModel(aitest.EmbedFunc(func(text string) []float32 { return []float32{1, float32(len(text))} }))
	res, _ := ai.Embed(context.Background(), model, ai.EmbedRequest{Texts: []string{"abc"}})
	fmt.Println(res.Embeddings[0])
	// Output:
	// [1 3]
}

func ExampleFail() {
	model := aitest.NewEmbeddingModel(aitest.Fail(errors.New("throttled")))
	ctx := context.Background()
	_, err := ai.Embed(ctx, model, ai.EmbedRequest{Texts: []string{"a"}})
	fmt.Println(err)
	res, _ := ai.Embed(ctx, model, ai.EmbedRequest{Texts: []string{"a"}})
	fmt.Println(len(res.Embeddings), len(model.Calls()))
	// Output:
	// throttled
	// 1 2
}
