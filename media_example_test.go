package ai_test

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

func ExampleGenerateImage() {
	res, err := ai.GenerateImage(context.Background(), aitest.NewImageModel(), ai.ImageRequest{Prompt: "a lighthouse at dusk", N: 2, Size: "1024x1024"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(res.Images), res.Images[0].MediaType)
	// Output: 2 image/png
}

func ExampleGenerateVideo() {
	clip := ai.File{MediaType: "video/mp4", Data: []byte("mp4")}
	model := aitest.NewVideoModel(aitest.VideoProgress(30), aitest.VideoDone(clip))
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	res, err := ai.GenerateVideo(ctx, model, ai.VideoRequest{Prompt: "waves on a beach", Duration: 8 * time.Second, PollInterval: time.Millisecond})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.ID, res.Videos[0].MediaType)
	// Output: video_1 video/mp4
}

func ExampleGenerateSpeech() {
	res, err := ai.GenerateSpeech(context.Background(), aitest.NewSpeechModel(), ai.SpeechRequest{Text: "Hello", Voice: "marin", Format: "mp3"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Audio.MediaType, len(res.Audio.Data))
	// Output: audio/mpeg 5
}

func ExampleStreamSpeech() {
	text := func(yield func(string) bool) {
		for _, piece := range []string{"Hello, ", "world."} {
			if !yield(piece) {
				return
			}
		}
	}
	s := ai.StreamSpeech(context.Background(), aitest.NewSpeechModel(), ai.SpeechStreamRequest{Text: text})
	total := 0
	for chunk := range s.Audio() {
		total += len(chunk)
	}
	if err := s.Err(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(total, s.MediaType())
	// Output: 13 audio/mpeg
}

func ExampleSpeechStream_Audio() {
	text := func(yield func(string) bool) { yield("one piece") }
	s := ai.StreamSpeech(context.Background(), aitest.NewSpeechModel(), ai.SpeechStreamRequest{Text: text})
	for range s.Audio() {
	}
	for range s.Audio() {
	}
	fmt.Println(s.Err())
	// Output: ai: stream already consumed
}

func ExampleTranscribe() {
	model := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "Hello", End: time.Second}, ai.Segment{Text: "world", Start: time.Second, End: 2 * time.Second}))
	res, err := ai.Transcribe(context.Background(), model, ai.TranscribeRequest{Audio: strings.NewReader("audio"), MediaType: "audio/wav"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text, len(res.Segments))
	// Output: Hello world 2
}

func ExampleStreamTranscribe() {
	model := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "Hel", Partial: true}, ai.Segment{Text: "Hello"}))
	s := ai.StreamTranscribe(context.Background(), model, ai.TranscribeStreamRequest{Audio: strings.NewReader("audio")})
	for seg := range s.Segments() {
		fmt.Println(seg.Text, seg.Partial)
	}
	if err := s.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// Hel true
	// Hello false
}

func ExampleStartBatch() {
	model := aitest.NewBatchModel(aitest.Turn(aitest.Text("blue")), aitest.Turn(aitest.Text("green")))
	items := []ai.BatchItem{
		{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Reply with blue.")}}},
		{ID: "q2", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Reply with green.")}}},
	}
	ctx := context.Background()
	job, err := ai.StartBatch(ctx, model, items)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := job.Wait(ctx, time.Second); err != nil {
		fmt.Println(err)
		return
	}
	answers := map[string]string{}
	for r := range job.Results(ctx) {
		answers[r.ID] = r.Result.Text
	}
	fmt.Println(job.Err(), answers["q1"], answers["q2"])
	// Output: <nil> blue green
}

func ExampleResumeBatch() {
	model := aitest.NewBatchModel()
	job := ai.ResumeBatch(model, "batch_42")
	st, err := job.Status(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(job.ID(), st.State, st.State.Done())
	// Output: batch_42 completed true
}

func ExampleBatchJob_Cancel() {
	model := aitest.NewBatchModel(aitest.States(ai.BatchRunning))
	job := ai.ResumeBatch(model, "batch_1")
	fmt.Println(job.Cancel(context.Background()), model.Canceled())
	// Output: <nil> true
}

func ExampleUploadFile() {
	store := aitest.NewFiles()
	f, err := ai.UploadFile(context.Background(), store, ai.FileUpload{Name: "notes.txt", MediaType: "text/plain", Data: strings.NewReader("hello")})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(f.Filename, aitest.FileID(f), len(f.Data))
	// Output: notes.txt file_1 0
}

func ExampleGetFile() {
	store := aitest.NewFiles()
	ctx := context.Background()
	up, err := ai.UploadFile(ctx, store, ai.FileUpload{Name: "a.txt", Data: strings.NewReader("a")})
	if err != nil {
		fmt.Println(err)
		return
	}
	f, err := ai.GetFile(ctx, store, aitest.FileID(up))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(f.Filename)
	// Output: a.txt
}

func ExampleListFiles() {
	store := aitest.NewFiles()
	ctx := context.Background()
	for _, name := range []string{"a", "b", "c"} {
		if _, err := ai.UploadFile(ctx, store, ai.FileUpload{Name: name, Data: strings.NewReader(name)}); err != nil {
			fmt.Println(err)
			return
		}
	}
	after := ""
	for {
		page, err := ai.ListFiles(ctx, store, ai.FileListRequest{Limit: 2, After: after})
		if err != nil {
			fmt.Println(err)
			return
		}
		for _, f := range page.Files {
			fmt.Println(f.Filename)
		}
		if after = page.Next; after == "" {
			return
		}
	}
	// Output:
	// a
	// b
	// c
}

func ExampleDeleteFile() {
	store := aitest.NewFiles()
	ctx := context.Background()
	up, err := ai.UploadFile(ctx, store, ai.FileUpload{Data: strings.NewReader("x")})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(ai.DeleteFile(ctx, store, aitest.FileID(up)))
	_, err = ai.GetFile(ctx, store, aitest.FileID(up))
	fmt.Println(err != nil)
	// Output:
	// <nil>
	// true
}

func ExampleBatchJob_ID() {
	model := aitest.NewBatchModel()
	job, err := ai.StartBatch(context.Background(), model, []ai.BatchItem{{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("hi")}}}})
	fmt.Println(job.ID() != "", err)
	// Output: true <nil>
}

func ExampleBatchJob_Status() {
	model := aitest.NewBatchModel(aitest.States(ai.BatchRunning, ai.BatchCompleted))
	job := ai.ResumeBatch(model, "batch_7")
	for range 2 {
		st, err := job.Status(context.Background())
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(st.State, st.State.Done())
	}
	// Output:
	// running false
	// completed true
}

func ExampleBatchJob_Wait() {
	model := aitest.NewBatchModel(aitest.States(ai.BatchRunning, ai.BatchCompleted))
	job := ai.ResumeBatch(model, "batch_7")
	err := job.Wait(context.Background(), time.Millisecond)
	fmt.Println(err)
	// Output: <nil>
}

func ExampleBatchJob_Results() {
	model := aitest.NewBatchModel(aitest.Turn(aitest.Text("blue")))
	items := []ai.BatchItem{{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Reply with blue.")}}}}
	ctx := context.Background()
	job, err := ai.StartBatch(ctx, model, items)
	if err != nil {
		fmt.Println(err)
		return
	}
	for r := range job.Results(ctx) {
		fmt.Println(r.ID, r.Result.Text, r.Err)
	}
	// Output: q1 blue <nil>
}

func ExampleBatchJob_WithOutput() {
	type answer struct {
		Color string `json:"color"`
	}
	model := aitest.NewBatchModel(aitest.Turn(aitest.Text(`{"color":"blue"}`)))
	items := []ai.BatchItem{{ID: "q1", Request: ai.TextRequest{Messages: []ai.Message{ai.User("Pick a color.")}}}}
	ctx := context.Background()
	job, err := ai.StartBatch(ctx, model, items)
	if err != nil {
		fmt.Println(err)
		return
	}
	job.WithOutput("q1", ai.ObjectOutput[answer]())
	for r := range job.Results(ctx) {
		v, err := ai.DecodeObject[answer](r.Result)
		fmt.Println(v.Color, err)
	}
	// Output: blue <nil>
}

func ExampleBatchJob_Err() {
	model := aitest.NewBatchModel()
	job := ai.ResumeBatch(model, "batch_7")
	for range job.Results(context.Background()) {
	}
	fmt.Println(job.Err())
	// Output: <nil>
}

func ExampleBatchState_Done() {
	fmt.Println(ai.BatchRunning.Done(), ai.BatchCompleted.Done())
	// Output: false true
}

func ExampleSpeechStream_MediaType() {
	text := func(yield func(string) bool) { yield("hello") }
	s := ai.StreamSpeech(context.Background(), aitest.NewSpeechModel(), ai.SpeechStreamRequest{Text: text})
	for range s.Audio() {
	}
	fmt.Println(s.MediaType())
	// Output: audio/mpeg
}

func ExampleSpeechStream_Err() {
	text := func(yield func(string) bool) { yield("hello") }
	s := ai.StreamSpeech(context.Background(), aitest.NewSpeechModel(), ai.SpeechStreamRequest{Text: text})
	for range s.Audio() {
	}
	fmt.Println(s.Err())
	// Output: <nil>
}

func ExampleTranscribeStream_Segments() {
	model := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "Hello"}))
	s := ai.StreamTranscribe(context.Background(), model, ai.TranscribeStreamRequest{Audio: strings.NewReader("audio")})
	for seg := range s.Segments() {
		fmt.Println(seg.Text)
	}
	// Output: Hello
}

func ExampleTranscribeStream_Err() {
	model := aitest.NewTranscriptionModel(aitest.Segments(ai.Segment{Text: "Hello"}))
	s := ai.StreamTranscribe(context.Background(), model, ai.TranscribeStreamRequest{Audio: strings.NewReader("audio")})
	for range s.Segments() {
	}
	fmt.Println(s.Err())
	// Output: <nil>
}
