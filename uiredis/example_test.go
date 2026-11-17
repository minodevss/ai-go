package uiredis_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/minodevss/ai-go/ui"
	"github.com/minodevss/ai-go/uiredis"
)

func newClient() (*redis.Client, func()) {
	server, err := miniredis.Run()
	if err != nil {
		panic(err)
	}
	return redis.NewClient(&redis.Options{Addr: server.Addr()}), server.Close
}

func ExampleNew() {
	client, closeServer := newClient()
	defer closeServer()
	ctx := context.Background()
	store := uiredis.New(client)
	w, err := store.Begin(ctx, ui.StreamSpec{ChatID: "chat", Owner: "user", Stop: func() {}})
	if err != nil {
		panic(err)
	}
	if err := w.Append(ctx, []byte("frame")); err != nil {
		panic(err)
	}
	r, err := store.Attach(ctx, "chat", "user")
	if err != nil {
		panic(err)
	}
	if err := w.Close(ctx); err != nil {
		panic(err)
	}
	for {
		frame, err := r.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		fmt.Println(string(frame))
	}
	// Output: frame
}

func ExampleWithTTL() {
	client, closeServer := newClient()
	defer closeServer()
	store := uiredis.New(client, uiredis.WithTTL(time.Minute))
	_, err := store.Begin(context.Background(), ui.StreamSpec{ChatID: "chat", Owner: "user", Stop: func() {}})
	fmt.Println(err)
	// Output: <nil>
}

func ExampleWithGrace() {
	client, closeServer := newClient()
	defer closeServer()
	store := uiredis.New(client, uiredis.WithGrace(10*time.Second))
	w, err := store.Begin(context.Background(), ui.StreamSpec{ChatID: "chat", Owner: "user", Stop: func() {}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(w.Close(context.Background()))
	// Output: <nil>
}

func ExampleWithPrefix() {
	client, closeServer := newClient()
	defer closeServer()
	ctx := context.Background()
	store := uiredis.New(client, uiredis.WithPrefix("app"))
	if _, err := store.Begin(ctx, ui.StreamSpec{ChatID: "chat", Owner: "user", Stop: func() {}}); err != nil {
		fmt.Println(err)
		return
	}
	_, err := uiredis.New(client).Attach(ctx, "chat", "user")
	fmt.Println(errors.Is(err, ui.ErrNoStream))
	// Output: true
}

func ExampleStore_Begin() {
	client, closeServer := newClient()
	defer closeServer()
	ctx := context.Background()
	store := uiredis.New(client)
	spec := ui.StreamSpec{ChatID: "chat", Owner: "user", MaxConcurrent: 1, Stop: func() {}}
	if _, err := store.Begin(ctx, spec); err != nil {
		fmt.Println(err)
		return
	}
	spec.ChatID = "other"
	_, err := store.Begin(ctx, spec)
	fmt.Println(errors.Is(err, ui.ErrTooManyStreams))
	// Output: true
}

func ExampleStore_Attach() {
	client, closeServer := newClient()
	defer closeServer()
	_, err := uiredis.New(client).Attach(context.Background(), "chat", "user")
	fmt.Println(errors.Is(err, ui.ErrNoStream))
	// Output: true
}

func ExampleStore_Stop() {
	client, closeServer := newClient()
	defer closeServer()
	ctx := context.Background()
	store := uiredis.New(client)
	stopped := make(chan struct{})
	if _, err := store.Begin(ctx, ui.StreamSpec{ChatID: "chat", Owner: "user", Stop: func() { close(stopped) }}); err != nil {
		fmt.Println(err)
		return
	}
	if err := store.Stop(ctx, "chat", "user"); err != nil {
		fmt.Println(err)
		return
	}
	<-stopped
	fmt.Println("stopped")
	// Output: stopped
}
