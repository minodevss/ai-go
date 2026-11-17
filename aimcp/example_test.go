package aimcp_test

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aimcp"
)

func exampleSession(ctx context.Context) *mcp.ClientSession {
	server := mcp.NewServer(&mcp.Implementation{Name: "files", Version: "1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "read", Description: "Reads a file."}, echo)
	mcp.AddTool(server, &mcp.Tool{Name: "write", Description: "Writes a file."}, echo)
	serverTransport, clientTransport := mcp.NewInMemoryTransports()
	if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
		panic(err)
	}
	session, err := mcp.NewClient(&mcp.Implementation{Name: "app", Version: "1.0.0"}, nil).Connect(ctx, clientTransport, nil)
	if err != nil {
		panic(err)
	}
	return session
}

func exampleNames(tools []ai.Tool, err error) {
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, t := range tools {
		fmt.Println(t.Name, t.Untrusted)
	}
}

func ExampleTools() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	exampleNames(aimcp.Tools(ctx, session, aimcp.Filter(func(t *mcp.Tool) bool { return t.Name == "read" })))
	// Output: files_read true
}

func ExamplePrefix() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	exampleNames(aimcp.Tools(ctx, session, aimcp.Prefix("fs_"), aimcp.Filter(func(t *mcp.Tool) bool { return t.Name == "read" })))
	// Output: fs_read true
}

func ExampleFilter() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	exampleNames(aimcp.Tools(ctx, session, aimcp.Filter(func(t *mcp.Tool) bool { return t.Name == "write" })))
	// Output: files_write true
}

func ExampleTrustHints() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	tools, err := aimcp.Tools(ctx, session, aimcp.TrustHints())
	fmt.Println(len(tools), err)
	// Output: 2 <nil>
}

func ExampleLocalTools() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	local := ai.NewClientTool[struct{}]("files_read", "The application's own reader.")
	exampleNames(aimcp.Tools(ctx, session, aimcp.LocalTools(local)))
	// Output: files_write true
}

func ExampleWithLimits() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	_, err := aimcp.Tools(ctx, session, aimcp.WithLimits(aimcp.Limits{MaxTools: 1}))
	fmt.Println(err)
	// Output: aimcp: tool definition or output exceeds its limit: more than 1 tools
}

func ExampleWithPins() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	var pins aimcp.Pins
	first, _ := aimcp.Tools(ctx, session, aimcp.WithPins(&pins))
	second, _ := aimcp.Tools(ctx, session, aimcp.WithPins(&pins))
	fmt.Println(len(first), len(second))
	// Output: 2 2
}

func ExampleRefresh() {
	ctx := context.Background()
	session := exampleSession(ctx)
	defer session.Close()
	var pins aimcp.Pins
	if _, err := aimcp.Tools(ctx, session, aimcp.WithPins(&pins)); err != nil {
		fmt.Println(err)
		return
	}
	tools, err := aimcp.Tools(ctx, session, aimcp.WithPins(&pins), aimcp.Refresh())
	fmt.Println(len(tools), err)
	// Output: 2 <nil>
}
