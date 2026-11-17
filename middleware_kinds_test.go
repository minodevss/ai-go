package ai_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/aitest"
)

type imageStub struct {
	calls []ai.ImageCall
}

func (*imageStub) Provider() string { return "stub" }

func (*imageStub) ModelID() string { return "image" }

func (s *imageStub) GenerateImage(_ context.Context, c *ai.ImageCall) (*ai.ImageResponse, error) {
	s.calls = append(s.calls, *c)
	return &ai.ImageResponse{Images: []ai.File{{MediaType: "image/png", Data: []byte(c.Prompt)}}}, nil
}

func TestWrapEmbeddingModel_transformRequestSeesCloneOnly(t *testing.T) {
	inner := aitest.NewEmbeddingModel(aitest.MaxBatch(3))
	m := ai.WrapEmbeddingModel(inner, ai.EmbeddingMiddleware{TransformRequest: func(_ context.Context, c *ai.EmbedCall) error {
		c.Inputs[0] = ai.Parts{ai.Text{Text: "rewritten"}}
		c.Dimensions = 8
		return nil
	}})
	call := &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "original"}}}}
	if _, err := m.Embed(t.Context(), call); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if diff := cmp.Diff(ai.Parts{ai.Text{Text: "original"}}, call.Inputs[0]); diff != "" || call.Dimensions != 0 {
		t.Errorf("caller's call was mutated (-want +got):\n%s", diff)
	}
	seen := inner.Calls()[0]
	if rewritten, ok := seen.Inputs[0][0].(ai.Text); seen.Dimensions != 8 || !ok || rewritten.Text != "rewritten" {
		t.Errorf("inner model saw %+v, want the transformed call", seen)
	}
	if m.MaxBatch() != 3 || m.Provider() != "aitest" || m.ModelID() != "scripted-embedding" {
		t.Errorf("wrapper identity = %q %q %d", m.Provider(), m.ModelID(), m.MaxBatch())
	}
}

func TestWrapEmbeddingModel_wrapAndErrors(t *testing.T) {
	boom := errors.New("rejected")
	inner := aitest.NewEmbeddingModel()
	failing := ai.WrapEmbeddingModel(inner, ai.EmbeddingMiddleware{TransformRequest: func(context.Context, *ai.EmbedCall) error { return boom }})
	if _, err := failing.Embed(t.Context(), &ai.EmbedCall{}); !errors.Is(err, boom) {
		t.Errorf("Embed() error = %v, want %v", err, boom)
	}
	if len(inner.Calls()) != 0 {
		t.Error("inner model was called after a transform error")
	}
	wrapped := ai.WrapEmbeddingModel(inner, ai.EmbeddingMiddleware{WrapEmbed: func(ctx context.Context, c *ai.EmbedCall, next func(context.Context, *ai.EmbedCall) (*ai.EmbedResponse, error)) (*ai.EmbedResponse, error) {
		res, err := next(ctx, c)
		if err != nil {
			return nil, err
		}
		res.Warnings = append(res.Warnings, ai.Warning{Message: "wrapped"})
		return res, nil
	}})
	res, err := wrapped.Embed(t.Context(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "a"}}}})
	if err != nil || len(res.Warnings) != 1 {
		t.Errorf("Embed() = %+v, %v, want the wrapper's warning", res, err)
	}
}

func TestWrapImageModel_transformAndWrap(t *testing.T) {
	inner := &imageStub{}
	m := ai.WrapImageModel(inner, ai.ImageMiddleware{
		TransformRequest: func(_ context.Context, c *ai.ImageCall) error {
			c.Prompt = "cat, " + c.Prompt
			return nil
		},
		WrapGenerate: func(ctx context.Context, c *ai.ImageCall, next func(context.Context, *ai.ImageCall) (*ai.ImageResponse, error)) (*ai.ImageResponse, error) {
			res, err := next(ctx, c)
			if err != nil {
				return nil, err
			}
			res.Warnings = []ai.Warning{{Message: "wrapped"}}
			return res, nil
		},
	})
	call := &ai.ImageCall{Prompt: "sunset", Files: []ai.File{{URL: "u"}}}
	res, err := m.GenerateImage(t.Context(), call)
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if call.Prompt != "sunset" || inner.calls[0].Prompt != "cat, sunset" || string(res.Images[0].Data) != "cat, sunset" || len(res.Warnings) != 1 {
		t.Errorf("call = %+v, inner saw %+v, result = %+v", call, inner.calls[0], res)
	}
}

func TestAs_findsCapabilitiesThroughEmbeddingAndImageWrappers(t *testing.T) {
	type maxBatcher interface{ MaxBatch() int }
	inner := aitest.NewEmbeddingModel()
	emb := ai.WrapEmbeddingModel(ai.WrapEmbeddingModel(inner, ai.EmbeddingMiddleware{}), ai.EmbeddingMiddleware{})
	if got, ok := ai.As[*aitest.EmbeddingModel](emb); !ok || got != inner {
		t.Errorf("As[*aitest.EmbeddingModel] = %v, %v, want the innermost model", got, ok)
	}
	if _, ok := ai.As[maxBatcher](emb); !ok {
		t.Error("As[maxBatcher] = false, want true")
	}
	stub := &imageStub{}
	img := ai.WrapImageModel(ai.WrapImageModel(stub, ai.ImageMiddleware{}), ai.ImageMiddleware{})
	if got, ok := ai.As[*imageStub](img); !ok || got != stub {
		t.Errorf("As[*imageStub] = %v, %v, want the innermost model", got, ok)
	}
	if _, ok := ai.As[*imageStub](emb); ok {
		t.Error("As[*imageStub] on an embedding model = true, want false")
	}
}

func ExampleWrapEmbeddingModel() {
	base := aitest.NewEmbeddingModel()
	m := ai.WrapEmbeddingModel(base, ai.EmbeddingMiddleware{TransformRequest: func(_ context.Context, c *ai.EmbedCall) error {
		c.Dimensions = 256
		return nil
	}})
	if _, err := m.Embed(context.Background(), &ai.EmbedCall{Inputs: []ai.Parts{{ai.Text{Text: "hello"}}}}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(base.Calls()[0].Dimensions)
	// Output: 256
}

func ExampleWrapImageModel() {
	m := ai.WrapImageModel(&imageStub{}, ai.ImageMiddleware{TransformRequest: func(_ context.Context, c *ai.ImageCall) error {
		c.Prompt += ", watercolor"
		return nil
	}})
	res, err := m.GenerateImage(context.Background(), &ai.ImageCall{Prompt: "a lighthouse"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(res.Images[0].Data))
	// Output: a lighthouse, watercolor
}
