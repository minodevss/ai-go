package ai

import (
	"context"
	"slices"
)

// EmbeddingMiddleware hooks run in order: TransformRequest on a clone of the call, then WrapEmbed around the model.
type EmbeddingMiddleware struct {
	TransformRequest func(ctx context.Context, c *EmbedCall) error
	WrapEmbed        func(ctx context.Context, c *EmbedCall, next func(context.Context, *EmbedCall) (*EmbedResponse, error)) (*EmbedResponse, error)
}

// ImageMiddleware hooks run in order: TransformRequest on a clone of the call, then WrapGenerate around the model.
type ImageMiddleware struct {
	TransformRequest func(ctx context.Context, c *ImageCall) error
	WrapGenerate     func(ctx context.Context, c *ImageCall, next func(context.Context, *ImageCall) (*ImageResponse, error)) (*ImageResponse, error)
}

type wrappedEmbeddingModel struct {
	inner EmbeddingModel
	mw    EmbeddingMiddleware
}

type wrappedImageModel struct {
	inner ImageModel
	mw    ImageMiddleware
}

func WrapEmbeddingModel(m EmbeddingModel, mw EmbeddingMiddleware) EmbeddingModel {
	return &wrappedEmbeddingModel{inner: m, mw: mw}
}

func WrapImageModel(m ImageModel, mw ImageMiddleware) ImageModel {
	return &wrappedImageModel{inner: m, mw: mw}
}

func (w *wrappedEmbeddingModel) Provider() string { return w.inner.Provider() }

func (w *wrappedEmbeddingModel) ModelID() string { return w.inner.ModelID() }

func (w *wrappedEmbeddingModel) MaxBatch() int { return w.inner.MaxBatch() }

func (w *wrappedEmbeddingModel) Unwrap() EmbeddingModel { return w.inner }

func (w *wrappedEmbeddingModel) Embed(ctx context.Context, c *EmbedCall) (*EmbedResponse, error) {
	c = c.Clone()
	if w.mw.TransformRequest != nil {
		if err := w.mw.TransformRequest(ctx, c); err != nil {
			return nil, err
		}
	}
	if w.mw.WrapEmbed == nil {
		return w.inner.Embed(ctx, c)
	}
	return w.mw.WrapEmbed(ctx, c, w.inner.Embed)
}

func (w *wrappedImageModel) Provider() string { return w.inner.Provider() }

func (w *wrappedImageModel) ModelID() string { return w.inner.ModelID() }

func (w *wrappedImageModel) Unwrap() ImageModel { return w.inner }

func (w *wrappedImageModel) GenerateImage(ctx context.Context, c *ImageCall) (*ImageResponse, error) {
	c = c.Clone()
	if w.mw.TransformRequest != nil {
		if err := w.mw.TransformRequest(ctx, c); err != nil {
			return nil, err
		}
	}
	if w.mw.WrapGenerate == nil {
		return w.inner.GenerateImage(ctx, c)
	}
	return w.mw.WrapGenerate(ctx, c, w.inner.GenerateImage)
}

func (c *EmbedCall) Clone() *EmbedCall {
	out := *c
	out.Inputs = make([]Parts, len(c.Inputs))
	for i, in := range c.Inputs {
		out.Inputs[i] = slices.Clone(in)
	}
	if c.Inputs == nil {
		out.Inputs = nil
	}
	out.Headers = c.Headers.Clone()
	out.Provider = c.Provider.clone()
	return &out
}

func (c *ImageCall) Clone() *ImageCall {
	out := *c
	out.Seed = clonePtr(c.Seed)
	out.Files = slices.Clone(c.Files)
	if c.Mask != nil {
		mask := *c.Mask
		out.Mask = &mask
	}
	out.Headers = c.Headers.Clone()
	out.Provider = c.Provider.clone()
	return &out
}
