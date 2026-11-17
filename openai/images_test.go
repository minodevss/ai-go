package openai_test

import (
	"encoding/base64"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/openai"
)

func TestImageModel_generate(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	m := srv.provider().ImageModel("gpt-image-2.5-flare")
	res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
		Prompt: "a calico cat", N: 2, Size: "1536x1024", AspectRatio: "3:2", Seed: new(3),
		Provider: ai.Providers(openai.ImageOptions{Quality: "high", OutputFormat: "webp", OutputCompression: new(80), Background: "transparent", Moderation: "low", SafetyIdentifier: "u-hash"}),
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	got := srv.last(t)
	if got.Path != "/v1/images/generations" || got.Header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("request = %s %v, want /v1/images/generations with the key", got.Path, got.Header)
	}
	equalJSON(t, `{"model":"gpt-image-2.5-flare","prompt":"a calico cat","n":2,"size":"1536x1024","quality":"high","output_format":"webp","output_compression":80,"background":"transparent","moderation":"low","safety_identifier":"u-hash"}`, got.Body)
	png, _ := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")
	if len(res.Images) != 1 || res.Images[0].MediaType != "image/png" || string(res.Images[0].Data) != string(png) {
		t.Errorf("images = %+v, want one png decoded from base64", res.Images)
	}
	if res.Usage.Input != 50 || res.Usage.Output != 50 {
		t.Errorf("usage = %+v, want 50/50", res.Usage)
	}
	if len(res.Warnings) != 2 || !strings.Contains(res.Warnings[0].Message, "aspectRatio") || !strings.Contains(res.Warnings[1].Message, "seed") {
		t.Errorf("warnings = %+v, want aspectRatio and seed", res.Warnings)
	}
}

func TestImageModel_defaultsKeepRequestMinimal(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	if _, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("gpt-image-1.5"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	equalJSON(t, `{"model":"gpt-image-1.5","prompt":"p"}`, srv.last(t).Body)
}

func TestImageModel_edit(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	m := srv.provider().ImageModel("gpt-image-2")
	mask := ai.File{MediaType: "image/png", Data: []byte("mask-bytes")}
	_, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
		Prompt: "add a hat", N: 1, Size: "1024x1024", Mask: &mask,
		Files:    []ai.File{{MediaType: "image/png", Filename: "a.png", Data: []byte("img-a")}, {MediaType: "image/jpeg", Data: []byte("img-b")}},
		Provider: ai.Providers(openai.ImageOptions{InputFidelity: "high", Quality: "low"}),
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	got := srv.last(t)
	if got.Path != "/v1/images/edits" {
		t.Fatalf("path = %s, want /v1/images/edits", got.Path)
	}
	parts := parseForm(t, got)
	wantFields := map[string]string{"model": "gpt-image-2", "prompt": "add a hat", "n": "1", "size": "1024x1024", "quality": "low", "input_fidelity": "high"}
	if diff := cmp.Diff(wantFields, formFields(parts)); diff != "" {
		t.Errorf("fields mismatch (-want +got):\n%s", diff)
	}
	images := formFiles(parts, "image[]")
	if len(images) != 2 || images[0].Filename != "a.png" || string(images[0].Data) != "img-a" || images[1].Type != "image/jpeg" || string(images[1].Data) != "img-b" {
		t.Errorf("image parts = %+v, want a.png and a jpeg", images)
	}
	if masks := formFiles(parts, "mask"); len(masks) != 1 || string(masks[0].Data) != "mask-bytes" {
		t.Errorf("mask parts = %+v, want the mask", masks)
	}
}

func TestImageModel_rejectsBadInput(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	m := srv.provider().ImageModel("gpt-image-2")
	tests := []struct {
		name string
		req  ai.ImageRequest
		want string
	}{
		{name: "mask without input images", req: ai.ImageRequest{Prompt: "p", Mask: &ai.File{Data: []byte("m")}}, want: "mask needs input"},
		{name: "input by url", req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{URL: "https://example.com/a.png"}}}, want: "needs Data"},
		{name: "bad options", req: ai.ImageRequest{Prompt: "p", Provider: ai.ProviderData{"openai": []byte(`{"quality":5}`)}}, want: "provider options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ai.GenerateImage(t.Context(), m, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) || !strings.HasPrefix(err.Error(), "openai:") {
				t.Errorf("GenerateImage() error = %v, want openai error containing %q", err, tt.want)
			}
		})
	}
	if srv.count() != 0 {
		t.Errorf("%d requests were sent for invalid input", srv.count())
	}
}

func TestImageModel_apiError(t *testing.T) {
	bad := newFakeServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Your request was rejected by the safety system.","type":"image_generation_user_error","code":"moderation_blocked"}}`))
	})
	_, err := ai.GenerateImage(t.Context(), bad.provider().ImageModel("gpt-image-2"), ai.ImageRequest{Prompt: "p"})
	apiErr, ok := errors.AsType[*ai.APIError](err)
	if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Code != "moderation_blocked" {
		t.Errorf("GenerateImage() error = %v, want a 400 moderation_blocked APIError", err)
	}
	if bad.count() != 1 {
		t.Errorf("400 was retried: %d requests", bad.count())
	}
}

func TestImageModel_decodeErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "bad base64", body: `{"data":[{"b64_json":"!!!"}]}`, want: "decode image 0"},
		{name: "empty item", body: `{"data":[{}]}`, want: "neither data nor url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.body))
			_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("m"), ai.ImageRequest{Prompt: "p"})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
	srv := newFakeServer(t, jsonReply(`{"data":[{"url":"https://example.com/x.png"}],"output_format":"jpeg"}`))
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("m"), ai.ImageRequest{Prompt: "p"})
	if err != nil || res.Images[0].URL != "https://example.com/x.png" || res.Images[0].MediaType != "image/jpeg" {
		t.Errorf("GenerateImage() = %+v, %v, want the url image", res, err)
	}
}
