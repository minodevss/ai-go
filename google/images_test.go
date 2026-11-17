package google_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/google"
)

func TestImageModel_generate(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	m := srv.provider().ImageModel("gemini-nano-banana-2.1")
	res, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
		Prompt:      "Create a picture of a nano banana dish",
		AspectRatio: "16:9",
		Size:        "2k",
		Seed:        new(7),
		Provider:    ai.Providers(google.ImageOptions{SafetySettings: []google.SafetySetting{{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_ONLY_HIGH"}}}),
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(res.Images) != 1 || res.Images[0].MediaType != "image/png" || string(res.Images[0].Data) != "png-bytes" {
		t.Errorf("Images = %+v, want one png without the thought image", res.Images)
	}
	if res.Usage.Input != 12 || res.Usage.Output != 1290 {
		t.Errorf("Usage = %+v, want 12 in and 1290 out", res.Usage)
	}
	got := srv.last(t)
	if got.Path != "/v1beta/models/gemini-nano-banana-2.1:generateContent" {
		t.Errorf("path = %q", got.Path)
	}
	equalJSON(t, `{"contents":[{"role":"user","parts":[{"text":"Create a picture of a nano banana dish"}]}],"safetySettings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_ONLY_HIGH"}],"generationConfig":{"responseModalities":["TEXT","IMAGE"],"imageConfig":{"aspectRatio":"16:9","imageSize":"2K"},"seed":7}}`, got.Body)
}

func TestImageModel_editSendsReferenceImagesFirst(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	m := srv.provider().ImageModel("gemini-nano-banana-2.1")
	_, err := ai.GenerateImage(t.Context(), m, ai.ImageRequest{
		Prompt: "make it night",
		Files:  []ai.File{{MediaType: "image/png", Data: []byte("src")}, {MediaType: "image/jpeg", URL: "https://generativelanguage.googleapis.com/v1beta/files/abc"}},
	})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	equalJSON(t, `{"contents":[{"role":"user","parts":[{"inlineData":{"mimeType":"image/png","data":"c3Jj"}},{"fileData":{"mimeType":"image/jpeg","fileUri":"https://generativelanguage.googleapis.com/v1beta/files/abc"}},{"text":"make it night"}]}],"generationConfig":{"responseModalities":["TEXT","IMAGE"]}}`, srv.last(t).Body)
}

func TestImageModel_nSendsSequentialRequests(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("gemini-nano-banana-2.1"), ai.ImageRequest{Prompt: "p", N: 3})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(res.Images) != 3 || srv.count() != 3 || res.Usage.Input != 36 {
		t.Errorf("images = %d, requests = %d, input tokens = %d, want 3, 3, 36", len(res.Images), srv.count(), res.Usage.Input)
	}
}

func TestImageModel_vertexPath(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	p := vertexProvider(srv, func(context.Context) (string, error) { return "tok", nil })
	if _, err := ai.GenerateImage(t.Context(), p.ImageModel("gemini-3-pro-image"), ai.ImageRequest{Prompt: "p"}); err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if got := srv.last(t).Path; got != "/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-3-pro-image:generateContent" {
		t.Errorf("path = %q", got)
	}
}

func TestImageModel_errors(t *testing.T) {
	tests := []struct {
		name  string
		reply string
		req   ai.ImageRequest
		want  string
	}{
		{name: "blocked", reply: fixture(t, "image_blocked.json"), req: ai.ImageRequest{Prompt: "p"}, want: "image_safety"},
		{name: "mask", reply: fixture(t, "image_response.json"), req: ai.ImageRequest{Prompt: "p", Mask: &ai.File{Data: []byte("m")}}, want: "masks"},
		{name: "too many inputs", reply: fixture(t, "image_response.json"), req: ai.ImageRequest{Prompt: "p", Files: make([]ai.File, 15)}, want: "at most 14"},
		{name: "file without data", reply: fixture(t, "image_response.json"), req: ai.ImageRequest{Prompt: "p", Files: []ai.File{{MediaType: "image/png"}}}, want: "no data or url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, jsonReply(tt.reply))
			tt.req.MaxRetries = new(0)
			_, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("gemini-nano-banana-2.1"), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("GenerateImage() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestImageModel_unknownSizeWarns(t *testing.T) {
	srv := newFakeServer(t, jsonReply(fixture(t, "image_response.json")))
	res, err := ai.GenerateImage(t.Context(), srv.provider().ImageModel("gemini-nano-banana-2.1"), ai.ImageRequest{Prompt: "p", Size: "1024x1024"})
	if err != nil {
		t.Fatalf("GenerateImage() error = %v", err)
	}
	if len(res.Warnings) != 1 || !strings.Contains(res.Warnings[0].Message, "1024x1024") {
		t.Errorf("Warnings = %+v, want the size warning", res.Warnings)
	}
	var req struct {
		GenerationConfig struct {
			ImageConfig json.RawMessage `json:"imageConfig"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(srv.last(t).Body, &req); err != nil || len(req.GenerationConfig.ImageConfig) != 0 {
		t.Errorf("imageConfig = %s (%v), want none", req.GenerationConfig.ImageConfig, err)
	}
}
