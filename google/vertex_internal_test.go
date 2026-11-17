package google

import "testing"

func TestVertexBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		project  string
		location string
		want     string
	}{
		{name: "global", project: "p", location: "global", want: "https://aiplatform.googleapis.com/v1/projects/p/locations/global"},
		{name: "regional", project: "p", location: "europe-west4", want: "https://europe-west4-aiplatform.googleapis.com/v1/projects/p/locations/europe-west4"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := vertexBaseURL(tt.project, tt.location); got != tt.want {
				t.Errorf("vertexBaseURL(%q, %q) = %q, want %q", tt.project, tt.location, got, tt.want)
			}
		})
	}
}
