package pdata_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/internal/pdata"
)

type sample struct {
	Name string `json:"name,omitzero"`
	HTML string `json:"html,omitzero"`
}

func TestDecode_roundTrip(t *testing.T) {
	want := sample{Name: "n", HTML: "<b>"}
	got := pdata.Decode[sample]("p", pdata.Encode("p", want))
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Decode(Encode(%v)) mismatch (-want +got):\n%s", want, diff)
	}
}

func TestDecode_missingOrInvalid(t *testing.T) {
	tests := []struct {
		name string
		data ai.ProviderData
	}{
		{name: "missing", data: ai.ProviderData{"other": []byte(`{"name":"n"}`)}},
		{name: "invalid", data: ai.ProviderData{"p": []byte(`{"name":`)}},
		{name: "nil", data: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := pdata.Decode[sample]("p", tt.data); got != (sample{}) {
				t.Errorf("Decode() = %+v, want zero", got)
			}
		})
	}
}

func TestJSON_keepsHTML(t *testing.T) {
	if got, want := string(pdata.JSON("p", sample{HTML: "<b>"})), `{"html":"<b>"}`; got != want {
		t.Errorf("JSON() = %s, want %s", got, want)
	}
}
