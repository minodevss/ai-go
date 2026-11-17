package emit_test

import (
	"errors"
	"testing"

	"github.com/minodevss/ai-go/internal/emit"
)

func TestNew_stopsAfterFalse(t *testing.T) {
	calls := 0
	send := emit.New(func(_ int, _ error) bool {
		calls++
		return calls < 2
	})
	got := []bool{send(1, nil), send(2, errors.New("x")), send(3, nil)}
	want := []bool{true, false, false}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("send #%d = %v, want %v", i+1, got[i], want[i])
		}
	}
	if calls != 2 {
		t.Errorf("yield calls = %d, want 2", calls)
	}
}
