// Package emit wraps a provider-boundary yield function so that it is never called again after it reports false.
package emit

func New[T any](yield func(T, error) bool) func(T, error) bool {
	stopped := false
	return func(v T, err error) bool {
		if stopped || !yield(v, err) {
			stopped = true
		}
		return !stopped
	}
}
