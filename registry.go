package ai

import "strings"

// Registry resolves "provider:model" specs. An alias maps a short name to such a spec.
type Registry struct {
	Providers map[string]Provider
	Aliases   map[string]string
}

type embeddingProvider interface {
	EmbeddingModel(id string) (EmbeddingModel, error)
}

type rerankProvider interface {
	RerankModel(id string) (RerankModel, error)
}

func (r Registry) LanguageModel(spec string) (LanguageModel, error) {
	p, id, ok := r.resolve(spec)
	if !ok {
		return nil, &NoSuchModelError{Spec: spec, Kind: "language"}
	}
	return p.LanguageModel(id)
}

func (r Registry) EmbeddingModel(spec string) (EmbeddingModel, error) {
	p, id, ok := r.resolve(spec)
	if !ok {
		return nil, &NoSuchModelError{Spec: spec, Kind: "embedding"}
	}
	ep, ok := p.(embeddingProvider)
	if !ok {
		return nil, &NoSuchModelError{Spec: spec, Kind: "embedding"}
	}
	return ep.EmbeddingModel(id)
}

func (r Registry) RerankModel(spec string) (RerankModel, error) {
	p, id, ok := r.resolve(spec)
	if !ok {
		return nil, &NoSuchModelError{Spec: spec, Kind: "rerank"}
	}
	rp, ok := p.(rerankProvider)
	if !ok {
		return nil, &NoSuchModelError{Spec: spec, Kind: "rerank"}
	}
	return rp.RerankModel(id)
}

func (r Registry) resolve(spec string) (Provider, string, bool) {
	if target, ok := r.Aliases[spec]; ok {
		spec = target
	}
	name, id, ok := strings.Cut(spec, ":")
	if !ok || id == "" {
		return nil, "", false
	}
	p, ok := r.Providers[name]
	if !ok || p == nil {
		return nil, "", false
	}
	return p, id, true
}
