package schema

import (
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

const (
	maxRootNodes  = 1 << 12
	maxTotalNodes = 1 << 17
	maxDepth      = 64
)

var ErrUnsafe = errors.New("schema: unsafe schema")

func sameInstanceSingle() []string { return []string{"not", "if", "then", "else"} }

func sameInstanceLists() []string { return []string{"allOf", "anyOf", "oneOf"} }

func descentSingle() []string {
	return []string{"items", "additionalItems", "additionalProperties", "contains", "propertyNames", "unevaluatedItems", "unevaluatedProperties"}
}

func descentLists() []string { return []string{"prefixItems"} }

func descentMaps() []string {
	return []string{"properties", "patternProperties", "dependentSchemas", "dependencies"}
}

type analysis struct {
	doc     any
	roots   map[string]bool
	pending []string
	total   int
}

func checkSafe(doc any) error {
	a := &analysis{doc: doc, roots: map[string]bool{"": true}, pending: []string{""}}
	for len(a.pending) > 0 {
		ptr := a.pending[len(a.pending)-1]
		a.pending = a.pending[:len(a.pending)-1]
		node, err := a.lookup(ptr)
		if err != nil {
			return err
		}
		budget := maxRootNodes
		if err := a.walk(node, ptr, map[string]bool{ptr: true}, &budget, 0); err != nil {
			return err
		}
	}
	return nil
}

func (a *analysis) walk(node any, ptr string, stack map[string]bool, budget *int, depth int) error {
	obj, ok := node.(map[string]any)
	if !ok {
		return nil
	}
	a.total++
	*budget--
	if *budget < 0 || a.total > maxTotalNodes {
		return fmt.Errorf("%w: too many schema expansions", ErrUnsafe)
	}
	if depth > maxDepth {
		return fmt.Errorf("%w: nesting deeper than %d", ErrUnsafe, maxDepth)
	}
	for _, key := range []string{"$ref", "$dynamicRef"} {
		ref, ok := obj[key].(string)
		if !ok {
			continue
		}
		target, ok := localPointer(ref)
		if !ok {
			return fmt.Errorf("%w: unsupported %s %q", ErrUnsafe, key, ref)
		}
		if stack[target] {
			return fmt.Errorf("%w: cyclic %s %q", ErrUnsafe, key, ref)
		}
		resolved, err := a.lookup(target)
		if err != nil {
			return err
		}
		stack[target] = true
		err = a.walk(resolved, target, stack, budget, depth+1)
		delete(stack, target)
		if err != nil {
			return err
		}
	}
	for _, key := range sameInstanceSingle() {
		if child, ok := obj[key]; ok {
			if err := a.walk(child, ptr+"/"+key, stack, budget, depth+1); err != nil {
				return err
			}
		}
	}
	for _, key := range sameInstanceLists() {
		list, _ := obj[key].([]any)
		for i, child := range list {
			if err := a.walk(child, ptr+"/"+key+"/"+strconv.Itoa(i), stack, budget, depth+1); err != nil {
				return err
			}
		}
	}
	for _, key := range descentSingle() {
		switch child := obj[key].(type) {
		case map[string]any:
			a.descend(ptr + "/" + key)
		case []any:
			for i := range child {
				a.descend(ptr + "/" + key + "/" + strconv.Itoa(i))
			}
		}
	}
	for _, key := range descentLists() {
		list, _ := obj[key].([]any)
		for i := range list {
			a.descend(ptr + "/" + key + "/" + strconv.Itoa(i))
		}
	}
	for _, key := range descentMaps() {
		children, _ := obj[key].(map[string]any)
		for name := range children {
			a.descend(ptr + "/" + key + "/" + escapeToken(name))
		}
	}
	return nil
}

func (a *analysis) descend(ptr string) {
	if a.roots[ptr] {
		return
	}
	a.roots[ptr] = true
	a.pending = append(a.pending, ptr)
}

func (a *analysis) lookup(ptr string) (any, error) {
	node := a.doc
	if ptr == "" {
		return node, nil
	}
	for token := range strings.SplitSeq(ptr[1:], "/") {
		token = unescapeToken(token)
		switch n := node.(type) {
		case map[string]any:
			child, ok := n[token]
			if !ok {
				return nil, fmt.Errorf("%w: unresolved reference %q", ErrUnsafe, ptr)
			}
			node = child
		case []any:
			i, err := strconv.Atoi(token)
			if err != nil || i < 0 || i >= len(n) {
				return nil, fmt.Errorf("%w: unresolved reference %q", ErrUnsafe, ptr)
			}
			node = n[i]
		default:
			return nil, fmt.Errorf("%w: unresolved reference %q", ErrUnsafe, ptr)
		}
	}
	return node, nil
}

func localPointer(ref string) (string, bool) {
	frag, ok := strings.CutPrefix(ref, "#")
	if !ok {
		return "", false
	}
	if frag == "" {
		return "", true
	}
	decoded, err := url.PathUnescape(frag)
	if err != nil || !strings.HasPrefix(decoded, "/") {
		return "", false
	}
	return normalizePointer(decoded), true
}

func normalizePointer(ptr string) string {
	parts := strings.Split(ptr[1:], "/")
	for i, p := range parts {
		parts[i] = escapeToken(unescapeToken(p))
	}
	return "/" + strings.Join(parts, "/")
}

func escapeToken(s string) string {
	return strings.NewReplacer("~", "~0", "/", "~1").Replace(s)
}

func unescapeToken(s string) string {
	return strings.NewReplacer("~1", "/", "~0", "~").Replace(s)
}
