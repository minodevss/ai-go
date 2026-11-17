package ai

import (
	"context"
	"fmt"
	"maps"
	"slices"
)

type approvalRecord struct {
	request ToolApprovalRequest
	msgIdx  int
}

func (r *runner) resolveApprovals(ctx context.Context) (bool, error) {
	items, pending, err := planApprovals(r.messages)
	if err != nil {
		return false, err
	}
	if len(items) > 0 {
		h, err := r.handleCalls(ctx, r.req.Tools, items)
		r.insertResults(items, h.results)
		if err != nil {
			return false, err
		}
		if err := r.publishTools(h); err != nil {
			return false, err
		}
	}
	if len(pending) == 0 {
		return false, nil
	}
	r.finish = FinishApprovalPending
	r.pending = true
	for _, req := range pending {
		r.content = append(r.content, req)
		if !r.emit(req) {
			return false, errConsumerStopped
		}
	}
	return true, nil
}

func (r *runner) insertResults(items []callItem, results []ToolResult) {
	if len(results) == 0 {
		return
	}
	at := make(map[string]int, len(items))
	for _, it := range items {
		at[it.call.ID] = it.msgIdx
	}
	groups := map[int][]ToolResult{}
	for _, res := range results {
		groups[at[res.CallID]] = append(groups[at[res.CallID]], res)
	}
	for _, idx := range slices.Backward(slices.Sorted(maps.Keys(groups))) {
		m := Message{Role: RoleTool, Content: resultParts(groups[idx])}
		r.messages = slices.Insert(r.messages, idx+1, m)
		r.newMsgs = append(r.newMsgs, m)
	}
}

func planApprovals(msgs []Message) ([]callItem, []ToolApprovalRequest, error) {
	calls := map[string]ToolCall{}
	callIdx := map[string]int{}
	resolved := map[string]bool{}
	byID := map[string]approvalRecord{}
	var order []approvalRecord
	var responses []ToolApprovalResponse
	lastUser := -1
	for i, m := range msgs {
		if m.Role == RoleUser {
			lastUser = i
		}
		for _, p := range m.Content {
			switch v := p.(type) {
			case ToolCall:
				if m.Role == RoleAssistant && !v.ProviderExecuted {
					calls[v.ID] = v
					callIdx[v.ID] = i
				}
			case ToolApprovalRequest:
				if m.Role == RoleAssistant {
					rec := approvalRecord{request: v, msgIdx: i}
					byID[v.ApprovalID] = rec
					order = append(order, rec)
				}
			case ToolResult:
				resolved[v.CallID] = true
			case ToolApprovalResponse:
				if m.Role == RoleTool {
					responses = append(responses, v)
				}
			default:
			}
		}
	}
	answers := map[string]ToolApprovalResponse{}
	for _, resp := range responses {
		rec, ok := byID[resp.ApprovalID]
		if !ok || rec.request.CallID != resp.CallID {
			return nil, nil, fmt.Errorf("%w: unknown approval %q", ErrInvalidApproval, resp.ApprovalID)
		}
		if _, dup := answers[resp.ApprovalID]; !dup {
			answers[resp.ApprovalID] = resp
		}
	}
	var items []callItem
	var pending []ToolApprovalRequest
	for _, rec := range order {
		req := rec.request
		if resolved[req.CallID] {
			continue
		}
		call, ok := calls[req.CallID]
		if !ok {
			return nil, nil, fmt.Errorf("%w: approval %q has no tool call", ErrInvalidApproval, req.ApprovalID)
		}
		resp, answered := answers[req.ApprovalID]
		switch {
		case answered && resp.Approved:
			if inputHash(call.Name, call.Input) != req.InputHash {
				return nil, nil, fmt.Errorf("%w: input of call %q changed after approval was requested", ErrInvalidApproval, call.ID)
			}
			items = append(items, callItem{call: call, msgIdx: callIdx[call.ID], approved: true})
		case answered:
			items = append(items, callItem{call: call, msgIdx: callIdx[call.ID], denied: true, reason: resp.Reason})
		case rec.msgIdx < lastUser:
			items = append(items, callItem{call: call, msgIdx: callIdx[call.ID], denied: true, reason: "no approval was given"})
		default:
			pending = append(pending, req)
		}
	}
	return items, pending, nil
}
