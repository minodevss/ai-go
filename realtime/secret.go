package realtime

import (
	"context"
	"fmt"
	"time"

	"github.com/minodevss/ai-go"
)

// MintOptions.TTL is how long the secret can start sessions; 0 = 60 seconds, and values above one hour are rejected.
type MintOptions struct {
	TTL time.Duration
}

// ClientSecret.Locked reports whether the provider refuses client changes to the model, instructions, tools and voice. When it is false, Warnings say so.
type ClientSecret struct {
	Value     string
	ExpiresAt time.Time
	Locked    bool
	Warnings  []ai.Warning
}

// MintClientSecret creates a short-lived credential for a client that connects to the provider directly. Tools run on the client, so tools with an Approval policy are rejected with ErrApprovalUnenforceable, and Execute functions are not sent.
func MintClientSecret(ctx context.Context, model Model, cfg Config, opts MintOptions) (*ClientSecret, error) {
	minter, ok := model.(SecretMinter)
	if !ok {
		return nil, fmt.Errorf("%w: client secrets", ErrUnsupported)
	}
	switch {
	case opts.TTL < 0:
		return nil, fmt.Errorf("realtime: negative secret TTL %s", opts.TTL)
	case opts.TTL > maxSecretTTL:
		return nil, fmt.Errorf("realtime: secret TTL %s exceeds the %s limit", opts.TTL, maxSecretTTL)
	case opts.TTL == 0:
		opts.TTL = defaultSecretTTL
	}
	var warnings []ai.Warning
	cfg = cfg.clone()
	for i, t := range cfg.Tools {
		if t.Approval != nil {
			return nil, fmt.Errorf("%w: tool %q", ErrApprovalUnenforceable, t.Name)
		}
		if t.Execute != nil {
			warnings = append(warnings, ai.Warning{Message: fmt.Sprintf("realtime: tool %q runs on the client; its Execute function is not used", t.Name)})
			cfg.Tools[i].Execute = nil
		}
	}
	secret, err := minter.MintClientSecret(ctx, &cfg, opts)
	if err != nil {
		return nil, err
	}
	if secret == nil || secret.Value == "" {
		return nil, fmt.Errorf("realtime: %s returned no client secret", model.Provider())
	}
	out := *secret
	out.Warnings = append(warnings, secret.Warnings...)
	return &out, nil
}
