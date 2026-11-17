package bedrock

import (
	"context"
	"errors"
	"os"
)

const (
	envAccessKeyID     = "AWS_ACCESS_KEY_ID"
	envSecretAccessKey = "AWS_SECRET_ACCESS_KEY"
	envSessionToken    = "AWS_SESSION_TOKEN"
)

var (
	ErrNoCredentials = errors.New("bedrock: no credentials: set AWS_BEARER_TOKEN_BEDROCK, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or use WithCredentials")
	ErrNoRegion      = errors.New("bedrock: no region: set AWS_REGION, or use WithRegion")
)

// Keys are AWS access keys. SessionToken is set for temporary credentials.
type Keys struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// Credentials supplies the keys that sign each request. Retrieve is called before every request, so implementations that fetch keys over the network should cache them.
type Credentials interface {
	Retrieve(ctx context.Context) (Keys, error)
}

// CredentialsFunc adapts a function to Credentials. It is the bridge to any credential chain, such as the one in aws-sdk-go-v2.
type CredentialsFunc func(ctx context.Context) (Keys, error)

func (f CredentialsFunc) Retrieve(ctx context.Context) (Keys, error) { return f(ctx) }

type staticCredentials Keys

type envCredentials struct{}

// StaticCredentials always returns k.
func StaticCredentials(k Keys) Credentials { return staticCredentials(k) }

// EnvCredentials reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN on every call.
func EnvCredentials() Credentials { return envCredentials{} }

func (s staticCredentials) Retrieve(context.Context) (Keys, error) {
	return Keys(s), nil
}

func (envCredentials) Retrieve(context.Context) (Keys, error) {
	return Keys{
		AccessKeyID:     os.Getenv(envAccessKeyID),
		SecretAccessKey: os.Getenv(envSecretAccessKey),
		SessionToken:    os.Getenv(envSessionToken),
	}, nil
}

var (
	_ Credentials = CredentialsFunc(nil)
	_ Credentials = staticCredentials{}
	_ Credentials = envCredentials{}
)
