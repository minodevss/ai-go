package bedrock_test

import (
	"context"
	"fmt"

	"github.com/minodevss/ai-go"
	"github.com/minodevss/ai-go/bedrock"
)

type awsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	CanExpire       bool
}

type awsCredentialsProvider interface {
	Retrieve(ctx context.Context) (awsCredentials, error)
}

type awsChain struct{}

func (awsChain) Retrieve(context.Context) (awsCredentials, error) {
	return awsCredentials{AccessKeyID: "ASIAEXAMPLE", SecretAccessKey: "secret", SessionToken: "session-token", CanExpire: true}, nil
}

func Example_awsSDKCredentials() {
	var provider awsCredentialsProvider = awsChain{}

	creds := bedrock.CredentialsFunc(func(ctx context.Context) (bedrock.Keys, error) {
		c, err := provider.Retrieve(ctx)
		return bedrock.Keys{AccessKeyID: c.AccessKeyID, SecretAccessKey: c.SecretAccessKey, SessionToken: c.SessionToken}, err
	})

	srv := exampleServer(200, exampleResponse)
	defer srv.Close()
	p := bedrock.New(bedrock.WithRegion("us-east-1"), bedrock.WithCredentials(creds), bedrock.WithBaseURL(srv.URL))
	res, err := ai.GenerateText(context.Background(), p.Model("amazon.nova-pro-v1:0"), ai.TextRequest{Messages: []ai.Message{ai.User("hi")}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(res.Text)
	// Output:
	// Hello from Converse.
}
