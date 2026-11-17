package mediaio

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"syscall"
	"time"
)

const (
	maxDownloadRedirects = 5
	dialTimeout          = 30 * time.Second
)

var ErrBlockedAddress = errors.New("mediaio: refusing to connect to a non-public address")

func blockedNets() []netip.Prefix {
	return []netip.Prefix{netip.MustParsePrefix("100.64.0.0/10"), netip.MustParsePrefix("192.0.0.0/24"), netip.MustParsePrefix("198.18.0.0/15"), netip.MustParsePrefix("240.0.0.0/4")}
}

func blockedAddr(ip netip.Addr) bool {
	ip = ip.Unmap()
	if ip.Is6() && netip.MustParsePrefix("64:ff9b::/96").Contains(ip) {
		b := ip.As16()
		ip = netip.AddrFrom4([4]byte(b[12:]))
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() || !ip.IsValid() {
		return true
	}
	for _, p := range blockedNets() {
		if p.Contains(ip) {
			return true
		}
	}
	return false
}

func guardDial(_, address string, _ syscall.RawConn) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("%w: %q", ErrBlockedAddress, address)
	}
	ip, err := netip.ParseAddr(host)
	if err != nil || blockedAddr(ip) {
		return fmt.Errorf("%w: %q", ErrBlockedAddress, host)
	}
	return nil
}

func publicClient() (*http.Client, func()) {
	dialer := &net.Dialer{Timeout: dialTimeout, Control: guardDial}
	tr := &http.Transport{
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		TLSHandshakeTimeout:   dialTimeout,
		ResponseHeaderTimeout: 2 * time.Minute,
		DisableKeepAlives:     true,
	}
	client := &http.Client{
		Transport: tr,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if req.URL.Scheme != "https" {
				return fmt.Errorf("mediaio: refusing redirect to %s", req.URL.Redacted())
			}
			if len(via) >= maxDownloadRedirects {
				return fmt.Errorf("mediaio: stopped after %d redirects", maxDownloadRedirects)
			}
			return nil
		},
	}
	return client, tr.CloseIdleConnections
}
