package eventstream_test

import (
	"encoding/binary"
	"hash/crc32"
)

func crcOf(b []byte) []byte {
	return binary.BigEndian.AppendUint32(nil, crc32.ChecksumIEEE(b))
}
