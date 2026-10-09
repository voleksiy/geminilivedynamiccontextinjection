import struct

def pcm_rms(pcm_bytes: bytes) -> float:
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0.0
    # unpack 16-bit signed little endian integers
    samples = struct.unpack(f"<{count}h", pcm_bytes)
    sum_squares = sum(s * s for s in samples)
    return (sum_squares / count) ** 0.5

# Test with 1000 zero samples (pure silence)
silence = b"\x00\x00" * 1000
print("Silence RMS:", pcm_rms(silence))

# Test with low background noise (amplitude 50)
noise = (struct.pack("<h", 50) + struct.pack("<h", -50)) * 500
print("Background Noise RMS:", pcm_rms(noise))

# Test with speech (amplitude 2000)
speech = (struct.pack("<h", 2000) + struct.pack("<h", -1500)) * 500
print("Speech RMS:", pcm_rms(speech))
