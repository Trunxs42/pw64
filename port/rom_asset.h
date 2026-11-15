#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<unsigned char>;

static constexpr std::size_t fileTableStart = 0xDE720;
static constexpr std::size_t fileSystemStart = 0xDF5B0;

static void RequireRange(
    std::size_t offset,
    std::size_t length,
    std::size_t limit) {
	if (offset > limit || length > limit - offset) {
		throw std::runtime_error(
		    "A declared data range exceeds its containing region.");
		}
	}

static uint32_t ReadBE32(const Bytes& data, std::size_t offset) {
	RequireRange(offset, 4, data.size());

	return (uint32_t(data[offset]) << 24) |
	       (uint32_t(data[offset + 1]) << 16) |
	       (uint32_t(data[offset + 2]) << 8) |
	       uint32_t(data[offset + 3]);
	}

static std::string TagText(uint32_t tag) {
	std::string text(4, '.');

	for (int i = 0; i < 4; ++i) {
		const unsigned char c =
		    static_cast<unsigned char>(tag >> (24 - i * 8));

		if (c >= 32 && c <= 126) {
			text[i] = static_cast<char>(c);
			}
		}

	return text;
	}

static constexpr uint32_t MakeTag(
    char a, char b, char c, char d) {
	return (uint32_t(a) << 24) |
	       (uint32_t(b) << 16) |
	       (uint32_t(c) << 8) |
	       uint32_t(d);
	}

static Bytes DecodeMio0(
    const Bytes& input,
    std::size_t expectedSize) {
	if (input.size() < 16 ||
	        ReadBE32(input, 0) != MakeTag('M', 'I', 'O', '0')) {
		throw std::runtime_error("Missing MIO0 header.");
		}

	const std::size_t outputSize = ReadBE32(input, 4);
	const std::size_t compressedStart = ReadBE32(input, 8);
	const std::size_t literalStart = ReadBE32(input, 12);

	if (outputSize != expectedSize ||
	        outputSize > 64u * 1024u * 1024u) {
		throw std::runtime_error("Invalid MIO0 output size.");
		}

	if (compressedStart < 16 ||
	        compressedStart > literalStart ||
	        literalStart > input.size()) {
		throw std::runtime_error("Invalid MIO0 stream offsets.");
		}

	Bytes output;
	output.reserve(outputSize);

	std::size_t control = 16;
	std::size_t compressed = compressedStart;
	std::size_t literal = literalStart;

	unsigned int flags = 0;
	unsigned int mask = 0;

	while (output.size() < outputSize) {
		if (mask == 0) {
			RequireRange(control, 1, compressedStart);
			flags = input[control++];
			mask = 0x80;
			}

		if ((flags & mask) != 0) {
			RequireRange(literal, 1, input.size());
			output.push_back(input[literal++]);
			}
		else {
			RequireRange(compressed, 2, literalStart);

			const unsigned int token =
			    (static_cast<unsigned int>(input[compressed]) << 8) |
			    static_cast<unsigned int>(input[compressed + 1]);

			compressed += 2;

			const std::size_t length = (token >> 12) + 3;
			const std::size_t distance = (token & 0xFFF) + 1;

			if (distance > output.size() ||
			        length > outputSize - output.size()) {
				throw std::runtime_error(
				    "Invalid MIO0 back-reference.");
				}

			for (std::size_t i = 0; i < length; ++i) {
				const unsigned char value =
				    output[output.size() - distance];
				output.push_back(value);
				}
			}

		mask >>= 1;
		}

	return output;
	}

struct Chunk {
	uint32_t tag;
	std::size_t payload;
	std::size_t size;
	};

struct Container {
	uint32_t type;
	std::size_t size;
	std::vector<Chunk> chunks;
	};

static Container ReadContainer(
    const Bytes& data,
    std::size_t start,
    std::size_t available) {
	RequireRange(start, available, data.size());

	if (available < 12) {
		throw std::runtime_error("Container is shorter than its header.");
		}

	const uint64_t declaredSize =
	    uint64_t(ReadBE32(data, start + 4)) + 8;

	if (declaredSize < 12 || declaredSize > available) {
		throw std::runtime_error("Invalid container length.");
		}

	Container result{
		ReadBE32(data, start + 8),
		static_cast<std::size_t>(declaredSize),
			{}
		};

	const std::size_t end = start + result.size;
	std::size_t cursor = start + 12;

	while (cursor < end) {
		RequireRange(cursor, 8, end);

		const uint32_t tag = ReadBE32(data, cursor);
		const std::size_t size = ReadBE32(data, cursor + 4);
		const std::size_t payload = cursor + 8;

		RequireRange(payload, size, end);
		result.chunks.push_back({tag, payload, size});

		cursor = payload + size;
		}

	return result;
	}

static Bytes ReadLogicalChunk(
    const Bytes& rom,
    const Chunk& chunk,
    uint32_t& logicalTag) {
	logicalTag = chunk.tag;

	if (chunk.tag != MakeTag('G', 'Z', 'I', 'P')) {
		return Bytes(
		           rom.begin() + chunk.payload,
		           rom.begin() + chunk.payload + chunk.size);
		}

	if (chunk.size < 8) {
		throw std::runtime_error("Truncated GZIP wrapper.");
		}

	logicalTag = ReadBE32(rom, chunk.payload);

	const std::size_t expectedSize =
	    ReadBE32(rom, chunk.payload + 4);

	const std::size_t streamStart = chunk.payload + 8;
	const std::size_t streamEnd = chunk.payload + chunk.size;

	Bytes stream(
	    rom.begin() + streamStart,
	    rom.begin() + streamEnd);

	return DecodeMio0(stream, expectedSize);
	}