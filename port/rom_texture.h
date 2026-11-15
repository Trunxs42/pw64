#pragma once

#include "rom_model.h"

#include <array>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

using RomPreviewTexture2 = std::array<unsigned char, 4096>;

struct RomPreviewTextureSpec {
	uint32_t width;
	uint32_t height;
	std::size_t imageBytes;
	std::size_t commandCount;
	std::size_t tileCommand;
	uint32_t tileWord0;
	uint32_t tileWord1;
	std::size_t rowBytes;
	bool normalizeOddRows;
	};

inline bool IsTerrainBaseTextureSupported(uint32_t textureId) {
	switch (textureId) {
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
		case 6:
		case 18:
		case 19:
		case 20:
		case 21:
		case 23:
		case 28:
		case 29:
		case 32:
		case 36:
		case 51:
		case 52:
		case 53:
		case 54:
			return true;

		default:
			return false;
		}
	}

inline RomPreviewTextureSpec GetRomPreviewTextureSpec(
    std::size_t textureId) {

	switch (textureId) {
		case 0:
			return {
				64, 16, 2736, 25, 12,
				0xF5102000u, 0x01010200u, 128, true
				};

		case 1:
			return {
				32, 16, 704, 25, 12,
				0xF5880800u, 0x01010050u, 32, true
				};

		case 2:
			return {
				32, 64, 4096, 13, 10,
				0xF5101000u, 0x01018250u, 64, false
				};
		case 3:
			return {
				64, 64, 2800, 25, 12,
				0xF5800800u, 0x01018060u, 32, true
				};

		case 4:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014050u, 64, true
				};

		case 5:
			return {
				64, 32, 1400, 25, 12,
				0xF5800800u, 0x01014060u, 32, true
				};

		case 6:
			return {
				64, 16, 2736, 25, 12,
				0xF5102000u, 0x01080200u, 128, true
				};

		case 18:
			return {
				32, 128, 3040, 25, 12,
				0xF5800400u, 0x0105C150u, 16, true
				};

		case 19:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014150u, 64, true
				};

		case 20:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014050u, 64, true
				};

		case 21:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014050u, 64, true
				};

		case 23:
			return {
				64, 32, 4096, 13, 10,
				0xF5102000u, 0x01014060u, 128, false
				};

		case 28:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014050u, 64, true
				};

		case 29:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01080050u, 64, true
				};

		case 32:
			return {
				16, 16, 248, 23, 12,
				0xF5800200u, 0x01050140u, 8, true
				};

		case 36:
			return {
				32, 32, 2744, 25, 12,
				0xF5101000u, 0x01014050u, 64, true
				};

		case 51:
			return {
				32, 64, 4096, 13, 10,
				0xF5101000u, 0x01098250u, 64, false
				};

		case 52:
			return {
				32, 64, 4096, 13, 10,
				0xF5101000u, 0x01098250u, 64, false
				};

		case 53:
			return {
				32, 47, 4056, 25, 12,
				0xF5101000u, 0x01080200u, 64, true
				};

		case 54:
			return {
				64, 23, 3976, 25, 12,
				0xF5102000u, 0x01080200u, 128, true
				};
		case 86:
			return {
				32, 32, 512, 13, 10,
				0xF5800400u, 0x01014050u, 16, false
				};

		case 102:
			return {
				16, 2, 16, 13, 10,
				0xF5800200u, 0x01004040u, 8, true
				};

		case 419:
			return {
				16, 16, 696, 23, 12,
				0xF5100800u, 0x01080040u, 32, true
				};

		case 420:
			return {
				2, 16, 248, 23, 12,
				0xF5100200u, 0x01080200u, 8, true
				};

		case 421:
			return {
				52, 28, 4008, 25, 12,
				0xF5101A00u, 0x01080200u, 104, true
				};

		case 422:
			return {
				32, 47, 4056, 25, 12,
				0xF5101000u, 0x01080200u, 64, true
				};

		case 424:
			return {
				64, 16, 1376, 25, 12,
				0xF5681000u, 0x01080200u, 64, true
				};

		case 426:
			return {
				48, 31, 4024, 25, 12,
				0xF5101800u, 0x01080200u, 96, true
				};

		case 447:
			return {
				32, 64, 2800, 25, 12,
				0xF5680800u, 0x01080200u, 32, true
				};

		default:
			throw std::runtime_error(
			    "Unsupported preview texture ID: " +
			    std::to_string(textureId));
		}
	}

inline RomPreviewTexture2 LoadRomPreviewTextureBytes(
    const std::string& romPath,
    std::size_t textureId) {

	const RomPreviewTextureSpec spec =
	    GetRomPreviewTextureSpec(textureId);

	std::ifstream input(romPath, std::ios::binary);

	if (!input) {
		throw std::runtime_error(
		    "Could not open local ROM: " + romPath);
		}

	Bytes rom{
		std::istreambuf_iterator<char>(input),
		std::istreambuf_iterator<char>()
		};

	if (input.bad()) {
		throw std::runtime_error("Error reading local ROM.");
		}

	RequireRange(fileTableStart, 12, rom.size());
	RequireRange(fileSystemStart, 0, rom.size());

	const Container table = ReadContainer(
	                            rom,
	                            fileTableStart,
	                            fileSystemStart - fileTableStart);

	const Bytes entries =
	    FindRomChunk(rom, table, MakeTag('T', 'A', 'B', 'L'));

	if (entries.size() % 8 != 0) {
		throw std::runtime_error("Invalid asset table size.");
		}

	std::size_t offset = fileSystemStart;
	std::size_t currentTexture = 0;
	Bytes texture;
	bool found = false;

	for (std::size_t p = 0; p < entries.size(); p += 8) {
		const uint32_t type = ReadBE32(entries, p);
		const std::size_t storedSize = ReadBE32(entries, p + 4);

		RequireRange(offset, storedSize, rom.size());

		if (type == MakeTag('U', 'V', 'T', 'X')) {
			if (currentTexture == textureId) {
				const Container asset =
				    ReadContainer(rom, offset, storedSize);

				if (asset.type != type) {
					throw std::runtime_error(
					    "Texture container type mismatch.");
					}

				texture = FindRomChunk(
				              rom, asset, MakeTag('C', 'O', 'M', 'M'));

				found = true;
				break;
				}

			++currentTexture;
			}

		offset += storedSize;
		}

	if (!found) {
		throw std::runtime_error("Requested ROM texture was not found.");
		}

	auto read16At = [&](std::size_t position) -> uint32_t {
		RequireRange(position, 2, texture.size());

		return (uint32_t(texture[position]) << 8) |
		uint32_t(texture[position + 1]);
		};

	RequireRange(0, 20, texture.size());

	const std::size_t imageBytes = read16At(0);
	const std::size_t commandCount = read16At(2);

	if (imageBytes != spec.imageBytes ||
	        commandCount != spec.commandCount) {
		throw std::runtime_error(
		    "Texture differs from the inspected layout: " +
		    std::to_string(textureId));
		}

	constexpr std::size_t imageOffset = 20;

	const std::size_t commandOffset =
	    imageOffset + imageBytes;

	const std::size_t commandBytes =
	    commandCount * 8;

	const std::size_t metadataOffset =
	    commandOffset + commandBytes;

	RequireRange(imageOffset, imageBytes, texture.size());
	RequireRange(commandOffset, commandBytes, texture.size());
	RequireRange(metadataOffset, 11, texture.size());
	RequireRange(spec.tileCommand, 2, commandCount);

	if (read16At(metadataOffset) != spec.width ||
	        read16At(metadataOffset + 2) != spec.height) {
		throw std::runtime_error(
		    "Texture dimensions differ from inspection: " +
		    std::to_string(textureId));
		}

	const std::size_t tileOffset =
	    commandOffset + spec.tileCommand * 8;

	if (ReadBE32(texture, tileOffset) != spec.tileWord0 ||
	        ReadBE32(texture, tileOffset + 4) != spec.tileWord1) {
		throw std::runtime_error(
		    "Texture render-tile setup differs from inspection: " +
		    std::to_string(textureId));
		}

	const uint32_t sizeWord0 =
	    ReadBE32(texture, tileOffset + 8);

	const uint32_t sizeWord1 =
	    ReadBE32(texture, tileOffset + 12);

	const uint32_t expectedSizeWord1 =
	    (1u << 24) |
	    (((spec.width - 1u) * 4u) << 12) |
	    ((spec.height - 1u) * 4u);

	if (sizeWord0 != 0xF2000000u ||
	        sizeWord1 != expectedSizeWord1) {
		throw std::runtime_error(
		    "Texture base-tile dimensions differ from inspection: " +
		    std::to_string(textureId));
		}

	if (textureId != 2 &&
	        ((read16At(metadataOffset + 7) & 0xFFFu) != textureId ||
	         read16At(metadataOffset + 9) != 0xFFFu)) {
		throw std::runtime_error(
		    "Texture has unexpected image dependencies: " +
		    std::to_string(textureId));
		}

	const std::size_t baseImageBytes =
	    spec.rowBytes * spec.height;

	RomPreviewTexture2 result{};

	RequireRange(0, baseImageBytes, imageBytes);
	RequireRange(0, baseImageBytes, result.size());

	if (spec.rowBytes == 0 || spec.rowBytes % 8 != 0) {
		throw std::runtime_error(
		    "Unsupported preview texture row stride.");
		}

	for (std::size_t y = 0; y < spec.height; ++y) {
		const std::size_t row = y * spec.rowBytes;

		for (std::size_t x = 0; x < spec.rowBytes; ++x) {
			const std::size_t sourceX =
			    spec.normalizeOddRows && (y & 1u)
			    ? (x ^ std::size_t{4})
			    : x;

			result[row + x] =
			    texture[imageOffset + row + sourceX];
			}
		}

	return result;
	}