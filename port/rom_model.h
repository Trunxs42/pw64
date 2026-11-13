#pragma once

#include "rom_asset.h"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iterator>

struct RomModelVertex {
	std::array<int16_t, 3> position{};
	uint16_t flag{};
	std::array<int16_t, 2> texture{};
	std::array<unsigned char, 4> color{};
	};

using RomModelMatrix =
    std::array<std::array<float, 4>, 4>;

struct RomPreviewPart {
	std::size_t firstBatch{};
	std::size_t batchCount{};
	uint8_t unknown5{};
	uint8_t depth{};
	};

struct RomPreviewBatch {
	uint32_t state{};
	std::size_t firstTriangle{};
	std::size_t triangleCount{};
	std::size_t partIndex{};
	};

struct RomPreviewModel {
	std::vector<RomModelVertex> vertices;
	std::vector<std::array<unsigned int, 3>> triangles;
	std::vector<RomPreviewBatch> batches;
	std::vector<RomPreviewPart> parts;
	std::vector<RomModelMatrix> matrices;
	};

class RomModelReader {
	public:
		explicit RomModelReader(const Bytes& bytes) : data(bytes) {}

		uint32_t U8() {
			RequireRange(cursor, 1, data.size());
			return data[cursor++];
			}

		uint32_t U16() {
			const uint32_t high = U8();
			const uint32_t low = U8();
			return (high << 8) | low;
			}

		int16_t S16() {
			const uint32_t value = U16();
			const int signedValue = value < 0x8000u
			                        ? static_cast<int>(value)
			                        : static_cast<int>(value) - 65536;

			return static_cast<int16_t>(signedValue);
			}

		uint32_t U32() {
			const uint32_t value = ReadBE32(data, cursor);
			cursor += 4;
			return value;
			}

		float Float() {
			const float value = std::bit_cast<float>(U32());

			if (!std::isfinite(value)) {
				throw std::runtime_error("Non-finite model float.");
				}

			return value;
			}

		void CheckEnd() {
			if (cursor > data.size()) {
				throw std::runtime_error(
				    "Model reader advanced past the payload.");
				}

			const std::size_t remaining = data.size() - cursor;

			const std::size_t paddingTo4 =
			    (4 - (cursor % 4)) % 4;

			const std::size_t paddingTo8 =
			    (8 - (cursor % 8)) % 8;

			if (remaining != 0 &&
			        remaining != paddingTo4 &&
			        remaining != paddingTo8) {
				throw std::runtime_error(
				    "Unexpected trailing model data.");
				}

			while (cursor < data.size()) {
				if (U8() != 0) {
					throw std::runtime_error(
					    "Nonzero trailing model data.");
					}
				}
			}

	private:
		const Bytes& data;
		std::size_t cursor = 0;
	};

inline Bytes FindRomChunk(
    const Bytes& rom,
    const Container& container,
    uint32_t wanted) {
	Bytes result;
	bool found = false;

	for (const Chunk& chunk : container.chunks) {
		uint32_t logicalTag = chunk.tag;

		if (chunk.tag == MakeTag('G', 'Z', 'I', 'P')) {
			if (chunk.size < 8) {
				throw std::runtime_error("Truncated compressed wrapper.");
				}

			logicalTag = ReadBE32(rom, chunk.payload);
			}

		if (logicalTag != wanted) {
			continue;
			}

		if (found) {
			throw std::runtime_error("Duplicate requested chunk.");
			}

		found = true;
		result = ReadLogicalChunk(rom, chunk, logicalTag);
		}

	if (!found) {
		throw std::runtime_error("Requested chunk was not found.");
		}

	return result;
	}

inline RomPreviewModel LoadRomPreviewModel(
    const std::string& romPath,
    std::size_t modelId) {
	std::ifstream input(romPath, std::ios::binary);

	if (!input) {
		throw std::runtime_error("Could not open local ROM: " + romPath);
		}

	Bytes rom{
		std::istreambuf_iterator<char>(input),
		std::istreambuf_iterator<char>()
		};

	if (input.bad()) {
		throw std::runtime_error("Error reading local ROM.");
		}

	// This preview uses the US offsets already checked by the inspector.
	RequireRange(fileTableStart, 12, rom.size());
	RequireRange(fileSystemStart, 0, rom.size());

	const Container table = ReadContainer(
	                            rom, fileTableStart, fileSystemStart - fileTableStart);

	const Bytes entries =
	    FindRomChunk(rom, table, MakeTag('T', 'A', 'B', 'L'));

	if (entries.size() % 8 != 0) {
		throw std::runtime_error("Invalid asset table size.");
		}

	std::size_t offset = fileSystemStart;
	std::size_t currentModel = 0;
	Bytes modelBytes;
	bool foundModel = false;

	for (std::size_t p = 0; p < entries.size(); p += 8) {
		const uint32_t type = ReadBE32(entries, p);
		const std::size_t storedSize = ReadBE32(entries, p + 4);

		RequireRange(offset, storedSize, rom.size());

		if (type == MakeTag('U', 'V', 'M', 'D')) {
			if (currentModel == modelId) {
				const Container asset =
				    ReadContainer(rom, offset, storedSize);

				if (asset.type != type) {
					throw std::runtime_error("Model container type mismatch.");
					}

				modelBytes =
				    FindRomChunk(rom, asset, MakeTag('C', 'O', 'M', 'M'));

				foundModel = true;
				break;
				}

			++currentModel;
			}

		offset += storedSize;
		}

	if (!foundModel) {
		throw std::runtime_error("Requested model ID was not found.");
		}

	RomModelReader reader(modelBytes);

	const uint32_t vertexCount = reader.U16();
	const uint32_t lodCount = reader.U8();
	const uint32_t matrixCount = reader.U8();
	const uint32_t extraCount = reader.U8();
	const uint32_t transparency = reader.U8();
	const uint32_t auxiliaryCount = reader.U16();

	if (vertexCount == 0 || vertexCount > 256 ||
	        lodCount != 1 || matrixCount == 0 || matrixCount > 8 ||
	        extraCount != 0 || transparency != 0 ||
	        auxiliaryCount != 0) {
		throw std::runtime_error(
		    "Model is outside the supported simple-preview profile.");
		}

	RomPreviewModel result;
	result.vertices.resize(vertexCount);

	for (RomModelVertex& vertex : result.vertices) {
		for (int axis = 0; axis < 3; ++axis) {
			vertex.position[axis] = reader.S16();
			}

		vertex.flag = static_cast<uint16_t>(reader.U16());

		for (int axis = 0; axis < 2; ++axis) {
			vertex.texture[axis] = reader.S16();
			}

		for (int component = 0; component < 4; ++component) {
			vertex.color[component] =
			    static_cast<unsigned char>(reader.U8());
			}
		}

	const uint32_t partCount = reader.U8();
	const uint32_t billboard = reader.U8();

	if (partCount == 0 || partCount > 8 ||
	        billboard != 0 || matrixCount != partCount) {
		throw std::runtime_error(
		    "Unsupported preview part/matrix layout.");
		}

	result.parts.reserve(partCount);
	result.batches.reserve(8);

	for (uint32_t partIndex = 0;
	        partIndex < partCount;
	        ++partIndex) {

		const uint32_t stateCount = reader.U8();
		const uint32_t unknown5 = reader.U8();
		const uint32_t depth = reader.U8();

		if (stateCount == 0 || stateCount > 8 ||
		        result.batches.size() + stateCount > 8 ||
		        depth > 8) {
			throw std::runtime_error(
			    "Unsupported preview part properties or batch count.");
			}

		RomPreviewPart part;
		part.firstBatch = result.batches.size();
		part.unknown5 = static_cast<uint8_t>(unknown5);
		part.depth = static_cast<uint8_t>(depth);

		for (uint32_t stateIndex = 0; stateIndex < stateCount; ++stateIndex) {
			RomPreviewBatch batch;
			batch.partIndex = partIndex;
			batch.state = reader.U32();
			batch.firstTriangle = result.triangles.size();

			const uint32_t transformCount = reader.U16();
			const uint32_t triangleCount = reader.U16();
			const uint32_t commandCount = reader.U16();

			const bool supportedMaterial =
			    batch.state == 0x00700FFFu ||
			    batch.state == 0x00720FFFu ||
			    batch.state == 0x04F00FFFu ||
			    batch.state == 0x00F001BFu ||
			    batch.state == 0x00700056u;

			if (!supportedMaterial ||
			        triangleCount > 128 || commandCount > 512) {
				throw std::runtime_error(
				    "Unsupported preview batch material or size.");
				}

			std::array<int, 32> slots;
			slots.fill(-1);

			uint32_t loadedVertices = 0;

			for (uint32_t cmd = 0; cmd < commandCount; ++cmd) {
				const uint32_t word = reader.U16();

				if ((word & 0x4000u) != 0) {
					const std::array<unsigned int, 3> indices{
						(word >> 8) & 0xFu,
						(word >> 4) & 0xFu,
						word & 0xFu
						};

					std::array<unsigned int, 3> triangle{};

					for (int corner = 0; corner < 3; ++corner) {
						const int vertex = slots[indices[corner]];

						if (vertex < 0) {
							throw std::runtime_error(
							    "Batch triangle references an unloaded vertex slot.");
							}

						triangle[corner] =
						    static_cast<unsigned int>(vertex);
						}

					result.triangles.push_back(triangle);
					}
				else {
					const uint32_t packed = reader.U8();
					const uint32_t first = word & 0x3FFFu;
					const uint32_t count = (packed >> 4) + 1;
					const uint32_t destination = packed & 0xFu;

					RequireRange(first, count, result.vertices.size());
					RequireRange(destination, count, slots.size());

					for (uint32_t i = 0; i < count; ++i) {
						slots[destination + i] =
						    static_cast<int>(first + i);
						}

					loadedVertices += count;
					}
				}

			batch.triangleCount =
			    result.triangles.size() - batch.firstTriangle;

			if (batch.triangleCount != triangleCount ||
			        loadedVertices != transformCount) {
				throw std::runtime_error(
				    "Batch geometry counts do not match.");
				}

			result.batches.push_back(batch);
			} // End state loop.

		part.batchCount = result.batches.size() - part.firstBatch;
		result.parts.push_back(part);
		} // End part loop.

	reader.Float(); // LOD radius.

	result.matrices.resize(matrixCount);

	for (RomModelMatrix& matrix : result.matrices) {
		for (int row = 0; row < 4; ++row) {
			for (int col = 0; col < 4; ++col) {
				matrix[row][col] = reader.Float();
				}
			}

		// The preview supports affine transforms, not projective part matrices.
		if (std::fabs(matrix[0][3]) > 0.00001f ||
		        std::fabs(matrix[1][3]) > 0.00001f ||
		        std::fabs(matrix[2][3]) > 0.00001f ||
		        std::fabs(matrix[3][3] - 1.0f) > 0.00001f) {
			throw std::runtime_error(
			    "Unsupported projective part matrix.");
			}
		}

	reader.Float();
	reader.Float();
	reader.Float();
	reader.CheckEnd();

	return result;
	}