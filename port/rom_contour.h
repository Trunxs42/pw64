#pragma once

#include "rom_model.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include "rom_texture.h"

inline RomPreviewModel LoadRomContourDiagnostic(
    const std::string& romPath,
    std::size_t contourId,
    bool textured = false) {

	if (contourId > 3) {
		throw std::runtime_error(
		    "Only inspected contours 0 through 3 are enabled.");
		}

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

	const Bytes entries = FindRomChunk(
	                          rom, table, MakeTag('T', 'A', 'B', 'L'));

	if (entries.size() % 8 != 0) {
		throw std::runtime_error("Invalid resource table size.");
		}

	std::size_t offset = fileSystemStart;
	std::size_t currentContour = 0;
	Bytes contour;
	bool found = false;

	for (std::size_t p = 0; p < entries.size(); p += 8) {
		const uint32_t type = ReadBE32(entries, p);
		const std::size_t size = ReadBE32(entries, p + 4);

		RequireRange(offset, size, rom.size());

		if (type == MakeTag('U', 'V', 'C', 'T')) {
			if (currentContour == contourId) {
				const Container asset =
				    ReadContainer(rom, offset, size);

				if (asset.type != type) {
					throw std::runtime_error(
					    "Contour container type mismatch.");
					}

				contour = FindRomChunk(
				              rom, asset, MakeTag('C', 'O', 'M', 'M'));

				found = true;
				break;
				}

			++currentContour;
			}

		offset += size;
		}

	if (!found) {
		throw std::runtime_error(
		    "Requested contour was not found.");
		}

	RomModelReader reader(contour);

	const uint32_t vertexCount = reader.U16();
	const uint32_t auxiliaryCount = reader.U16();
	const uint32_t staticObjectCount = reader.U16();
	const uint32_t sourceBatchCount = reader.U16();

	struct InspectedContour {
		std::size_t bytes;
		uint32_t vertices;
		uint32_t auxiliaryRecords;
		uint32_t staticObjects;
		uint32_t batches;
		std::size_t triangles;
		};

	constexpr std::array<InspectedContour, 4> inspected{{
				{21856, 724, 505, 40, 46, 505},
				{18768, 628, 382, 38, 44, 382},
				{22632, 675, 482, 56, 41, 482},
				{16384, 515, 382, 39, 25, 382}
			}};

	const InspectedContour& expected = inspected[contourId];

	if (contour.size() != expected.bytes ||
	        vertexCount != expected.vertices ||
	        auxiliaryCount != expected.auxiliaryRecords ||
	        staticObjectCount != expected.staticObjects ||
	        sourceBatchCount != expected.batches) {

		throw std::runtime_error(
		    "Contour differs from its inspected diagnostic layout.");
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

		for (unsigned char& component : vertex.color) {
			component = static_cast<unsigned char>(reader.U8());
			}

		// This diagnostic renders retained geometry as opaque.
		vertex.color[3] = 255;
		}

	auto skipBytes = [&](std::size_t count) {
		for (std::size_t i = 0; i < count; ++i) {
			reader.U8();
			}
		};

	skipBytes(std::size_t(auxiliaryCount) * 8);

	for (uint32_t object = 0;
	        object < staticObjectCount;
	        ++object) {

		const uint32_t matrixCount = reader.U8();
		skipBytes(std::size_t(matrixCount) * 64);

		reader.U16(); // Model ID.

		reader.Float();
		reader.Float();
		reader.Float();

		reader.U16();
		reader.U16();
		}

	std::size_t sourceTriangles = 0;
	std::size_t skippedBatches = 0;
	std::size_t skippedTriangles = 0;

	constexpr uint32_t excludedMaterialBits =
	    (1u << 23) | (1u << 24);

	constexpr uint32_t retainedGeometryBits =
	    (1u << 17) |
	    (1u << 19) |
	    (1u << 20) |
	    (1u << 21);

	for (uint32_t sourceBatch = 0;
	        sourceBatch < sourceBatchCount;
	        ++sourceBatch) {

		const uint32_t sourceState = reader.U32();
		const uint32_t expectedLoads = reader.U16();
		const uint32_t expectedTriangles = reader.U16();
		const uint32_t commandCount = reader.U16();

		const bool retain =
		    (sourceState & excludedMaterialBits) == 0;

		RomPreviewBatch batch{};
		batch.partIndex = 0;
		batch.firstTriangle = result.triangles.size();

		const uint32_t sourceTextureId = sourceState & 0xFFFu;

		const bool useTexture =
		    textured &&
		    IsTerrainBaseTextureSupported(sourceTextureId);

		batch.state =
		    (sourceState & retainedGeometryBits) |
		    (useTexture ? sourceTextureId : 0xFFFu);

		if (retain &&
		        textured &&
		        sourceTextureId != 0xFFFu &&
		        !useTexture) {

			std::cout
			        << "Terrain material fallback: texture "
			        << sourceTextureId
			        << " uses vertex colors only\n";
			}

		std::array<int, 32> slots;
		slots.fill(-1);

		uint32_t actualLoads = 0;
		uint32_t actualTriangles = 0;

		for (uint32_t command = 0;
		        command < commandCount;
		        ++command) {

			const uint32_t word = reader.U16();

			if ((word & 0x4000u) != 0) {
				const std::array<uint32_t, 3> indices{
					(word >> 8) & 0xFu,
					(word >> 4) & 0xFu,
					word & 0xFu
					};

				std::array<unsigned int, 3> triangle{};

				for (int corner = 0; corner < 3; ++corner) {
					const int vertex = slots[indices[corner]];

					if (vertex < 0) {
						throw std::runtime_error(
						    "Contour triangle references an unloaded slot.");
						}

					triangle[corner] =
					    static_cast<unsigned int>(vertex);
					}

				if (retain) {
					result.triangles.push_back(triangle);
					}

				++actualTriangles;
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

				actualLoads += count;
				}
			}

		if (actualLoads != expectedLoads ||
		        actualTriangles != expectedTriangles) {

			throw std::runtime_error(
			    "Contour batch geometry counts do not match.");
			}

		// Consume the metadata even for excluded batches.
		for (int field = 0; field < 4; ++field) {
			reader.U16();
			}

		for (int field = 0; field < 4; ++field) {
			reader.Float();
			}

		sourceTriangles += actualTriangles;

		if (retain) {
			if (result.batches.size() >= 64) {
				throw std::runtime_error(
				    "Contour diagnostic batch capacity exceeded.");
				}

			batch.triangleCount =
			    result.triangles.size() - batch.firstTriangle;

			result.batches.push_back(batch);
			}
		else {
			++skippedBatches;
			skippedTriangles += actualTriangles;
			}

		std::cout
		        << "Contour source batch " << sourceBatch
		        << ": state=0x" << std::hex << sourceState
		        << std::dec
		        << " triangles=" << actualTriangles
		        << (retain ? " retained" : " excluded")
		        << '\n';
		}

	for (int field = 0; field < 5; ++field) {
		reader.Float();
		}

	// Require the inspected layout to end with only zero alignment bytes.
	reader.CheckEnd();

	if (sourceTriangles != expected.triangles) {
		throw std::runtime_error(
		    "Contour triangle total differs from inspection.");
		}

	if (contourId == 3 &&
	        (result.batches.size() != 21 ||
	         result.triangles.size() != 332)) {

		throw std::runtime_error(
		    "Contour 3 diagnostic totals differ from inspection.");
		}

	RomPreviewPart part{};
	part.firstBatch = 0;
	part.batchCount = result.batches.size();
	part.depth = 0;
	result.parts.push_back(part);

	RomModelMatrix identity{};

	for (int axis = 0; axis < 4; ++axis) {
		identity[axis][axis] = 1.0f;
		}

	result.matrices.push_back(identity);

	std::cout
	        << (textured
	            ? "Contour diagnostic: supported base-level textures\n"
	            : "Contour diagnostic: vertex colors only\n")
	        << "Source vertices: " << vertexCount << '\n'
	        << "Source batches: " << sourceBatchCount << '\n'
	        << "Source triangles: " << sourceTriangles << '\n'
	        << "Excluded translucent/decal batches: "
	        << skippedBatches << '\n'
	        << "Excluded triangles: " << skippedTriangles << '\n'
	        << "Embedded static objects skipped: "
	        << staticObjectCount << '\n'
	        << "Rendered batches: " << result.batches.size() << '\n'
	        << "Rendered triangles: " << result.triangles.size() << '\n';

	return result;
	}

inline RomPreviewModel LoadRomTerrainDiagnostic(
    const std::string& romPath,
    std::size_t terrainId,
    bool textured = false) {

	if (terrainId != 0) {
		throw std::runtime_error(
		    "Only inspected terrain 0 is enabled.");
		}

	std::ifstream input(romPath, std::ios::binary);

	if (!input) {
		throw std::runtime_error("Could not open local ROM.");
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

	const Bytes entries = FindRomChunk(
	                          rom, table, MakeTag('T', 'A', 'B', 'L'));

	if (entries.size() % 8 != 0) {
		throw std::runtime_error("Invalid resource table size.");
		}

	Bytes terrain;
	bool found = false;
	std::size_t offset = fileSystemStart;

	for (std::size_t p = 0; p < entries.size(); p += 8) {
		const uint32_t type = ReadBE32(entries, p);
		const std::size_t size = ReadBE32(entries, p + 4);

		RequireRange(offset, size, rom.size());

		if (type == MakeTag('U', 'V', 'T', 'R')) {
			const Container asset =
			    ReadContainer(rom, offset, size);

			if (asset.type != type) {
				throw std::runtime_error(
				    "Terrain container type mismatch.");
				}

			std::size_t record = 0;

			for (const Chunk& chunk : asset.chunks) {
				uint32_t logicalTag = chunk.tag;

				if (chunk.tag == MakeTag('G', 'Z', 'I', 'P')) {
					RequireRange(0, 8, chunk.size);
					logicalTag = ReadBE32(rom, chunk.payload);
					}

				if (logicalTag != MakeTag('C', 'O', 'M', 'M')) {
					continue;
					}

				if (record == terrainId) {
					terrain = ReadLogicalChunk(
					              rom, chunk, logicalTag);
					found = true;
					break;
					}

				++record;
				}

			// Terrain IDs select COMM records in UVTR container 0.
			break;
			}

		offset += size;
		}

	if (!found) {
		throw std::runtime_error(
		    "Requested terrain COMM record was not found.");
		}

	RomModelReader reader(terrain);

	for (int field = 0; field < 6; ++field) {
		reader.Float();
		}

	const uint32_t columns = reader.U8();
	const uint32_t rows = reader.U8();

	const float cellWidth = reader.Float();
	const float cellHeight = reader.Float();
	reader.Float();

	if (terrain.size() != 312 ||
	        columns != 2 || rows != 2 ||
	        std::fabs(cellWidth - 600.0f) > 0.001f ||
	        std::fabs(cellHeight - 600.0f) > 0.001f) {

		throw std::runtime_error(
		    "Terrain 0 differs from its inspected grid.");
		}

	std::array<RomModelMatrix, 4> cellMatrices{};
	std::array<uint32_t, 4> contourIds{};

	constexpr std::array<std::array<float, 3>, 4> translations{{
				{-300.0f, -300.0f, 0.0f},
				{ 300.0f, -300.0f, 0.0f},
				{-300.0f,  300.0f, 0.0f},
				{ 300.0f,  300.0f, 0.0f}
			}};

	for (std::size_t cell = 0; cell < 4; ++cell) {
		if (reader.U8() != 1) {
			throw std::runtime_error(
			    "Unexpected terrain-cell presence value.");
			}

		RomModelMatrix& matrix = cellMatrices[cell];

		for (auto& row : matrix) {
			for (float& value : row) {
				value = reader.Float();
				}
			}

		const uint32_t unknownByte = reader.U8();
		contourIds[cell] = reader.U16();

		if (unknownByte != 0 || contourIds[cell] != cell) {
			throw std::runtime_error(
			    "Unexpected terrain-cell reference or byte field.");
			}

		for (int row = 0; row < 4; ++row) {
			for (int col = 0; col < 4; ++col) {
				float expected = row == col ? 1.0f : 0.0f;

				if (row == 3 && col < 3) {
					expected = translations[cell][col];
					}

				if (std::fabs(matrix[row][col] - expected) >
				        0.001f) {

					throw std::runtime_error(
					    "Terrain-cell matrix differs from inspection.");
					}
				}
			}
		}

	reader.CheckEnd();

	RomPreviewModel result;

	// A synthetic identity root keeps all terrain cells siblings.
	RomPreviewPart root{};
	result.parts.push_back(root);

	RomModelMatrix identity{};

	for (int axis = 0; axis < 4; ++axis) {
		identity[axis][axis] = 1.0f;
		}

	result.matrices.push_back(identity);

	std::size_t retainedSourceBatches = 0;

	for (std::size_t cell = 0; cell < 4; ++cell) {
		std::cout
		        << "\nLoading terrain cell " << cell
		        << ", contour " << contourIds[cell] << '\n';

		const RomPreviewModel source =
		    LoadRomContourDiagnostic(
		        romPath, contourIds[cell], textured);

		const std::size_t vertexBase = result.vertices.size();

		result.vertices.insert(
		          result.vertices.end(),
		          source.vertices.begin(),
		          source.vertices.end());

		using Triangle = std::array<unsigned int, 3>;

		std::map<uint32_t, std::vector<Triangle>> stateGroups;

		for (const RomPreviewBatch& sourceBatch : source.batches) {
			RequireRange(
			    sourceBatch.firstTriangle,
			    sourceBatch.triangleCount,
			    source.triangles.size());

			auto& triangles = stateGroups[sourceBatch.state];

			const std::size_t end =
			    sourceBatch.firstTriangle +
			    sourceBatch.triangleCount;

			for (std::size_t t = sourceBatch.firstTriangle;
			        t < end;
			        ++t) {

				Triangle triangle = source.triangles[t];

				for (unsigned int& index : triangle) {
					RequireRange(index, 1, source.vertices.size());

					index = static_cast<unsigned int>(
					            vertexBase + index);
					}

				triangles.push_back(triangle);
				}
			}

		RomPreviewPart part{};
		part.depth = 1;
		part.firstBatch = result.batches.size();

		const std::size_t partIndex = result.parts.size();

		for (const auto& [state, triangles] : stateGroups) {
			if (triangles.empty()) {
				continue;
				}

			if (result.batches.size() >=
			        (textured ? std::size_t{128} : kPreviewMaxBatches)) {
				throw std::runtime_error(
				    "Assembled terrain batch capacity exceeded.");
				}

			RomPreviewBatch batch{};
			batch.state = state;
			batch.partIndex = partIndex;
			batch.firstTriangle = result.triangles.size();
			batch.triangleCount = triangles.size();

			result.triangles.insert(
			          result.triangles.end(),
			          triangles.begin(),
			          triangles.end());

			result.batches.push_back(batch);
			}

		part.batchCount =
		    result.batches.size() - part.firstBatch;

		result.parts.push_back(part);
		result.matrices.push_back(cellMatrices[cell]);

		retainedSourceBatches += source.batches.size();

		std::cout
		        << "Cell " << cell
		        << ": retained source batches="
		        << source.batches.size()
		        << ", grouped diagnostic batches="
		        << part.batchCount
		        << ", triangles="
		        << source.triangles.size()
		        << '\n';
		}

	if (result.parts.size() > kPreviewMaxParts ||
	        result.vertices.size() != 2542 ||
	        result.triangles.empty()) {

		throw std::runtime_error(
		    "Invalid assembled terrain diagnostic.");
		}

	std::cout
	        << (textured
	            ? "\nTerrain diagnostic: four cells, base-level textures\n"
	            : "\nTerrain diagnostic: four cells, vertex colors only\n")
	        << "Terrain cells: 4\n"
	        << "Source vertices: " << result.vertices.size() << '\n'
	        << "Retained source batches: "
	        << retainedSourceBatches << '\n'
	        << "Grouped diagnostic batches: "
	        << result.batches.size() << '\n'
	        << "Rendered triangles: "
	        << result.triangles.size() << '\n'
	        << "Embedded objects and translucent/decal batches omitted\n";

	return result;
	}