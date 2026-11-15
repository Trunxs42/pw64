#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <imgui.h>
#include "controller_host.h"
#include <algorithm>
#include "rom_model.h"
#include "rom_contour.h"
#include "rom_texture.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <array>
#include <vector>
#include <map>
#include <string>

#include "fast/Fast3dWindow.h"
#include "libultraship/libultra/gbi.h"
#include "libultraship/controller/controldeck/ControlDeck.h"
#include "ship/Context.h"
#include "ship/resource/ResourceManager.h"
#include "ship/config/Config.h"
#include "ship/config/ConsoleVariable.h"

extern "C" {
#include <uv_vector.h>
#include <uv_matrix.h>
	}

using PreviewModelTextures =
    std::map<uint32_t, RomPreviewTexture2>;

static std::vector<Mtx4F> BuildPreviewPartTransforms(
    const RomPreviewModel& asset) {

	if (asset.parts.empty() ||
	        asset.parts.size() != asset.matrices.size() ||
	        asset.parts.size() > kPreviewMaxParts) {
		throw std::runtime_error(
		    "Invalid preview part/matrix counts.");
		}

	std::vector<Mtx4F> transforms(asset.parts.size());
	std::vector<std::size_t> ancestors;

	const unsigned int rootDepth = asset.parts.front().depth;

	for (std::size_t i = 0; i < asset.parts.size(); ++i) {
		const unsigned int storedDepth = asset.parts[i].depth;

		if (storedDepth < rootDepth) {
			throw std::runtime_error(
			    "Part hierarchy rises above its root.");
			}

		const std::size_t depth = storedDepth - rootDepth;

		if ((i != 0 && depth == 0) ||
		        depth > ancestors.size()) {
			throw std::runtime_error(
			    "Unsupported or discontinuous part hierarchy.");
			}

		ancestors.resize(depth);

		Mtx4F local{};

		for (int row = 0; row < 4; ++row) {
			for (int col = 0; col < 4; ++col) {
				local.m[row][col] = asset.matrices[i][row][col];
				}
			}

		if (depth == 0) {
			uvMat4Copy(&transforms[i], &local);
			}
		else {
			uvMat4Mul(
			    &transforms[i],
			    &local,
			    &transforms[ancestors.back()]);
			}

		ancestors.push_back(i);
		}

	return transforms;
	}

struct PreviewFraming {
	float center[3] {};
	float scale = 1.0f;
	};

struct TerrainPreviewCamera {
	float yaw = 0.0f;
	float tilt = -0.65f;
	float distance = 4.0f;
	};

static PreviewFraming CalculatePreviewFraming(
    const RomPreviewModel& asset,
    const std::vector<Mtx4F>& partTransforms) {

	std::vector<std::array<float, 3>> positions;

	for (const RomPreviewBatch& batch : asset.batches) {
		RequireRange(
		    batch.partIndex, 1, partTransforms.size());

		RequireRange(
		    batch.firstTriangle,
		    batch.triangleCount,
		    asset.triangles.size());

		const Mtx4F& matrix = partTransforms[batch.partIndex];

		const std::size_t end =
		    batch.firstTriangle + batch.triangleCount;

		for (std::size_t t = batch.firstTriangle; t < end; ++t) {
			for (unsigned int index : asset.triangles[t]) {
				RequireRange(index, 1, asset.vertices.size());

				const auto& source = asset.vertices[index].position;
				std::array<float, 3> position{};

				for (int axis = 0; axis < 3; ++axis) {
					position[axis] =
					    source[0] * matrix.m[0][axis] +
					    source[1] * matrix.m[1][axis] +
					    source[2] * matrix.m[2][axis] +
					    matrix.m[3][axis];

					if (!std::isfinite(position[axis])) {
						throw std::runtime_error(
						    "Non-finite assembled model position.");
						}
					}

				positions.push_back(position);
				}
			}
		}

	if (positions.empty()) {
		throw std::runtime_error(
		    "Cannot frame a model without triangles.");
		}

	auto minimum = positions.front();
	auto maximum = positions.front();

	for (const auto& position : positions) {
		for (int axis = 0; axis < 3; ++axis) {
			minimum[axis] =
			    std::min(minimum[axis], position[axis]);
			maximum[axis] =
			    std::max(maximum[axis], position[axis]);
			}
		}

	PreviewFraming framing;

	for (int axis = 0; axis < 3; ++axis) {
		framing.center[axis] =
		    (minimum[axis] + maximum[axis]) * 0.5f;
		}

	float radiusSquared = 0.0f;

	for (const auto& position : positions) {
		float distanceSquared = 0.0f;

		for (int axis = 0; axis < 3; ++axis) {
			const float offset =
			    position[axis] - framing.center[axis];

			distanceSquared += offset * offset;
			}

		radiusSquared = std::max(radiusSquared, distanceSquared);
		}

	if (!std::isfinite(radiusSquared) || radiusSquared <= 0.0f) {
		throw std::runtime_error(
		    "Cannot frame a zero-size or invalid model.");
		}

	framing.scale = 1.5f / std::sqrt(radiusSquared);
	return framing;
	}

static void PackHostMatrix(
    uint32_t (&packed)[16],
    const Mtx4F& matrix) {
	for (int row = 0; row < 4; ++row) {
		for (int pair = 0; pair < 2; ++pair) {
			const int col = pair * 2;

			const uint32_t a = static_cast<uint32_t>(
			                       static_cast<int32_t>(
			                           matrix.m[row][col] * 65536.0f + 0.5f));

			const uint32_t b = static_cast<uint32_t>(
			                       static_cast<int32_t>(
			                           matrix.m[row][col + 1] * 65536.0f + 0.5f));

			const int index = row * 2 + pair;

			packed[index] =
			    (a & 0xFFFF0000u) | (b >> 16);

			packed[index + 8] =
			    (a << 16) | (b & 0x0000FFFFu);
			}
		}
	}

struct PreviewDrawGroup {
	std::size_t batchIndex{};
	std::size_t vertexCount{};
	std::array<Vtx, 30> vertices{};
	};

static std::vector<PreviewDrawGroup> BuildPreviewDrawGroups(
    const RomPreviewModel& asset) {

	std::vector<PreviewDrawGroup> groups;

	for (std::size_t b = 0; b < asset.batches.size(); ++b) {
		const RomPreviewBatch& batch = asset.batches[b];

		RequireRange(
		    batch.firstTriangle,
		    batch.triangleCount,
		    asset.triangles.size());

		for (std::size_t first = 0;
		        first < batch.triangleCount;
		        first += 10) {

			PreviewDrawGroup group;
			group.batchIndex = b;

			const std::size_t triangleCount =
			    std::min<std::size_t>(
			        10, batch.triangleCount - first);

			for (std::size_t t = 0; t < triangleCount; ++t) {
				const auto& triangle =
				    asset.triangles[
				        batch.firstTriangle + first + t];

				for (int corner = 0; corner < 3; ++corner) {
					const std::size_t sourceIndex =
					    triangle[corner];

					RequireRange(
					    sourceIndex, 1, asset.vertices.size());

					const RomModelVertex& source =
					    asset.vertices[sourceIndex];

					Vtx& destination =
					    group.vertices[group.vertexCount++];

					for (int axis = 0; axis < 3; ++axis) {
						destination.v.ob[axis] =
						    source.position[axis];
						}

					destination.v.flag = source.flag;

					for (int axis = 0; axis < 2; ++axis) {
						destination.v.tc[axis] =
						    source.texture[axis];
						}

					for (int component = 0;
					        component < 4;
					        ++component) {
						destination.v.cn[component] =
						    source.color[component];
						}
					}
				}

			groups.push_back(group);
			}
		}

	return groups;
	}

static void BindAdditionalPreviewTexture(
    Gfx*& p,
    uint32_t textureId,
    const RomPreviewTexture2& texture) {

	const RomPreviewTextureSpec spec =
	    GetRomPreviewTextureSpec(textureId);

	if (textureId == 102u) {
		gDPSetCombineMode(
		    p++, G_CC_MODULATEI, G_CC_MODULATEI);

		gDPLoadTextureBlock_4b(
		    p++,
		    texture.data(),
		    G_IM_FMT_I,
		    16,
		    2,
		    0,
		    G_TX_CLAMP,
		    G_TX_CLAMP,
		    G_TX_NOMASK,
		    G_TX_NOMASK,
		    G_TX_NOLOD,
		    G_TX_NOLOD);

		return;
		}

	const bool isIA8 = textureId == 424u;

	const bool isRGBA16 =
	    textureId == 419u ||
	    textureId == 420u ||
	    textureId == 421u ||
	    textureId == 422u ||
	    textureId == 426u;

	if (!isIA8 && !isRGBA16) {
		throw std::runtime_error(
		    "Unsupported additional texture binding.");
		}

	const uint32_t format =
	    isIA8 ? G_IM_FMT_IA : G_IM_FMT_RGBA;

	const uint32_t size =
	    isIA8 ? G_IM_SIZ_8b : G_IM_SIZ_16b;

	const std::size_t bytesPerTexel =
	    isIA8 ? 1 : 2;

	const std::size_t baseImageBytes =
	    spec.rowBytes * spec.height;

	RequireRange(0, baseImageBytes, texture.size());

	if (isIA8) {
		gDPSetCombineMode(
		    p++,
		    G_CC_MODULATEIDECALA,
		    G_CC_MODULATEIDECALA);
		}
	else {
		gDPSetCombineMode(
		    p++,
		    G_CC_MODULATERGBA,
		    G_CC_MODULATERGBA);
		}

	// Preserve the stored row stride, including padding.
	gDPSetTextureImage(
	    p++,
	    format,
	    size,
	    spec.rowBytes / bytesPerTexel,
	    texture.data());

	gDPSetTile(
	    p++,
	    format,
	    size,
	    0,
	    0,
	    G_TX_LOADTILE,
	    0,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD);

	gDPLoadSync(p++);

	gDPLoadBlock(
	    p++,
	    G_TX_LOADTILE,
	    0,
	    0,
	    baseImageBytes / bytesPerTexel - 1,
	    0);

	gDPPipeSync(p++);

	gDPSetTile(
	    p++,
	    format,
	    size,
	    spec.rowBytes / 8,
	    0,
	    G_TX_RENDERTILE,
	    0,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD);

	gDPSetTileSize(
	    p++,
	    G_TX_RENDERTILE,
	    0,
	    0,
	    (spec.width - 1) << G_TEXTURE_IMAGE_FRAC,
	    (spec.height - 1) << G_TEXTURE_IMAGE_FRAC);
	}

static void BindTerrainBaseTexture(
    Gfx*& p,
    uint32_t textureId,
    const RomPreviewTexture2& texture) {

	if (!IsTerrainBaseTextureSupported(textureId)) {
		throw std::runtime_error(
		    "Unsupported terrain texture binding.");
		}

	const RomPreviewTextureSpec spec =
	    GetRomPreviewTextureSpec(textureId);

	const uint32_t format =
	    (spec.tileWord0 >> 21) & 7u;

	const uint32_t size =
	    (spec.tileWord0 >> 19) & 3u;

	const uint32_t palette =
	    (spec.tileWord1 >> 20) & 15u;

	const uint32_t clampMirrorT =
	    (spec.tileWord1 >> 18) & 3u;

	const uint32_t maskT =
	    (spec.tileWord1 >> 14) & 15u;

	const uint32_t clampMirrorS =
	    (spec.tileWord1 >> 8) & 3u;

	const uint32_t maskS =
	    (spec.tileWord1 >> 4) & 15u;

	const bool rgba16 =
	    format == G_IM_FMT_RGBA &&
	    size == G_IM_SIZ_16b;

	const bool intensity =
	    format == G_IM_FMT_I &&
	    (size == G_IM_SIZ_4b ||
	     size == G_IM_SIZ_8b);

	if (!rgba16 && !intensity) {
		throw std::runtime_error(
		    "Unsupported terrain texture format.");
		}

	const std::size_t baseImageBytes =
	    spec.rowBytes * spec.height;

	RequireRange(0, baseImageBytes, texture.size());

	if (baseImageBytes == 0 ||
	        baseImageBytes > 4096 ||
	        baseImageBytes % 2 != 0 ||
	        spec.rowBytes == 0 ||
	        spec.rowBytes % 8 != 0) {

		throw std::runtime_error(
		    "Invalid terrain base-image layout.");
		}

	gDPSetTexturePersp(p++, G_TP_PERSP);
	gDPSetTextureLOD(p++, G_TL_TILE);
	gDPSetTextureDetail(p++, G_TD_CLAMP);
	gDPSetTextureLUT(p++, G_TT_NONE);
	gDPSetTextureFilter(p++, G_TF_BILERP);
	gDPSetTextureConvert(p++, G_TC_FILT);

	if (rgba16) {
		gDPSetCombineMode(
		    p++, G_CC_MODULATERGBA, G_CC_MODULATERGBA);
		}
	else {
		gDPSetCombineMode(
		    p++, G_CC_MODULATEI, G_CC_MODULATEI);
		}

	// Transfer the normalized base image as 16-bit units.
	// Its actual sampling format is restored below.
	gDPSetTextureImage(
	    p++,
	    format,
	    G_IM_SIZ_16b,
	    1,
	    texture.data());

	gDPSetTile(
	    p++,
	    format,
	    G_IM_SIZ_16b,
	    0,
	    0,
	    G_TX_LOADTILE,
	    0,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD,
	    G_TX_CLAMP,
	    G_TX_NOMASK,
	    G_TX_NOLOD);

	gDPLoadSync(p++);

	gDPLoadBlock(
	    p++,
	    G_TX_LOADTILE,
	    0,
	    0,
	    baseImageBytes / 2 - 1,
	    0);

	gDPPipeSync(p++);

	// Remap the recorded base level onto render tile 0.
	// Base-level sampling uses no mip-level coordinate shifts.
	gDPSetTile(
	    p++,
	    format,
	    size,
	    spec.rowBytes / 8,
	    0,
	    G_TX_RENDERTILE,
	    palette,
	    clampMirrorT,
	    maskT,
	    G_TX_NOLOD,
	    clampMirrorS,
	    maskS,
	    G_TX_NOLOD);

	gDPSetTileSize(
	    p++,
	    G_TX_RENDERTILE,
	    0,
	    0,
	    (spec.width - 1) << G_TEXTURE_IMAGE_FRAC,
	    (spec.height - 1) << G_TEXTURE_IMAGE_FRAC);
	}

static Gfx* BuildGraphicsTest(
    int x, int y, float angle,
    const RomPreviewModel& asset,
    const PreviewFraming& framing,
    const RomPreviewTexture2& texture,
    const PreviewModelTextures& modelTextures,
    const std::vector<PreviewDrawGroup>& drawGroups,
    const std::vector<Mtx4F>& partTransforms,
    bool terrainView,
    const TerrainPreviewCamera& terrainCamera) {
	static Gfx commands[16384] {};
	alignas(8) static uint16_t colorBuffer[320 * 240] {};
	alignas(8) static uint16_t depthBuffer[320 * 240] {};

	// Persistent packed matrices used by the display list.
	alignas(8) static uint32_t projectionMatrix[16] {};
	alignas(8) static uint32_t partMatrices[kPreviewMaxParts][16] {};

	Mtx4F projection{};
	Mtx4F model{};

// Camera at the origin, looking along negative Z.
// Near plane: 1; far plane: 10.
// Horizontal extent accounts for the 320:240 aspect ratio.
	constexpr float aspect = 320.0f / 240.0f;

	uvMat4SetFrustrum(
	    &projection,
	    -aspect, aspect,
	    -1.0f, 1.0f,
	    1.0f, 10.0f);

	uvMat4SetIdentity(&model);

	const float viewDistance =
	    terrainView ? terrainCamera.distance : 4.0f;

	uvMat4LocalTranslate(
	    &model, 0.0f, 0.0f, -viewDistance);

	if (terrainView) {
		uvMat4RotateAxis(
		    &model, terrainCamera.tilt, 'x');

		// Terrain cells are arranged in the XY plane.
		uvMat4RotateAxis(
		    &model, terrainCamera.yaw, 'z');
		}
	else {
		// Preserve the existing model/contour rotation.
		uvMat4RotateAxis(&model, angle, 'x');
		uvMat4RotateAxis(&model, angle, 'y');
		}

	uvMat4Scale(
	    &model,
	    framing.scale,
	    framing.scale,
	    framing.scale);

	uvMat4LocalTranslate(
	    &model,
	    -framing.center[0],
	    -framing.center[1],
	    -framing.center[2]);

	PackHostMatrix(projectionMatrix, projection);

	if (partTransforms.size() != asset.parts.size() ||
	        partTransforms.empty() ||
	        partTransforms.size() > kPreviewMaxParts) {
		throw std::runtime_error(
		    "Invalid renderer part transforms.");
		}

	for (std::size_t i = 0; i < partTransforms.size(); ++i) {
		Mtx4F localToAssembled = partTransforms[i];
		Mtx4F modelView{};

		uvMat4Mul(&modelView, &localToAssembled, &model);
		PackHostMatrix(partMatrices[i], modelView);
		}

	static Vp viewport{};
	// A 120x80 viewport centered in the 320x240 logical screen.
	// Viewport X/Y values have two fractional bits.
	viewport.vp.vscale[0] = 160 * 4;
	viewport.vp.vscale[1] = 120 * 4;
	viewport.vp.vscale[2] = G_MAXZ / 2;
	viewport.vp.vscale[3] = 0;

	viewport.vp.vtrans[0] = 160 * 4;
	viewport.vp.vtrans[1] = 120 * 4;
	viewport.vp.vtrans[2] = G_MAXZ / 2;
	viewport.vp.vtrans[3] = 0;

	const std::size_t rendererBatchLimit =
	    terrainView ? std::size_t{128} :
	    kPreviewMaxBatches;

	if (asset.batches.size() > rendererBatchLimit) {
		throw std::runtime_error(
		    "Renderer batch capacity exceeded.");
		}

// Conservative budget for the current renderer:
// fixed frame/overlay setup, material setup, and geometry.
// Revisit the material budget when adding mipmapped commands.
	constexpr std::size_t fixedCommandBudget = 128;
	constexpr std::size_t commandsPerBatchBudget = 64;
	constexpr std::size_t commandCapacity =
	    sizeof(commands) / sizeof(commands[0]);

	std::size_t commandBudget =
	    fixedCommandBudget +
	    asset.batches.size() * commandsPerBatchBudget;

	for (const PreviewDrawGroup& group : drawGroups) {
		RequireRange(group.batchIndex, 1, asset.batches.size());

		if (group.vertexCount == 0 ||
		        group.vertexCount > group.vertices.size() ||
		        group.vertexCount % 3 != 0) {
			throw std::runtime_error(
			    "Invalid preview draw-group size.");
			}

		const std::size_t groupCommands =
		    1 + group.vertexCount / 3;

		if (commandBudget > commandCapacity ||
		        groupCommands > commandCapacity - commandBudget) {
			throw std::runtime_error(
			    "Preview display-list capacity exceeded.");
			}

		commandBudget += groupCommands;
		}

	if (commandBudget > commandCapacity) {
		throw std::runtime_error(
		    "Preview display-list capacity exceeded.");
		}

	Gfx* p = commands;

	gDPPipeSync(p++);
	gDPSetScissor(p++, G_SC_NON_INTERLACE, 0, 0, 320, 240);

	gDPSetCycleType(p++, G_CYC_FILL);
	gDPSetRenderMode(p++, G_RM_NOOP, G_RM_NOOP2);

// Identify the depth buffer.
	gDPSetDepthImage(p++, depthBuffer);

// Temporarily target it with fill commands to clear depth.
	gDPSetColorImage(
	    p++, G_IM_FMT_RGBA, G_IM_SIZ_16b, 320, depthBuffer);

	const uint32_t clearDepth = GPACK_ZDZ(G_MAXFBZ, 0);

	gDPSetFillColor(
	    p++, (clearDepth << 16) | clearDepth);
	gDPFillRectangle(p++, 0, 0, 319, 239);

	gDPPipeSync(p++);

// Restore the color target before clearing the background.
	gDPSetColorImage(
	    p++, G_IM_FMT_RGBA, G_IM_SIZ_16b, 320, colorBuffer);

	// Clear the entire background.
	const uint32_t background =
	    GPACK_RGBA5551(160, 176, 192, 1);

	gDPSetFillColor(
	    p++, (background << 16) | background);
	gDPFillRectangle(p++, 0, 0, 319, 239);

	gDPPipeSync(p++);

	if (!terrainView) {
		const uint32_t rectangle =
		    GPACK_RGBA5551(255, 160, 32, 1);

		gDPSetFillColor(
		    p++, (rectangle << 16) | rectangle);

		gDPFillRectangle(
		    p++, x, y, x + 63, y + 47);
		}

	gDPPipeSync(p++);

// Shared triangle-rendering setup; materials are applied per batch.
	gDPSetCycleType(p++, G_CYC_1CYCLE);
	gDPSetDepthSource(p++, G_ZS_PIXEL);
	gDPSetAlphaCompare(p++, G_AC_NONE);

	gSPViewport(p++, &viewport);

	gSPMatrix(
	    p++, projectionMatrix,
	    G_MTX_PROJECTION | G_MTX_LOAD | G_MTX_NOPUSH);

	for (std::size_t batchIndex = 0;
	        batchIndex < asset.batches.size();
	        ++batchIndex) {

		const RomPreviewBatch& batch = asset.batches[batchIndex];

		RequireRange(batch.partIndex, 1, partTransforms.size());

		gSPMatrix(
		    p++,
		    partMatrices[batch.partIndex],
		    G_MTX_MODELVIEW | G_MTX_LOAD | G_MTX_NOPUSH);

		gDPPipeSync(p++);

		const bool translucent =
		    (batch.state & (1u << 23)) != 0;

		const bool decal =
		    (batch.state & (1u << 24)) != 0;

		if (decal) {
			if (!translucent) {
				throw std::runtime_error(
				    "Opaque decal preview is not enabled.");
				}

			gDPSetRenderMode(
			    p++,
			    G_RM_ZB_XLU_DECAL,
			    G_RM_ZB_XLU_DECAL2);
			}
		else if (translucent) {
			gDPSetRenderMode(
			    p++, G_RM_ZB_XLU_SURF, G_RM_ZB_XLU_SURF2);
			}
		else {
			gDPSetRenderMode(
			    p++, G_RM_ZB_OPA_SURF, G_RM_ZB_OPA_SURF2);
			}

		uint32_t geometryMode = G_SHADE;

		if ((batch.state & (1u << 17)) != 0) {
			geometryMode |= G_SHADING_SMOOTH;
			}

		if ((batch.state & (1u << 19)) != 0) {
			geometryMode |= G_CULL_FRONT;
			}

		if ((batch.state & (1u << 20)) != 0) {
			geometryMode |= G_CULL_BACK;
			}

		if ((batch.state & (1u << 21)) != 0) {
			geometryMode |= G_ZBUFFER;
			}

		gSPClearGeometryMode(p++, 0xFFFFFFFFu);
		gSPSetGeometryMode(p++, geometryMode);

		// Reset state explicitly so a previous textured batch cannot leak.
		gSPTexture(p++, 0, 0, 0, G_TX_RENDERTILE, G_OFF);
		gDPSetCombineMode(p++, G_CC_SHADE, G_CC_SHADE);

		const uint32_t textureId = batch.state & 0xFFFu;

		if (textureId != 0xFFFu) {
			GetRomPreviewTextureSpec(textureId);

			const auto entry = modelTextures.find(textureId);

			if (entry == modelTextures.end()) {
				throw std::runtime_error(
				    "Missing loaded batch texture ID: " +
				    std::to_string(textureId));
				}

			const RomPreviewTexture2* modelTexture = &entry->second;

			gDPSetTexturePersp(p++, G_TP_PERSP);
			gDPSetTextureLOD(p++, G_TL_TILE);
			gDPSetTextureDetail(p++, G_TD_CLAMP);
			gDPSetTextureLUT(p++, G_TT_NONE);
			gDPSetTextureFilter(p++, G_TF_BILERP);
			gDPSetTextureConvert(p++, G_TC_FILT);

			if (terrainView) {
				BindTerrainBaseTexture(
				    p, textureId, *modelTexture);
				}
			else if (textureId == 86u) {
				gDPSetCombineMode(
				    p++, G_CC_MODULATEI, G_CC_MODULATEI);

				gDPLoadTextureBlock_4b(
				    p++,
				    modelTexture->data(),
				    G_IM_FMT_I,
				    32,
				    32,
				    0,
				    G_TX_CLAMP,
				    G_TX_CLAMP,
				    G_TX_NOMASK,
				    G_TX_NOMASK,
				    G_TX_NOLOD,
				    G_TX_NOLOD);
				}
			else if (textureId == 447u) {
				gDPSetCombineMode(
				    p++, G_CC_MODULATEIDECALA, G_CC_MODULATEIDECALA);

				gDPLoadTextureBlock(
				    p++,
				    modelTexture->data(),
				    G_IM_FMT_IA,
				    G_IM_SIZ_8b,
				    32,
				    64,
				    0,
				    G_TX_CLAMP,
				    G_TX_CLAMP,
				    G_TX_NOMASK,
				    G_TX_NOMASK,
				    G_TX_NOLOD,
				    G_TX_NOLOD);
				}

			else {
				BindAdditionalPreviewTexture(
				    p, textureId, *modelTexture);
				}

			gSPTexture(
			    p++, 0xFFFF, 0xFFFF, 0, G_TX_RENDERTILE, G_ON);
			}

		for (const PreviewDrawGroup& group : drawGroups) {
			if (group.batchIndex != batchIndex) {
				continue;
				}

			const Gfx vertexLoad = {
				gsSPVertex(group.vertices.data(), group.vertexCount, 0)
				};
			*p++ = vertexLoad;

			for (std::size_t v = 0; v < group.vertexCount; v += 3) {
				gSP1Triangle(p++, v, v + 1, v + 2, 0);
				} // End triangle loop.
			} // End draw-group loop.
		} // End material-batch loop.

	if (!terrainView) {
		// Preserve the texture overlay in model/contour previews.
		gDPPipeSync(p++);

		gDPSetCycleType(p++, G_CYC_1CYCLE);
		gDPSetRenderMode(p++, G_RM_XLU_SURF, G_RM_XLU_SURF2);
		gDPSetAlphaCompare(p++, G_AC_NONE);

		gDPSetTexturePersp(p++, G_TP_NONE);
		gDPSetTextureLOD(p++, G_TL_TILE);
		gDPSetTextureDetail(p++, G_TD_CLAMP);
		gDPSetTextureLUT(p++, G_TT_NONE);
		gDPSetTextureFilter(p++, G_TF_POINT);
		gDPSetTextureConvert(p++, G_TC_FILT);

		gDPSetCombineMode(
		    p++, G_CC_DECALRGBA, G_CC_DECALRGBA);

		gDPLoadTextureBlock(
		    p++,
		    texture.data(),
		    G_IM_FMT_RGBA,
		    G_IM_SIZ_16b,
		    32,
		    64,
		    0,
		    G_TX_CLAMP,
		    G_TX_CLAMP,
		    G_TX_NOMASK,
		    G_TX_NOMASK,
		    G_TX_NOLOD,
		    G_TX_NOLOD);

		gSPTextureRectangle(
		    p++,
		    8 << 2,
		    8 << 2,
		    72 << 2,
		    136 << 2,
		    G_TX_RENDERTILE,
		    0,
		    0,
		    512,
		    512);
		}

	gDPPipeSync(p++);

// Do not leave perspective correction disabled for later frames.
	gDPSetTexturePersp(p++, G_TP_PERSP);

	gDPFullSync(p++);
	gSPEndDisplayList(p++);

	return commands;
	}

int main(int argc, char* argv[]) {
	try {
		const bool contourPreview =
		    argc == 3 &&
		    std::string_view(argv[1]) == "--contour";

		const bool terrainTextured =
		    argc == 3 &&
		    std::string_view(argv[1]) == "--terrain-textured";

		const bool terrainPreview =
		    terrainTextured ||
		    (argc == 3 &&
		     std::string_view(argv[1]) == "--terrain");

		if (argc != 1 && argc != 2 &&
		        !contourPreview && !terrainPreview) {

			std::cerr
			        << "Usage: " << argv[0] << " [model-id]\n"
			        << "       " << argv[0] << " --contour 3\n"
			        << "       " << argv[0] << " --terrain 0\n"
			        << "       " << argv[0] << " --terrain-textured 0\n";
			return 1;
			}

		std::size_t modelId = 2;

		if (argc >= 2) {
			const std::string_view argument(
			    argv[(contourPreview || terrainPreview) ? 2 : 1]);

			if (argument.empty()) {
				throw std::runtime_error(
				    "Asset ID must be a non-negative decimal integer.");
				}

			const auto parsed = std::from_chars(
			                        argument.data(),
			                        argument.data() + argument.size(),
			                        modelId,
			                        10);

			if (parsed.ec != std::errc{} ||
			        parsed.ptr != argument.data() + argument.size()) {

				throw std::runtime_error(
				    "Asset ID must be a non-negative decimal integer.");
				}
			}

		if (contourPreview && modelId != 3) {
			throw std::runtime_error(
			    "The standalone contour regression mode remains limited to 3.");
			}

		const RomPreviewModel loadedModel =
		    terrainPreview
		    ? LoadRomTerrainDiagnostic(
		        "baserom.us.z64", modelId, terrainTextured)
		    : contourPreview
		    ? LoadRomContourDiagnostic(
		        "baserom.us.z64", modelId)
		    : LoadRomPreviewModel(
		        "baserom.us.z64", modelId);

		const std::vector<Mtx4F> partTransforms =
		    BuildPreviewPartTransforms(loadedModel);

		const PreviewFraming framing =
		    CalculatePreviewFraming(loadedModel, partTransforms);

		std::cout << "Parts: " << loadedModel.parts.size() << '\n';

		for (std::size_t i = 0; i < loadedModel.parts.size(); ++i) {
			const RomPreviewPart& part = loadedModel.parts[i];
			const Mtx4F& accumulated = partTransforms[i];

			std::cout
			        << "  Part " << i
			        << ": depth=" << static_cast<unsigned int>(part.depth)
			        << " firstBatch=" << part.firstBatch
			        << " batches=" << part.batchCount
			        << " assembled translation=("
			        << accumulated.m[3][0] << ", "
			        << accumulated.m[3][1] << ", "
			        << accumulated.m[3][2] << ")\n";
			}

		const RomPreviewTexture2 loadedTexture =
		    LoadRomPreviewTextureBytes("baserom.us.z64", 2);

		PreviewModelTextures modelTextures;

		for (const RomPreviewBatch& batch : loadedModel.batches) {
			const uint32_t textureId = batch.state & 0xFFFu;

			if (textureId == 0xFFFu) {
				continue;
				}

			GetRomPreviewTextureSpec(textureId);

			if (modelTextures.find(textureId) != modelTextures.end()) {
				continue;
				}

			modelTextures.emplace(
			                 textureId,
			                 LoadRomPreviewTextureBytes("baserom.us.z64", textureId));

			std::cout
			        << "Loaded model texture " << textureId << '\n';
			}

		std::cout
		        << "Unique model textures: "
		        << modelTextures.size() << '\n';

		const std::vector<PreviewDrawGroup> drawGroups =
		    BuildPreviewDrawGroups(loadedModel);

		std::cout << "Draw groups: " << drawGroups.size() << '\n';

		std::cout << "Loaded ROM texture 2: 32x64 RGBA16\n";

		std::cout
		        << (terrainPreview
		            ? "Loaded ROM terrain diagnostic "
		            : contourPreview
		            ? "Loaded ROM contour diagnostic "
		            : "Loaded ROM model ")
		        << modelId << ": "
		        << loadedModel.vertices.size() << " vertices, "
		        << loadedModel.triangles.size() << " triangles\n"
		        << "Preview center: ("
		        << framing.center[0] << ", "
		        << framing.center[1] << ", "
		        << framing.center[2] << "), scale="
		        << framing.scale << '\n';

		std::cout
		        << "Material batches: " << loadedModel.batches.size() << '\n';

		for (std::size_t i = 0; i < loadedModel.batches.size(); ++i) {
			const RomPreviewBatch& batch = loadedModel.batches[i];

			std::cout
			        << "  Batch " << i
			        << ": state=0x" << std::hex << batch.state << std::dec
			        << " firstTriangle=" << batch.firstTriangle
			        << " triangles=" << batch.triangleCount
			        << '\n';
			}

		const auto assetPath =
		    std::filesystem::absolute("host-assets");

		std::filesystem::create_directories(assetPath);
		std::ofstream(assetPath / "test.txt") << "Pilotwings host test\n";

		auto context =
		    Ship::Context::CreateInstance("Pilotwings 64", "pw64");

		context->Init();

		const nlohmann::json specification = {
				{
				"components", {
						{
							{"type", "Config"},
							{"initArgs", {{"path", "pw64-host.json"}}}
						},
						{
							{"type", "ConsoleVariable"}
						},
						{
							{"type", "ThreadPool"},
							{"initArgs", {{"threadCount", 1}}}
						},
						{
							{"type", "ResourceManager"},
							{
							"initArgs", {
									{"archivePaths", {assetPath.string()}}
								}
							}
						}
					}
				}
			};

		if (!Ship::Context::BuildComponentsFromJson(
		            context, specification)) {
			std::cerr << "Component initialization failed.\n";
			return 1;
			}

		auto resources = context->GetChildren()
		                 .GetFirst<Ship::ResourceManager>();

		if (!resources || !resources->IsInitialized()) {
			std::cerr << "Resource manager is not initialized.\n";
			return 1;
			}

		auto file = resources->LoadFileProcess("test.txt");

		if (!file) {
			std::cerr << "Could not load test.txt.\n";
			return 1;
			}

		std::cout << "LUS resource manager loaded test.txt successfully.\n";

		auto config =
		    context->GetChildren().GetFirst<Ship::Config>();

		auto consoleVariables =
		    context->GetChildren().GetFirst<Ship::ConsoleVariable>();

// Construct the window first; it can find the deck through Context later.
		auto window = std::make_shared<Fast::Fast3dWindow>(
		                  config, consoleVariables);

		auto controlDeck = std::make_shared<LUS::ControlDeck>(
		                       window, consoleVariables);

		const nlohmann::json presentationComponents = {
				{
				"components", {
						{{"type", "Console"}},
						{{"type", "Events"}},
						{{"type", "ControlDeck"}},
						{{"type", "Window"}},
						{{"type", "FileDrop"}}
					}
				}
			};

		const std::unordered_map<
		std::string, std::shared_ptr<Ship::Component>> overrides = {
				{"ControlDeck", controlDeck},
				{"Window", window}
			};

		if (!Ship::Context::BuildComponentsFromJson(
		            context,
		            presentationComponents,
		            nlohmann::json::object(),
		            overrides)) {
			std::cerr << "Window component initialization failed.\n";
			return 1;
			}

// The controller-specific Init needs an initialized window.
		uint8_t controllerBits = 0;
		controlDeck->Init(&controllerBits);

		uvControllerInit();

		OSContPad hostPads[MAXCONTROLLERS] {};
		unsigned int pressFrames = 0;
		unsigned int releaseFrames = 0;

		std::cout << "Window initialized using "
		          << window->GetWindowBackendName() << '\n';

		window->SetRendererUCode(ucode_f3dex2);

		float triangleAngle = 0.0f;

		TerrainPreviewCamera terrainCamera;

		if (terrainPreview) {
			std::cout
			        << "Terrain camera controls:\n"
			        << "  Stick: orbit and tilt\n"
			        << "  R: zoom in\n"
			        << "  Z: zoom out\n"
			        << "  Start: reset view\n";
			}

		Vec3F rectanglePosition{};
		rectanglePosition.x = 128.0f;
		rectanglePosition.y = 96.0f;
		auto previousUpdate = std::chrono::steady_clock::now();

		while (window->IsRunning()) {
			window->HandleEvents();

			if (!window->IsRunning()) {
				break;
				}

			if (!window->IsFrameReady()) {
				std::this_thread::sleep_for(
				    std::chrono::milliseconds(1));
				continue;
				}

			for (auto& pad : hostPads) {
				pad = {};
				}

			controlDeck->WriteToPad(hostPads);

			PW64_SetHostPad(
			    hostPads[0].button,
			    hostPads[0].stick_x,
			    hostPads[0].stick_y);

			if (!uvIOUpdate()) {
				std::cerr << "Pilotwings input update failed.\n";
				break;
				}

			const auto now = std::chrono::steady_clock::now();

			float dt =
			    std::chrono::duration<float>(
			        now - previousUpdate).count();

			previousUpdate = now;

			// Clamp before using dt for either animation or movement.
			dt = std::clamp(dt, 0.0f, 0.05f);

			if (terrainPreview) {
				auto cameraAxis = [](float value) {
					value = std::clamp(value, -1.0f, 1.0f);

					constexpr float deadzone = 0.15f;

					if (std::fabs(value) <= deadzone) {
						return 0.0f;
						}

					const float magnitude =
					    (std::fabs(value) - deadzone) /
					    (1.0f - deadzone);

					return std::copysign(magnitude, value);
					};

				const float horizontal =
				    cameraAxis(uvControllerGetStick(0, 0));

				const float vertical =
				    cameraAxis(uvControllerGetStick(0, 1));

				constexpr float orbitSpeed = 1.2f;
				constexpr float tiltSpeed = 0.8f;
				constexpr float zoomSpeed = 1.5f;
				constexpr float cameraTwoPi = 6.2831853f;

				terrainCamera.yaw +=
				    horizontal * orbitSpeed * dt;

				terrainCamera.yaw = std::remainder(
				                        terrainCamera.yaw, cameraTwoPi);

				terrainCamera.tilt = std::clamp(
				                         terrainCamera.tilt + vertical * tiltSpeed * dt,
				                         -1.35f,
				                         0.0f);

				const uint16_t buttons = hostPads[0].button;

				const float zoomDirection =
				    ((buttons & Z_TRIG) != 0 ? 1.0f : 0.0f) -
				    ((buttons & R_TRIG) != 0 ? 1.0f : 0.0f);

				terrainCamera.distance = std::clamp(
				                             terrainCamera.distance +
				                             zoomDirection * zoomSpeed * dt,
				                             2.75f,
				                             7.0f);

				if ((buttons & START_BUTTON) != 0) {
					terrainCamera = TerrainPreviewCamera{};
					}
				}

			constexpr float twoPi = 6.2831853f;
			constexpr float angularSpeed = 1.0f;

			triangleAngle += angularSpeed * dt;

			if (triangleAngle >= twoPi) {
				triangleAngle -= twoPi;
				}

			constexpr float speed = 100.0f;

			Vec3F input{};
			input.x = uvControllerGetStick(0, 0);
			input.y = -uvControllerGetStick(0, 1);

			if (uvVec3Len(&input) > 1.0f) {
				Vec3F normalized{};
				uvVec3Normal(&normalized, &input);
				uvVec3Copy(&input, &normalized);
				}

			Vec3F displacement{};
			uvVec3Mul(&displacement, &input, speed * dt);

			Vec3F nextPosition{};
			uvVec3Add(
			    &nextPosition,
			    &rectanglePosition,
			    &displacement);

			nextPosition.x =
			    std::clamp(nextPosition.x, 0.0f, 256.0f);
			nextPosition.y =
			    std::clamp(nextPosition.y, 0.0f, 192.0f);

			uvVec3Copy(&rectanglePosition, &nextPosition);

			Gfx* graphicsTest = BuildGraphicsTest(
			                        static_cast<int>(rectanglePosition.x),
			                        static_cast<int>(rectanglePosition.y),
			                        triangleAngle,
			                        loadedModel,
			                        framing,
			                        loadedTexture,
			                        modelTextures,
			                        drawGroups,
			                        partTransforms,
			                        terrainPreview,
			                        terrainCamera);

			window->DrawAndRunGraphicsCommands(
			    graphicsTest, {}, {});
			} // End while.

		return 0;
		} // End try.
	catch (const std::exception& error) {
		std::cerr << "Initialization error: "
		          << error.what() << '\n';
		return 1;
		} // End catch.
	} // End main.