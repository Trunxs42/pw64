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
#include "rom_texture.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <array>
#include <vector>

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

static std::vector<Mtx4F> BuildPreviewPartTransforms(
    const RomPreviewModel& asset) {

	if (asset.parts.empty() ||
	        asset.parts.size() != asset.matrices.size() ||
	        asset.parts.size() > 8) {
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

static Gfx* BuildGraphicsTest(
    int x, int y, float angle,
    const RomPreviewModel& asset,
    const PreviewFraming& framing,
    const RomPreviewTexture2& texture,
    const RomPreviewTexture2* modelTexture,
    const std::vector<PreviewDrawGroup>& drawGroups,
    const std::vector<Mtx4F>& partTransforms) {
	static Gfx commands[4096] {};
	alignas(8) static uint16_t colorBuffer[320 * 240] {};
	alignas(8) static uint16_t depthBuffer[320 * 240] {};

	// Persistent packed matrices used by the display list.
	alignas(8) static uint32_t projectionMatrix[16] {};
	alignas(8) static uint32_t partMatrices[8][16] {};

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

// Position the triangle in front of the camera.
	uvMat4SetIdentity(&model);
	uvMat4LocalTranslate(&model, 0.0f, 0.0f, -4.0f);

// Rotate around two axes so its 3D orientation is visible.
	uvMat4RotateAxis(&model, angle, 'x');
	uvMat4RotateAxis(&model, angle, 'y');

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
	        partTransforms.empty() || partTransforms.size() > 8) {
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

	// Preserve the controllable orange rectangle.
	const uint32_t rectangle =
	    GPACK_RGBA5551(255, 160, 32, 1);

	gDPSetFillColor(
	    p++, (rectangle << 16) | rectangle);
	gDPFillRectangle(p++, x, y, x + 63, y + 47);

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

		if (translucent) {
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
			if ((textureId != 86u && textureId != 447u) ||
			        modelTexture == nullptr) {
				throw std::runtime_error(
				    "Missing or unsupported batch texture.");
				}

			gDPSetTexturePersp(p++, G_TP_PERSP);
			gDPSetTextureLOD(p++, G_TL_TILE);
			gDPSetTextureDetail(p++, G_TD_CLAMP);
			gDPSetTextureLUT(p++, G_TT_NONE);
			gDPSetTextureFilter(p++, G_TF_BILERP);
			gDPSetTextureConvert(p++, G_TC_FILT);

			if (textureId == 86u) {
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
			else {
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

// Draw a 2x texture preview in the upper-left corner.
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

	gDPSetCombineMode(p++, G_CC_DECALRGBA, G_CC_DECALRGBA);

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

// Rectangle coordinates use two fractional bits.
// Texture derivatives use ten fractional bits:
// 512 means half a texel per screen pixel, giving 2x enlargement.
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

	gDPPipeSync(p++);

// Do not leave perspective correction disabled for later frames.
	gDPSetTexturePersp(p++, G_TP_PERSP);

	gDPFullSync(p++);
	gSPEndDisplayList(p++);

	return commands;
	}

int main(int argc, char* argv[]) {
	try {
		if (argc > 2) {
			std::cerr << "Usage: " << argv[0] << " [model-id]\n";
			return 1;
			}

		std::size_t modelId = 2;

		if (argc == 2) {
			const std::string_view argument(argv[1]);

			if (argument.empty()) {
				throw std::runtime_error(
				    "Model ID must be a non-negative decimal integer.");
				}

			const auto parsed = std::from_chars(
			                        argument.data(),
			                        argument.data() + argument.size(),
			                        modelId,
			                        10);

			if (parsed.ec != std::errc{} ||
			        parsed.ptr != argument.data() + argument.size()) {
				throw std::runtime_error(
				    "Model ID must be a non-negative decimal integer.");
				}
			}

		const RomPreviewModel loadedModel =
		    LoadRomPreviewModel("baserom.us.z64", modelId);

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

		bool hasModelTexture = false;
		uint32_t loadedModelTextureId = 0xFFFu;
		RomPreviewTexture2 modelTexture{};

		for (const RomPreviewBatch& batch : loadedModel.batches) {
			const uint32_t textureId = batch.state & 0xFFFu;

			if (textureId == 0xFFFu) {
				continue;
				}

			if (textureId != 86u && textureId != 447u) {
				throw std::runtime_error(
				    "A batch references an unsupported preview texture.");
				}

			if (hasModelTexture) {
				if (textureId != loadedModelTextureId) {
					throw std::runtime_error(
					    "This preview still supports one unique model texture.");
					}

				continue;
				}

			modelTexture =
			    LoadRomPreviewTextureBytes("baserom.us.z64", textureId);

			loadedModelTextureId = textureId;
			hasModelTexture = true;

			std::cout
			        << "Loaded model texture " << textureId << '\n';
			}

		const std::vector<PreviewDrawGroup> drawGroups =
		    BuildPreviewDrawGroups(loadedModel);

		std::cout << "Draw groups: " << drawGroups.size() << '\n';

		std::cout << "Loaded ROM texture 2: 32x64 RGBA16\n";

		std::cout
		        << "Loaded ROM model " << modelId << ": "
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
			                        hasModelTexture ? &modelTexture : nullptr,
			                        drawGroups,
			                        partTransforms);

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