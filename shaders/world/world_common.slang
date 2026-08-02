// Shared definitions for all ray-tracing stages. The generateShaderRecords task generates typed Java
// records and serializers from Slang reflection.
module world_common;

// Read-only device pointer with explicit std430 layout. The layout parameter is LOAD-BEARING: a bare
// Ptr<T> gets Slang's "natural" layout (no vec4 16-byte alignment → e.g. WorldPush.sunDir lands at 200
// instead of the Java writer's 208, silently reading garbage) and additionally requires the
// scalarBlockLayout device feature. Keep std430 unless RtDeviceBringup explicitly enables that feature;
// this alias is the single layout switch and generated Java serialization stays in lock-step.
public typealias ConstPtr<T> = Ptr<T, Access.Read, AddressSpace.Device, Std430DataLayout>;
public typealias DevicePtr<T> = Ptr<T, Access.ReadWrite, AddressSpace.Device, Std430DataLayout>;

// Push constant block. The hot inline lanes avoid dereferencing WorldPush for values that are read in
// hit shaders or used for raygen control flow — every 64-bit device address lives here for exactly that
// reason, rather than behind the worldPushAddr indirection. tableAddr/entityTableAddr/materialTableAddr
// are read in world.rahit/world.rchit, which never load WorldPush at all; the light-buffer addresses are
// only read in world.rgen, which loads WorldPush anyway, but keeping every Addr in one place means there
// is exactly one struct to check when adding a new device buffer, instead of two.
public struct WorldPushConstants {
    public uint64_t worldPushAddr;
    public uint64_t tableAddr;
    public uint64_t entityTableAddr;
    public uint64_t materialTableAddr;
    public uint64_t lightBufAddr;        // RIS emitter-NEE global light buffer (0 = none published)
    public uint64_t lightAliasAddr;      // power-weighted O(1) alias table (0 = uniform fallback)
    public uint64_t lightLocalAliasAddr; // power aliases relative to each section's range
    public uint64_t lightGridCellAddr;   // dense light grid cell headers (0 = global proposals only)
    public uint64_t lightGridSpanAddr;   // packed weighted section spans referenced by cell headers
    public uint64_t pathQueueAddr;       // packed primary -> indirect continuation records
    public uint64_t directReservoirAddr; // finalized per-pixel ReSTIR DI reservoirs
    public uint64_t pathReservoirAddr;   // current ReSTIR PT path-reservoir write slot (0 = disabled)
    public uint64_t pathReservoirPreviousAddr; // generation-matched history slot (0 = unavailable)
    public uint frameIndex;
    public uint debugView;
    public uint historyFlags;            // bits0-4 history/debug; bit5 mapping replay; bit6 cross-frame mapping replay
    public uint historyGeneration;       // low 24 bits are packed into path-sample compatibility metadata
};

// Block-breaking overlay entry: xyz = breaking block position (rebased, same space as gl_WorldRay*),
// w = bindless entityAlbedoTex slot of the destroy-stage texture.
public struct BreakEntry {
    public int4 posSlot;
};

// Per-frame world data, read through a 64-bit device address. WorldPushData.java is generated from this
// reflected type, including its serializer, byte size, vector/matrix types, and fixed-array capacities.
public struct WorldPush {
    public float4x4 invViewProj;
    public float3   camOffset;
    public uint     frameIndex;
    public float4x4 prevViewProj;
    public float3   camDelta;
    public uint     spp;
    public float2   jitter;
    public uint     flags;            // bit0 submerged, bit1 ReSTIR direct first-interface replacement, bit4 waves
    public uint     maxBounces;
    public float4   sunDir;           // xyz true sun direction, w dayFactor 0..1
    public float4   lightDir;         // xyz active NEE light dir, w square half-angle (rad)
    public float4   lightRadiance;    // xyz HDR light radiance, w star brightness
    public float4   moonDir;          // xyz moon direction, w moonPhase (0 full .. 4 new)
    public float4   celestial;        // xyz celestial rotation axis, w star angle (rad)
    public float4   sunUv;            // vanilla sun sprite atlas rect (u0,v0,u1,v1)
    public float4   moonUv;           // vanilla current-moon-phase sprite atlas rect
    public float4   waterParams;      // xyz camera-biome water tint, w wave time (s)
    public float4   waterAnchor;      // xy world-stable wave-domain anchor, z previous wave time
    public float4x4 curViewProj;      // forward camera-relative view-projection
    public uint     breakCount;
    public BreakEntry breaking[8];    // Java capacity and breakCount serialization are generated from this
    public float4   lightRebase;      // xyz hierarchy-to-current rebase, w inverse total light power
    public float4   lightGridOrigin;     // xyz minimum cell origin in rebased blocks, w cell size
    public int4     lightGridDims;       // xyz dense grid dimensions, w reserved
    public uint     lightCount;       // packed Light records in pc.lightBufAddr
    public uint     risCandidates;    // RIS candidate count M per diffuse vertex (0 = emitter NEE off)
    public uint     lightEpoch;       // low 32 bits of the atomically published light hierarchy generation
    // Opt-in view-20 readback addresses. They live in the per-frame BDA payload so the 120-byte
    // inline push-constant ABI stays below Vulkan's portable minimum limit; zero outside view 20.
    public uint64_t pathShiftDiagnosticCounterAddr;
    public uint64_t pathShiftDiagnosticPairAddr;
    public uint64_t pathShiftSourceRootAddr; // retained exact source root sidecar (0 outside view 20)
    public uint64_t pathShiftReceiverGuideAddr; // exact two-float4 material + Pass-B ray origin
    public uint64_t pathShiftGuideScratchAddr; // cleared full PathReservoir scratch, view 20 only
    public uint64_t pathShiftGuideRootScratchAddr; // matching cleared PathSourceRoot scratch
};

// 32-byte hot area-light record. Radiance uses packed R11G11B10 and the grid-relative owner section
// uses three unsigned 10-bit coordinates; the geometric normal is reconstructed from the half axes.
//
// 32 divides the 64-byte cache line, so a record never straddles one. At the previous 48 bytes about
// half of them did, costing two transactions apiece — and RIS fetches these at random indices, M per
// shading vertex, so that is the layout's whole purpose. float3 forces 16-byte struct alignment and 32
// is a multiple of it, so this lands with no tail padding.
//
// The rectangle area is derived rather than stored: the rect spans 2U x 2V, so area = 4*|U x V|, which
// is exactly what the collector used to write (halfU = 0.5*(aHi-aLo)*e01 and rectArea =
// |e01 x e03|*(aHi-aLo)*(bHi-bLo)). lightGeometricNormal already needs that cross product, so the
// freed lane costs one length() the compiler shares with the normalize.
public struct Light {
    public float3 pos;      // rebased world centre, f32 (the RIS target divides by distance to it)
    public uint   le;       // bitcast R11G11B10 radiance
    public uint   halfUxy;  // half2(U.x, U.y)
    public uint   halfUzVx; // half2(U.z, V.x)
    public uint   halfVyz;  // half2(V.y, V.z)
    public uint   section;  // 10:10:10 grid-relative section coord + bit 30 normal-flip
};

// Eight-byte Vose alias-table column. PDFs are reconstructed from the selected Light.
public struct LightAlias {
    public uint  aliasIndex;
    public float accept;
};

// Dense light grid cell entry. Empty coordinates have spanCount=0; populated section cells point into
// an immutable list of weighted section light ranges built and published alongside this grid.
public struct LightGridCell {
    public uint spanOffset;
    public uint spanCount;
    public float invWeightSum;
};

// One 16-byte Vose alias column for a cell's section distribution. The two light counts are uint16s.
public struct LightGridSpan {
    public uint firstLight;
    public uint aliasFirstLight;
    public uint packedLightCounts; // low16 self, high16 alias
    public float accept;
};

// Packed continuation record shared by the transient queue and the view-20 retained-source sidecar.
// Keeping this binding-free ABI in world_common lets the build-time layout probe reflect it without
// importing world_core's live descriptor bindings. The final field is a uint, so Std430DataLayout
// gives the record an exact 48-byte stride.
public struct PackedPathSegment {
    public float3 ro;
    public uint rd;
    public uint throughput;
    public uint currentExtinction;
    public uint outerExtinction;
    public uint mediumIors;
    public uint rayCone;
    public uint seed;
    public uint pathFlags;
    public uint nextRecord;
};

// View-20-only retained replay root for a selected one-hop spatial mapping. The two packed
// continuation records preserve the exact transient queue inputs that produced the canonical
// source path. Their origins are stored camera-relative so the record can later be rebased without
// depending on the transient queue or on the terrain rebase chosen for the capture frame.
//
// Source and receiver guide lanes are retained separately. Cross-frame diagnostics independently
// reproject both roots, reject moving/changed surfaces, then replay the original source. This
// sidecar is diagnostic-only and is not part of PathReservoir or the production estimator.
public struct PathSourceRoot {
    public float4 sourcePositionMaterial;
    public float4 sourceNormalRoughness;
    public float4 receiverPositionMaterial;
    public float4 receiverNormalRoughness;
    public PackedPathSegment segment0;
    public PackedPathSegment segment1;
};

// One sample may carry the ordinary continuation plus one branch split at the primary dielectric.
// This is shared by queue construction and the persistent random-replay contract.
public static const uint MAX_PATH_SEGMENTS = 2u;

// Clarity-first persistent ReSTIR DI record. This deliberately stays unpacked until captures prove
// reservoir bandwidth or residency is a real bottleneck. Five aligned lanes make the semantic contract
// explicit and give later GRIS work a stable seam without reusing the transient megakernel Reservoir.
//
// weights = {accumulated weight, effective candidate count M, final selected-sample W, confidence}
// metadata = {stable light id, light/content epoch, source frame, low16 age | high16 flags}
public struct DirectReservoir {
    public float4 samplePositionArea;   // xyz sample in light-hierarchy-local space, w emitter area
    public float4 lightNormalTargetPdf; // xyz emitter normal, w selected target density
    public float4 radianceProposalPdf;  // xyz emitted radiance, w source proposal density
    public float4 weights;
    public uint4 metadata;
};

// Clarity-first ReSTIR PT path-reservoir contract. The record is intentionally independent from the
// transient 48-byte continuation queue: one path sample may outlive a queue record and must retain the
// unshadowed contribution, density terms, shift footprint, and replay identity needed by a future GRIS
// reconnection pass. Eleven aligned lanes keep the first GPU substrate explicit while the estimator is still
// being validated against RtPathReservoirReference.
//
// sampleRadianceTarget = {unshadowed RGB contribution, target density}
// proposalShift = {bootstrap proposal density, shift Jacobian, path footprint, replay RNG state as float bits}
// proposalComponents = {light-selection PDF product, canonical continuation PDF product,
//                       canonical roulette PDF product, packed event counters + endpoint-valid bit}
// weights = {accumulated weight, effective M, final W, confidence}
// identity = {sample id, path/queue key, topology key, packed generation/depth/transport class}
// replay = {segment0 path seed, segment0 light-proposal seed,
//           segment1 path seed, segment1 light-proposal seed}
// replayControl = {replay ABI version, segment count, terminal path-state hash,
//                  terminal light-proposal-state hash}
// canonicalRadianceTarget = {one replayable sky/emissive endpoint RGB, endpoint target density}
// reconnectionVertex = {chosen path vertex position in current rebased world space,
//                       packed valid/depth/first-edge-event metadata}
// reconnectionNormalPdf = {oriented second-hit normal, continuous source-edge proposal density}
// reconnectionThroughput = {selected local first-edge RGB throughput,
//                           packed identity/diffuse-reconnection mapping control}
public struct PathReservoir {
    public float4 sampleRadianceTarget;
    public float4 proposalShift;
    public float4 proposalComponents;
    public float4 canonicalRadianceTarget;
    public float4 weights;
    public uint4 identity;
    public uint4 replay;
    public uint4 replayControl;
    public float4 reconnectionVertex;
    public float4 reconnectionNormalPdf;
    public float4 reconnectionThroughput;
};

// TerrainPrim.flags bit 0: this emissive quad is in the light buffer (RtLightCollector membership).
public static const uint TERRAIN_PRIM_IN_LIGHT_BUFFER = 1u;

// Largest finite half. The payload's half3 lanes carry HDR values (sky radiance most of all), so
// producers clamp to this instead of letting the conversion round up to +inf.
public static const float HALF_MAX = 65504.0;

// Radiance-ray payload (location 0). Member order and types are a cross-stage ABI.
//
// The three-component lanes are half3: they cost 6 bytes instead of 12, and payload storage is reserved
// per trace call site, so every byte here is paid twice per radiance trace (once for TraceRay, once for
// Invoke) and is preserved across the SER reorder. hitT stays f32 — it reaches 10000 blocks and the hit
// position is reconstructed from it, where half's ~4-block spacing at that magnitude would be visible.
public struct Payload {
    // Sky radiance on a miss, so this carries HDR values well above 1; half tops out at 65504.
    public half3  albedo;      // hit: block albedo. miss: sky radiance.
    public half3  normal;      // hit: geometric normal, viewer-oriented.
    public float  hitT;        // >= 0 on hit, < 0 on miss.
    public half3  motionPrev;  // per-vertex world displacement since last frame.
    public half3  f0;          // specular F0.
    public uint   flags;       // bits 0..7 hit flags; bits 8..31 stable registry material key.
    public uint   roughMetal;  // packHalf2x16(roughness, metalness)
    public uint   emissionSss; // packHalf2x16(emission, sss)
    public uint   iorTransmission; // packHalf2x16(IOR, transmission factor)
    public uint   rayCone;     // packHalf2x16(width, spread)
};

public static const uint PAYLOAD_SHOW_CELESTIAL = 4u;
// Set by world.rchit on a terrain hit whose prim carries TERRAIN_PRIM_IN_LIGHT_BUFFER: this emitter is
// RIS-sampled, so raygen gates its direct-hit emission on diffuse continuation rays (no double count).
public static const uint PAYLOAD_EMITTER_IN_LIST = 128u;
// Set by world.rchit on any dielectric hit (water or glass/ice): true when the incoming ray travels
// against the prim's outward face normal (entering the volume), false when it exits. Derived from face
// orientation rather than toggled, so a stray or missing face cannot corrupt the medium for the rest of
// the path — each crossing re-derives which side it is on. Bit 2 (PAYLOAD_SHOW_CELESTIAL) is only
// meaningful on a miss and this bit only on a hit, so they never need to be read together, but they get
// distinct bits anyway to keep the two concerns unambiguous.
public static const uint PAYLOAD_DIELECTRIC_ENTERING = 8u;
public static const uint PAYLOAD_EMISSION_SOURCE_SHIFT = 4u;
public static const uint PAYLOAD_EMISSION_SOURCE_MASK = 7u;
public static const uint EMISSION_SOURCE_NONE = 0u;
public static const uint EMISSION_SOURCE_UNIFORM = 1u;
public static const uint EMISSION_SOURCE_HEURISTIC = 2u;
public static const uint EMISSION_SOURCE_LAB_PBR = 3u;
public static const uint PAYLOAD_MATERIAL_KEY_SHIFT = 8u;
public static const uint PAYLOAD_MATERIAL_KEY_MASK = 0x00FFFFFFu;

public static const uint PAYLOAD_MATERIAL_MASK = 3u;
public static const uint MATERIAL_OPAQUE = 0u;
// Both dielectrics are volumes: they refract and push a participating medium. They stay distinct ids
// because water owns behaviour nothing else has — an animated wave surface, the caustic term, biome-tint
// absorption calibrated per block of depth, and a chit path fed by the fluid mesher rather than the
// (inset) translucent terrain layer. MATERIAL_DIELECTRIC is everything else transparent: glass, ice, and
// any solid transparent block a pack defines.
public static const uint MATERIAL_WATER = 1u;
public static const uint MATERIAL_PARTICLE = 2u;
public static const uint MATERIAL_DIELECTRIC = 3u;
// gRestirPositionMaterial.w is R32F and therefore preserves every integer through 24 bits. Keep the
// 2-bit transport material model in the low bits and a stable 22-bit registry identity above it.
public static const uint RECEIVER_MATERIAL_KEY_MASK = 0x003FFFFFu;
public static const uint RECEIVER_PARTICLE_MATERIAL_KEY = RECEIVER_MATERIAL_KEY_MASK;

public uint packReceiverMaterialIdentity(uint material, uint materialKey) {
    return ((materialKey & RECEIVER_MATERIAL_KEY_MASK) << 2u)
            | (material & PAYLOAD_MATERIAL_MASK);
}

public uint receiverMaterialModel(uint identity) {
    return identity & PAYLOAD_MATERIAL_MASK;
}

public uint receiverMaterialKey(uint identity) {
    return identity >> 2u;
}

// Entity per-triangle record (48 B). The final lane mirrors TerrainPrim's integer material metadata.
public struct Prim {
    public float4 normal; // xyz = geometric normal, w = per-primitive emission strength
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0;
    public uint aux1;
};

// Terrain per-triangle record (48 B). The final 16-byte lane contains integer geometry-local metadata;
// shared surface parameters live once in MaterialHeader instead of being copied into every triangle.
public struct TerrainPrim {
    public float4 normal;
    public float4 tint;
    public uint materialId;
    public uint flags;
    public uint aux0;
    public uint aux1;
};

// Compiled material record (80 B). Source-format data has already been decoded into canonical pages;
// texturePage indexes the dedicated canonical page arrays. materialUv maps sprite-local UV into the packed
// page, while albedoUv maps the vanilla atlas rectangle back to sprite-local UV.
public struct MaterialHeader {
    public uint model;
    public uint features;        // low feature bits; bits 24..31 = maximum valid material-page LOD
    public uint texturePage;
    public uint extensionOffset; // zero when no optional extension record is present
    public float4 materialUv; // packed-page uvMin.xy + uvExtent.xy
    public float4 albedoUv;   // vanilla atlas uvMin.xy + inverse uvExtent.xy
    public float4 params;  // roughness, metalness, IOR, transmission
    public float4 average; // whole-sprite linear-input average RGBA for translucent shadow filtering
};

public static const uint MATERIAL_FEATURE_SPEC = 1u;
public static const uint MATERIAL_FEATURE_NORMAL = 2u;
public static const uint MATERIAL_FEATURE_HEURISTIC_EMISSION = 4u;
public static const uint MATERIAL_FEATURE_STOCHASTIC_ALPHA = 16u;
// Final HDR emission strength (EMISSIVE_STRENGTH baseline * any JSON override multiplier), baked in Java
// at material-compile time (RtMaterialRegistry) and packed here as a 16-bit fraction of the max — every
// emissive material carries a value (0 for non-emissive), not just resource-pack-overridden ones, so the
// full features word's spare 16 bits (8..23) go to it instead of the 8 bits the old override-only field used.
public static const uint MATERIAL_EMISSION_STRENGTH_SHIFT = 8u;
public static const uint MATERIAL_EMISSION_STRENGTH_MASK = 65535u;
public static const float MATERIAL_MAX_EMISSION_STRENGTH = 32.0;
public static const uint MATERIAL_MAX_LOD_SHIFT = 24u;
public static const uint MATERIAL_MAX_LOD_MASK = 255u;

// Per-section geometry record: packed buffer addresses + per-bucket triangle offsets
// (gl_PrimitiveID restarts at 0 per geometry, so hit shaders re-add triBase[geometryIndex]).
public struct Section {
    public uint64_t primAddr;
    public uint64_t uvAddr;
    public uint triBase[4]; // fixed bucket order: solid/cutout/translucent/water
};

// Entity geometry record: packed buffer addresses, motion, and per-geometry triangle bases. Entity BLAS
// geometry order is opaque/any-hit; PrimitiveIndex restarts at zero for each geometry.
public struct EntityGeom {
    public uint64_t primAddr;
    public uint64_t idxAddr;
    public uint64_t uvAddr;
    public uint64_t dispAddr;
    public float4 rigidDisp;
    public uint triBase[2];
    public uint2 reserved;
};

// TLAS instance-custom-index encoding + terrain geometry buckets (shared by rahit/rchit).
public static const uint ENTITY_BIT = 0x800000u;
public static const uint PARTICLE_BIT = 0x400000u; // particles share the entity cutout path
public static const uint IDX_MASK = 0x3FFFFFu;     // low 22 bits = geom-table index
public static const uint BUCKET_CUTOUT = 1u;
public static const uint BUCKET_TRANSLUCENT = 2u;
public static const uint BUCKET_WATER = 3u;

// packHalf2x16 equivalent (Slang core module has no GLSL pack builtin; f32tof16 returns the half
// bit-pattern in the low 16 bits).
public uint packHalf2(float2 v) {
    return (f32tof16(v.y) << 16) | (f32tof16(v.x) & 0xFFFFu);
}

public float2 unpackHalf2(uint p) {
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}
