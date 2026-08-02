// Indirect pass (pass B of the wavefront split — see docs/WAVEFRONT_PLAN.md).
//
// Resumes the continuations the primary pass queued and runs the bounce loop: NEE toward the sun, RIS
// emitter lighting, thin-surface SSS, BSDF continuation and Russian roulette. It captures no guides and
// owns no motion vectors — guides and primary are deliberately absent from the include list
// below, so guide state cannot leak into this pass by accident.
//
// Lighting is iterative here rather than recursive from closest-hit, so maxRayRecursionDepth stays 1.
// At each surface vertex:
//   * NEE toward the sun — a single analytic directional light, gated by one hit-object visibility
//     query (TerminateOnFirstHit | SkipClosestHit). Lambertian BRDF = albedo/PI.
//   * a cosine-weighted Lambertian continuation ray (throughput *= albedo; the PI and cos cancel).
// Sky / ambient light needs no separate AO pass: continuation rays that escape land in world.rmiss,
// which returns sky radiance — so soft sky fill and contact AO both fall out of the path integral.
//
// Each frame emits a single SPP-averaged (still noisy) estimate; temporal convergence is the denoiser's
// job. With RR disabled the output is the raw noisy path trace — a reference view, not a converged
// image. Output is HDR (R16G16B16A16_SFLOAT); the HDR -> LDR tonemap happens at the display.comp seam.
//
// The build emits an ordinary TraceRay fallback and an optional EXT SER variant from this source.
import world_common;
import world_core;
import math;
import sampling_core;
import medium;
import segment;
import water;
import trace;
#ifdef CAUSTICA_ENABLE_EXT_SER
import trace_ser;
#endif
import lighting;
import direct_reservoir_common;
import path_reservoir_common;

static void writeExactReceiverRayOrigin(PathSegment segment, uint2 pix,
                                        int hitDepth, float3 rayOrigin) {
    // View 20 alone supplies this address. Only the unsplit camera segment owns the deterministic
    // receiver; retained-source/mapping replay dispatches must never overwrite the current sidecar.
    if (worldPush.pathShiftReceiverGuideAddr == 0u
            || (pc.historyFlags & PATH_MAPPING_REPLAY_PASS_FLAG) != 0u
            || (pc.historyFlags & PATH_GUIDE_PREVIOUS_REPLAY_PASS_FLAG) != 0u
            || segment.bounce != 0 || hitDepth != 0) {
        return;
    }
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pix.y * dimensions.x + pix.x;
    DevicePtr<float4>(worldPush.pathShiftReceiverGuideAddr)[pixelIndex * 2u + 1u] =
            float4(rayOrigin, 1.0);
}

public PathTraceResult tracePathSeeded(PathSegment seg, uint2 pix, bool captureMetadata,
                                       uint replaySeed, uint proposalReplaySeed) {
    float3 L = float3(0.0, 0.0, 0.0);
    float3 ro = seg.ro;
    float3 rd = seg.rd;
    float3 throughput = seg.throughput;
    uint seed = replaySeed;
    // Topology must describe transport structure, not the random seed. Temporal admission compares
    // this key across frames; seed/sample identity lives separately in sampleId and replayState.
    uint topologyKey = captureMetadata ? pathHash(0x91E10DA5u ^ uint(seg.bounce)) : 0u;
    uint transportMask = 0u;
    float rayConeWidth = seg.rayConeWidth;
    float rayConeSpread = max(seg.rayConeSpread, RAY_CONE_MIN_SPREAD);
    // Capture selected light proposal PDFs independently from the bootstrap estimator. This product
    // is intentionally incomplete until BSDF/technique PDFs are recorded for every continuation event.
    float lightProposalPdf = 1.0;
    uint lightProposalEvents = 0u;
    float continuationProposalPdf = 1.0;
    uint continuationProposalEvents = 0u;
    float rouletteProposalPdf = 1.0;
    uint rouletteProposalEvents = 0u;
    float3 canonicalRadiance = float3(0.0);
    bool canonicalValid = false;
    float canonicalContinuationProposalPdf = 1.0;
    uint canonicalContinuationProposalEvents = 0u;
    float canonicalRouletteProposalPdf = 1.0;
    uint canonicalRouletteProposalEvents = 0u;
    float3 reconnectionVertex = float3(0.0);
    bool reconnectionValid = false;
    uint reconnectionDepth = 0u;
    float3 reconnectionNormal = float3(0.0);
    // Full continuous proposal density for the hit-0 -> hit-1 edge, including lobe selection.
    // Delta reflection/refraction use a discrete measure and intentionally leave this at zero.
    float reconnectionSourcePdf = 0.0;
    uint reconnectionEventKind = PATH_RECONNECTION_EVENT_NONE;
    float3 reconnectionThroughput = float3(0.0);
    // A second-hit reconnection cannot shift an endpoint that was already gathered at hit 0.
    // View 20 therefore captures only endpoints downstream of the first selected edge. Other
    // diagnostics retain the general canonical contract, including primary emissive endpoints.
    bool requireReconnectionEndpoint = pc.debugView == PATH_SHIFTED_RADIANCE_DEBUG_VIEW;
    int maxBounces = int(worldPush.maxBounces);
    int rrStart = maxBounces <= 3 ? 1 : 2;
    // RIS emitter NEE: direct lighting from block emitters is active when lights are published and the
    // candidate count is non-zero. Off => emitters are gathered only on path hits (legacy behaviour).
    bool risOn = worldPush.risCandidates > 0u && worldPush.lightCount > 0u && pc.lightBufAddr != 0;
    // Independent light-proposal RNG. Screen-space coherent selection is intentionally deferred until
    // there is a real presampled RIS tile buffer; direct sharing made tile-shaped lighting noise visible.
    // Measured: making a 16x16 tile share this seed recovers at most 1.3ms, so a future pool should share
    // the POOL rather than the seed — the win is in removing dependent loads, not in coherence.
    uint proposalSeed = proposalReplaySeed;
    // The medium the segment starts in. For the camera segment that is water when the eye is submerged
    // (so the first ray already carries the right relative index and absorption); for a segment split off
    // a dielectric it is whatever the parent was travelling through.
    MediumStack medium = seg.medium;
    bool waterWaves = (worldPush.flags & 16u) != 0u; // animated wave-normal perturbation
    // Sky-disc gate (see world.rmiss): the sun/moon disc is shown to the primary ray and to rays spawned
    // by a specular/dielectric bounce (mirror reflections + refractions of the sun/moon), but hidden from
    // diffuse-sampled continuation rays — those are already covered by the sun NEE, so the tiny disc
    // would be a double-counted firefly. Reset true on every specular/water bounce, false on a diffuse one.
    bool showCelestial = seg.showCelestial;
    // Pass B treats every hit it sees as one indirect-depth step, regardless of material or selected
    // continuation lobe. Deliberately start at zero instead of seg.bounce: the first hit of either
    // queued split branch is still the primary-terminal shading point. Hits 0 and 1 retain SSS; RIS
    // remains active with its full configured candidate count at every hit. Pass A's primary/interface
    // prefix is outside this SSS quality budget.
    int indirectDepth = 0;
    for (int bounce = seg.bounce; bounce <= maxBounces; bounce++) {
        // Radiance SBT records run any-hit only for true alpha cutout. Translucent/water go straight to
        // closest-hit for dielectric handling. Geometry is double-sided; the chit flips the normal.
        // Primary (bounce 0) is the camera ray (CULL_PRIMARY): sees particles but not the first-person
        // player. Bounce rays are secondary (CULL_SECONDARY): exclude particles, include the player.
#ifdef CAUSTICA_ENABLE_EXT_SER
        // SER lifetime phase: keep paths that are not roulette-eligible, paths that may terminate via
        // roulette, and paths guaranteed to end at the bounce cap in separate coherence groups.
        uint pathPhaseHint = bounce >= maxBounces ? 2u : (bounce >= rrStart ? 1u : 0u);
        traceRadianceReordered(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                               showCelestial, rayConeWidth, rayConeSpread, pathPhaseHint);
#else
        traceRadiance(bounce == 0 ? CULL_PRIMARY : CULL_SECONDARY, ro, 0.0, rd, 10000.0,
                      showCelestial, rayConeWidth, rayConeSpread);
#endif

        if (payload.hitT < 0.0) {
            // world.rmiss writes sky radiance (gradient + sun/moon disc + stars) into payload.albedo. The
            // packed show-celestial flag decides whether this path segment may see the bright disc.
            float3 sky = payload.albedo;
            L += throughput * sky; // escaped to sky
            if (!canonicalValid && (!requireReconnectionEndpoint || indirectDepth > 1)) {
                canonicalRadiance = throughput * sky;
                canonicalValid = true;
            }
            if (captureMetadata) {
                topologyKey = pathTopologyStep(topologyKey, 0xF001u);
            }
            break;
        }

        // Beer–Lambert: attenuate along the segment just travelled by the medium it lay inside. Applies
        // to every hit reached while inside a volume dielectric (its own exit face, or whatever content
        // lies within it), shifting the transmitted radiance with distance. Air's extinction is zero, so
        // the ordinary case costs one comparison.
        if (any(medium.current.extinction > 0.0)) {
            throughput *= exp(-medium.current.extinction * payload.hitT);
        }

        float3 n = payload.normal; // oriented toward the incoming ray by the closest-hit
        float3 hitPos = ro + rd * payload.hitT;
        rayConeWidth = max(rayConeWidth + rayConeSpread * max(payload.hitT, 0.0), RAY_CONE_MIN_WIDTH);
        uint material = payloadMaterial();
        int hitDepth = indirectDepth++;
        if (captureMetadata && !reconnectionValid && hitDepth == 1) {
            reconnectionVertex = hitPos;
            // payload.normal is the viewer-oriented geometric normal. Capture it before water-wave
            // shading perturbation so the future solid-angle Jacobian uses geometric measure.
            reconnectionNormal = n;
            reconnectionValid = true;
            reconnectionDepth = uint(hitDepth);
        }
        if (captureMetadata) {
            topologyKey = pathTopologyStep(topologyKey,
                    (material & 0xFFu) | ((uint(hitDepth) & 0xFFu) << 8u));
            topologyKey = pathTopologyStep(topologyKey, payloadMaterialKey());
        }
        // The current frame's direct candidates are generated after this path trace because they need
        // its receiver guides. Never inspect directReservoirAddr here: before candidate generation it
        // contains stale slot data, so camera motion made this gate randomly alternate between inline
        // RIS and the later direct resolve. Enabling the replacement owns hit 0 unconditionally; an
        // empty current candidate simply contributes zero in the resolve, exactly like an empty RIS draw.
        bool restirAtFirstInterface = (worldPush.flags & 2u) != 0u && hitDepth == 0
                && pc.directReservoirAddr != 0;
        bool risAtVertex = risOn && !restirAtFirstInterface;

        // ---- Dielectric interface: water, stained glass, ice. ONE handler and one behaviour — every
        // dielectric is a volume. Fresnel from the relative index across the face, then reflect or
        // refract; no diffuse response and no sun NEE, because the surface is specular. Transmission
        // pushes or pops a participating medium, and the segment travelled inside it is attenuated by
        // that medium's extinction rather than by a tint multiply at the interface.
        //
        // Water is the only material that adds anything: the animated wave normal, and a medium marked
        // for the caustic term. Which side of the face we are on comes from the geometry
        // (payloadDielectricEntering), so a stray or missing face cannot desync the medium for the rest
        // of the path.
        if (material == MATERIAL_WATER || material == MATERIAL_DIELECTRIC) {
            // A dielectric has no local shading term. At the path cap its continuation cannot be traced,
            // so avoid computing an unused Fresnel choice, medium update, and roulette step.
            if (bounce >= maxBounces) {
                break;
            }
            bool isWater = material == MATERIAL_WATER;
            bool entering = payloadDielectricEntering();
            float3 tint = payload.albedo;
            float transmission = clamp(payloadTransmission(), 0.0, 1.0);

            // Ripple the (near-flat) surface normal before any Fresnel/guide use → glints, broken
            // reflections, wobbling refraction. The wave domain is world-stable: rebased hit XZ plus the
            // pushed anchor (terrain origin mod 4096) survives the per-move terrain rebase.
            float3 geometricNormal = n; // unrippled; ray origins offset along this
            if (isWater && waterWaves) {
                float2 waterDomain = hitPos.xz + worldPush.waterAnchor.xy;
                float waterFootprint = rayConeWidth / max(abs(dot(-rd, geometricNormal)), 0.2);
                n = applyWaterWaves(geometricNormal, waterDomain,
                                    worldPush.waterParams.w, waterFootprint);
            }

            // The medium this face opens into, and the one it returns to on the way out — which is what
            // the stack remembers.
            Medium entered = makeDielectricMedium(isWater, max(payloadIor(), 1.0), tint, transmission);
            float etaI = medium.current.ior;
            float etaT = entering ? entered.ior : medium.outer.ior;

            float cosI = clamp(dot(-rd, n), 0.0, 1.0);
            float F = fresnelDielectric(cosI, etaI, etaT);
            float3 transmittedDir = refract(rd, n, etaI / etaT);
            // The translucent terrain layer is recessed by TRANSLUCENT_INSET (RtTerrainMesher), so a
            // glass/ice face touching a slab or stair sits a hair behind that neighbour's surface. A full
            // SURF_BIAS would restart the transmitted ray past the neighbour and see through it. Water
            // comes from the fluid mesher, which applies no inset, so it takes the ordinary bias.
            float transmitBias = isWater ? SURF_BIAS : INSET_TRANSMIT_BIAS;

            bool chooseReflection = rndf(seed) < F;
            if (chooseReflection) {
                pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                        max(F, 1.0e-20));
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalContinuationProposalPdf,
                            canonicalContinuationProposalEvents, max(F, 1.0e-20));
                }
                transportMask |= PATH_TRANSPORT_DELTA;
                if (captureMetadata && hitDepth == 0) {
                    reconnectionEventKind = PATH_RECONNECTION_EVENT_DELTA;
                    reconnectionThroughput = float3(1.0);
                }
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xD101u);
                rd = reflect(rd, n);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, SURF_BIAS);
            } else {
                if (dot(transmittedDir, transmittedDir) <= 0.0) break; // TIR with F < 1 cannot happen
                pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                        max(1.0 - F, 1.0e-20));
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalContinuationProposalPdf,
                            canonicalContinuationProposalEvents, max(1.0 - F, 1.0e-20));
                }
                transportMask |= PATH_TRANSPORT_TRANSMISSION;
                if (captureMetadata && hitDepth == 0) {
                    reconnectionEventKind = PATH_RECONNECTION_EVENT_TRANSMISSION;
                    reconnectionThroughput = float3(1.0);
                }
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xD102u);
                rd = normalize(transmittedDir);
                ro = offsetSurfaceOrigin(hitPos, geometricNormal, rd, transmitBias);
                // Crossing into or out of the volume. Absorption is the medium's job from here, so the
                // tint is NOT also multiplied into the throughput — that would double-count it.
                if (entering) {
                    mediumPush(medium, entered);
                } else {
                    mediumPop(medium);
                }
            }
            showCelestial = true; // specular interface: the continuation ray may see the sun/moon disc
            if (bounce >= rrStart) {
                float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
                bool survive = rndf(seed) <= q;
                pathAccumulateProposal(rouletteProposalPdf, rouletteProposalEvents,
                        survive ? q : 1.0 - q);
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalRouletteProposalPdf,
                            canonicalRouletteProposalEvents, survive ? q : 1.0 - q);
                }
                if (!survive) {
                    if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xE102u);
                    break;
                }
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xE101u);
                throughput /= q;
            }
            continue;
        }

        // Particles (material == 2): camera-facing receiver billboard. The instance mask keeps particles
        // off secondary rays, so this only fires at bounce 0; they receive direct light and one diffuse
        // path continuation into the scene, but never appear in shadows/reflections/GI or emit radiance.
        if (material == MATERIAL_PARTICLE) {
            float3 albedo = payload.albedo;

            float3 lightDir = worldPush.lightDir.xyz;
            float lightHalfAngle = worldPush.lightDir.w;
            if (lightHalfAngle > 0.0) {
                lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
            }
            // Billboards are effectively two-sided receivers. Bias from the side facing the light; shadow
            // rays exclude particles anyway, but this keeps the origin sane when the light is behind the
            // camera side.
            float signedNdl = dot(n, lightDir);
            float ndl = abs(signedNdl);
            if (ndl > 0.0) {
                float3 shadowOrigin = hitPos + (signedNdl >= 0.0 ? n : -n) * SURF_BIAS;
                float3 vis = visibility(shadowOrigin, lightDir, 10000.0).transmittance;
                if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                    L += throughput * albedo * INV_PI * worldPush.lightRadiance.xyz * ndl * vis;
                }
            }

            // RIS direct lighting from block emitters (lava, glowstone, torches, ...) — same reservoir
            // sampler as terrain/entities, but two-sided: a billboard has no back face, so light
            // striking either side of the quad should still land (matches the sun/moon NEE above).
            if (risAtVertex) {
                float3 v = -rd;
                Reservoir r = risInitial(hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                                         true, 0.0, seed, proposalSeed);
                if (r.proposalPdf > 0.0) {
                    pathAccumulateProposal(lightProposalPdf, lightProposalEvents, r.proposalPdf);
                }
                L += throughput * shadeReservoir(r, hitPos, n, v, rd, albedo, float3(0.0, 0.0, 0.0), 1.0,
                                                 true, 0.0);
            }

            ro = hitPos + n * SURF_BIAS;
            writeExactReceiverRayOrigin(seg, pix, hitDepth, ro);
            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            rd = cosineDir(n, seed);
            float directionalPdf = max(abs(dot(n, rd)) * INV_PI, 1.0e-20);
            pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                    directionalPdf);
            if (!canonicalValid) {
                pathAccumulateProposal(canonicalContinuationProposalPdf,
                        canonicalContinuationProposalEvents,
                        directionalPdf);
            }
            if (captureMetadata && hitDepth == 0) {
                reconnectionSourcePdf = directionalPdf;
                reconnectionEventKind = PATH_RECONNECTION_EVENT_DIFFUSE;
                reconnectionThroughput = albedo;
            }
            transportMask |= PATH_TRANSPORT_DIFFUSE;
            if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xC101u);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse receiver: direct sun/moon was handled by NEE above
            continue;
        }

        float3 albedo = payload.albedo;
        float sss = payloadSss(); // LabPBR SSS strength (0 when absent)
        float3 p = hitPos + n * SURF_BIAS;
        writeExactReceiverRayOrigin(seg, pix, hitDepth, p);
        float3 v = -rd; // view direction (toward the camera / incoming ray)

        // Material split. Metals tint F0 by albedo and have no diffuse lobe; dielectrics use a fixed
        // 0.04 F0, sourced from the chit (LabPBR custom/metal F0, or the dielectric default it applies).
        float rough = clamp(payloadRoughness(), 0.0, 1.0);
        float metal = clamp(payloadMetalness(), 0.0, 1.0);
        // Unresolvably tight lobes become exact delta mirrors; everything else keeps its authored value.
        // Snapping to 0 also propagates into the guide buffers, which is what tells RR to hold the
        // reflection sharp instead of accumulating it as a glossy lobe.
        bool exactSpecular = isDeltaAlpha(rough);
        rough = exactSpecular ? 0.0 : rough;
        float3 diffAlb = albedo * (1.0 - metal);
        float3 F0 = payload.f0;

        // Emissive surfaces (lava, glowstone, torches, ...) add radiance directly, colored by albedo.
        // RIS emitter NEE: the direct-hit emission term is gated only for emitters RIS actually samples
        // (payload emitter-in-list bit) — and then only off diffuse continuation rays (showCelestial
        // false), which the previous vertex's RIS already accounted for; the primary ray and
        // specular/dielectric bounces (showCelestial true) still add it, since no emitter NEE happened
        // along those. Emitters NOT in the light buffer (sparse/sub-threshold footprints, entities) are
        // never NEE-sampled, so they always gather — bit-identical to the no-NEE path, no energy lost.
        // Emission is already the full HDR strength (EMISSIVE_STRENGTH baseline * any override
        // multiplier baked in Java at material-compile time — see MaterialHeader.features packing).
        float emission = payloadEmission();
        if (!canonicalValid && emission > 0.0
                && (!requireReconnectionEndpoint || hitDepth >= 1)) {
            canonicalRadiance = throughput * albedo * emission;
            canonicalValid = true;
        }
        // One gate drives both the RIS call below and the direct-hit emission suppression here. They must
        // agree: gateEmitter exists only because RIS already accounted for this emitter, so a bounce where
        // RIS is skipped must also gather emission directly or that light is lost outright.
        bool gateEmitter = risAtVertex && payloadEmitterInList();
        if (emission > 0.0 && (!gateEmitter || showCelestial)) {
            L += throughput * albedo * emission;
        }

        // NEE: direct light from the dominant celestial body (sun by day, moon by night) — Lambert
        // diffuse + GGX specular, one shadow ray. Jitter the light direction within its square angular
        // extent (worldPush.lightDir.w half-angle) so the shadow ray samples the light's quad — soft penumbrae
        // over accumulation. The same sampled direction drives ndl, the shadow ray, and the specular
        // half-vector.
        float3 lightDir = worldPush.lightDir.xyz;
        float lightHalfAngle = worldPush.lightDir.w;
        if (lightHalfAngle > 0.0) {
            lightDir = sampleSquare(lightDir, lightHalfAngle, seed);
        }
        float ndl = max(0.0, dot(n, lightDir));
        if (ndl > 0.0) {
            VisibilityResult shadow = visibility(p, lightDir, 10000.0);
            float3 vis = shadow.transmittance;
            // Underwater receiver whose shadow ray crossed a water surface: scale the direct light by
            // the wave-refraction caustic at the exit point. Focusing scales the incident irradiance,
            // so it applies to the whole NEE term (diffuse + specular).
            if (medium.current.water && waterWaves && shadow.waterHitT > 0.0) {
                vis *= waterCaustic(p + lightDir * shadow.waterHitT, lightDir, shadow.waterHitT);
            }
            if (max(vis.r, max(vis.g, vis.b)) > 0.0) {
                float3 brdf = diffAlb * INV_PI; // Lambertian diffuse (f = albedo/PI)
                float3 h = normalize(lightDir + v);
                float ndh = max(0.0, dot(n, h));
                float ndv = max(1.0e-4, dot(n, v));
                float vdh = max(0.0, dot(v, h));
                float D = ggxD(ndh, rough);
                float G = ggxG1(ndv, rough) * ggxG1(ndl, rough); // separable Smith masking
                float3 F = fresnelSchlick(vdh, F0);
                brdf += (D * G) * F / (4.0 * ndv * ndl); // Cook–Torrance specular
                L += throughput * brdf * worldPush.lightRadiance.xyz * ndl * vis;
            }
        }

        // RIS direct lighting from block emitters (single-frame only, no temporal reservoir reuse).
        // Front lobe + SSS backlighting are resampled together and share ONE shadow ray — a light can
        // only be in front of n or behind it, never both, so combining them into one target function is
        // exact, not an approximation. The SSS term is gated off past MAX_SSS_INDIRECT_DEPTH by passing
        // sss=0 there (falls back to plain front-only RIS).
        if (risAtVertex) {
            float activeSss = hitDepth <= MAX_SSS_INDIRECT_DEPTH ? sss : 0.0;
            Reservoir r = risInitial(hitPos, n, v, rd, diffAlb, F0, rough, false, activeSss,
                                     seed, proposalSeed);
            if (r.proposalPdf > 0.0) {
                pathAccumulateProposal(lightProposalPdf, lightProposalEvents, r.proposalPdf);
            }
            L += throughput * shadeReservoir(r, hitPos, n, v, rd, diffAlb, F0, rough, false,
                                             activeSss);
        }

        // Thin-surface SSS transmission. Light entering from the back face scatters through toward the
        // viewer via a forward-biased HG phase (leaves/grass backlit by the sun glow when you look
        // through them toward the light source). Shadow ray fires from the back face to avoid
        // self-occlusion. cosT = dot(lightDir, rd): rd tracks the ray toward the sun when the sun is
        // behind the slab, giving cosT ≈ 1 → forward-scatter peak.
        if (sss > 0.0 && hitDepth <= MAX_SSS_INDIRECT_DEPTH) {
            float backNdl = max(0.0, dot(-n, lightDir));
            if (backNdl > 0.0) {
                VisibilityResult shadowBack = visibility(hitPos - n * SURF_BIAS, lightDir, 10000.0);
                float3 visB = shadowBack.transmittance;
                // Same caustic as the front-face NEE — underwater kelp/seagrass transmission should
                // flicker with the same light bands as the floor around it.
                if (medium.current.water && waterWaves && shadowBack.waterHitT > 0.0) {
                    visB *= waterCaustic(hitPos + lightDir * shadowBack.waterHitT,
                                         lightDir, shadowBack.waterHitT);
                }
                if (max(visB.r, max(visB.g, visB.b)) > 0.0) {
                    float cosT = dot(lightDir, rd);
                    L += throughput * diffAlb * sss * SSS_STRENGTH * hg(cosT, SSS_G) * backNdl * worldPush.lightRadiance.xyz * visB;
                }
            }
        }

        // Preserve all local lighting at the terminal hit, but do not sample a continuation that the
        // bounce loop cannot trace.
        if (bounce >= maxBounces) {
            break;
        }

        // Indirect continuation: importance-sample the diffuse OR the GGX specular lobe, chosen by their
        // relative reflectance.
        float ps = exactSpecular && luminance(diffAlb) <= 1.0e-6
                       ? 1.0
                       : clamp(luminance(F0) / (luminance(F0) + luminance(diffAlb) + 1.0e-4), 0.1, 0.9);
        if (rndf(seed) < ps) {
            float3 l;
            if (exactSpecular) {
                // Authored zero is a delta distribution, not a narrow finite GGX lobe. This avoids the
                // degenerate VNDF at alpha=0 and preserves a genuinely sharp reflection without a floor.
                l = reflect(rd, n);
                pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                        max(ps, 1.0e-20));
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalContinuationProposalPdf,
                            canonicalContinuationProposalEvents, max(ps, 1.0e-20));
                }
                float3 eventThroughput =
                        fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), F0) / ps;
                throughput *= eventThroughput;
                transportMask |= PATH_TRANSPORT_DELTA;
                if (captureMetadata && hitDepth == 0) {
                    reconnectionEventKind = PATH_RECONNECTION_EVENT_DELTA;
                    reconnectionThroughput = eventThroughput;
                }
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xB101u);
            } else {
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                l = reflect(rd, h); // rd = -v: reflect the incoming ray about the microfacet normal
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.0) {
                    break; // microfacet reflects below the surface: terminate this path
                }
                float directionalPdf = ggxVndfReflectionPdf(n, v, h, rough);
                float proposalPdf = max(ps * directionalPdf, 1.0e-20);
                pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                        proposalPdf);
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalContinuationProposalPdf,
                            canonicalContinuationProposalEvents,
                            proposalPdf);
                }
                if (captureMetadata && hitDepth == 0) {
                    reconnectionSourcePdf = proposalPdf;
                    reconnectionEventKind = PATH_RECONNECTION_EVENT_GLOSSY;
                }
                // VNDF + separable Smith ⇒ weight = F · G2/G1(v) = F · G1(l); divide by the lobe pdf ps.
                float3 F = fresnelSchlick(max(0.0, dot(v, h)), F0);
                float3 eventThroughput = F * ggxG1(ndl2, rough) / ps;
                throughput *= eventThroughput;
                transportMask |= PATH_TRANSPORT_GLOSSY;
                if (captureMetadata && hitDepth == 0) {
                    reconnectionThroughput = eventThroughput;
                }
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xB102u);
                rayConeSpread = max(rayConeSpread, rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
            }
            ro = p;
            rd = l;
            showCelestial = true; // specular lobe: this ray is a (glossy) mirror — show the disc
        } else {
            float3 eventThroughput = diffAlb / (1.0 - ps);
            throughput *= eventThroughput;
            ro = p;
            rd = cosineDir(n, seed);
            float directionalPdf = max((1.0 - ps) * abs(dot(n, rd)) * INV_PI, 1.0e-20);
            pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                    directionalPdf);
            if (!canonicalValid) {
                pathAccumulateProposal(canonicalContinuationProposalPdf,
                        canonicalContinuationProposalEvents,
                        directionalPdf);
            }
            if (captureMetadata && hitDepth == 0) {
                reconnectionSourcePdf = directionalPdf;
                reconnectionEventKind = PATH_RECONNECTION_EVENT_DIFFUSE;
                reconnectionThroughput = eventThroughput;
            }
            transportMask |= PATH_TRANSPORT_DIFFUSE;
            if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xB103u);
            rayConeSpread = max(rayConeSpread, RAY_CONE_DIFFUSE_SPREAD);
            showCelestial = false; // diffuse lobe: disc covered by NEE, hide it (anti-firefly)
        }

        // Russian roulette on deeper bounces: survive with prob = max channel, rescale survivors.
        if (bounce >= rrStart) {
            float q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.02, 1.0);
            bool survive = rndf(seed) <= q;
            pathAccumulateProposal(rouletteProposalPdf, rouletteProposalEvents,
                    survive ? q : 1.0 - q);
            if (!canonicalValid) {
                pathAccumulateProposal(canonicalRouletteProposalPdf,
                        canonicalRouletteProposalEvents, survive ? q : 1.0 - q);
            }
            if (!survive) {
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xE202u);
                break;
            }
            if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xE201u);
            throughput /= q;
        }
    }
    PathTraceResult result;
    result.radiance = L;
    result.depth = uint(max(indirectDepth, 0));
    result.topologyKey = topologyKey;
    result.replaySeed = replaySeed;
    result.replayState = seed;
    result.proposalReplaySeed = proposalReplaySeed;
    result.proposalReplayState = proposalSeed;
    result.transportMask = transportMask;
    result.footprint = max(rayConeWidth, RAY_CONE_MIN_WIDTH);
    result.lightProposalPdf = lightProposalPdf;
    result.lightProposalEvents = lightProposalEvents;
    result.continuationProposalPdf = continuationProposalPdf;
    result.rouletteProposalPdf = rouletteProposalPdf;
    result.continuationProposalEvents = continuationProposalEvents;
    result.rouletteProposalEvents = rouletteProposalEvents;
    result.canonicalRadiance = canonicalRadiance;
    result.canonicalValid = canonicalValid ? 1u : 0u;
    result.canonicalContinuationProposalPdf = canonicalContinuationProposalPdf;
    result.canonicalRouletteProposalPdf = canonicalRouletteProposalPdf;
    result.canonicalContinuationProposalEvents = canonicalContinuationProposalEvents;
    result.canonicalRouletteProposalEvents = canonicalRouletteProposalEvents;
    result.reconnectionVertex = reconnectionVertex;
    result.reconnectionValid = reconnectionValid ? 1u : 0u;
    result.reconnectionDepth = reconnectionDepth;
    result.reconnectionNormal = reconnectionValid ? reconnectionNormal : float3(0.0);
    result.reconnectionSourcePdf = reconnectionValid ? reconnectionSourcePdf : 0.0;
    result.reconnectionEventKind = reconnectionValid
            ? reconnectionEventKind : PATH_RECONNECTION_EVENT_NONE;
    result.reconnectionThroughput = reconnectionValid
            ? reconnectionThroughput : float3(0.0);
    return result;
}

// Normal rendering derives independent RNG streams from the transient segment/sample identity. Random
// replay bypasses only this derivation and calls tracePathSeeded with the stored pair; all transport,
// light proposals, and roulette decisions then execute through the exact same implementation.
public PathTraceResult tracePath(PathSegment seg, uint2 pix, uint sampleIndex, bool captureMetadata) {
    uint replaySeed = seg.seed ^ sampleIndex * 2246822519u;
    replaySeed = pcg(replaySeed);
    uint proposalReplaySeed = pix.x * 1973u + pix.y * 9277u + 26699u
            ^ worldPush.frameIndex * 2654435761u ^ sampleIndex * 2246822519u;
    proposalReplaySeed = pcg(proposalReplaySeed);
    return tracePathSeeded(seg, pix, captureMetadata, replaySeed, proposalReplaySeed);
}

static float pathMappingDiffuseDensity(float3 normal, float3 edge) {
    float edgeLength = length(edge);
    if (edgeLength <= 1.0e-5) return 0.0;
    return abs(dot(normal, edge / edgeLength)) * INV_PI;
}

static float pathMappingThroughputRatio(float sourceThroughput,
                                        float receiverThroughput) {
    if (sourceThroughput <= 0.0 || receiverThroughput <= 0.0) return 0.0;
    return receiverThroughput / sourceThroughput;
}

// Capture the exact current-frame replay root for an identity sample. Both guide positions and the
// packed queue origins are stored camera-relative, matching the persistent source-root contract.
static bool pathCaptureCurrentGuideRoot(PathReservoir sample, uint2 dimensions,
                                        float4 receiverPositionMaterial,
                                        float4 receiverNormalRoughness,
                                        out PathSourceRoot root) {
    root = {};
    uint pixelCount = dimensions.x * dimensions.y;
    uint sourceIndex = sample.identity.y;
    uint segmentCount = sample.replayControl.y;
    if (pc.pathQueueAddr == 0u || sourceIndex >= pixelCount
            || sample.replayControl.x != PATH_REPLAY_VERSION
            || segmentCount < 1u || segmentCount > MAX_PATH_SEGMENTS) {
        return false;
    }

    ConstPtr<PackedPathSegment> queue =
            ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    PackedPathSegment segment0 = queue[sourceIndex];
    PackedPathSegment segment1 = segment0;
    bool chainComplete = false;
    if (segmentCount == 1u) {
        chainComplete = segment0.nextRecord == PATH_NO_NEXT;
    } else if (segment0.nextRecord != PATH_NO_NEXT
            && segment0.nextRecord < pixelCount * MAX_PATH_SEGMENTS) {
        segment1 = queue[segment0.nextRecord];
        chainComplete = segment1.nextRecord == PATH_NO_NEXT;
    }
    uint2 sourcePixel = uint2(sourceIndex % dimensions.x, sourceIndex / dimensions.x);
    float4 sourcePositionMaterial = gRestirPositionMaterial[int2(sourcePixel)];
    float4 sourceNormalRoughness = gRestirNormalRoughness[int2(sourcePixel)];
    if (!chainComplete || !all(isfinite(segment0.ro)) || !all(isfinite(segment1.ro))
            || !all(isfinite(sourcePositionMaterial))
            || !all(isfinite(sourceNormalRoughness))
            || !all(isfinite(receiverPositionMaterial))
            || !all(isfinite(receiverNormalRoughness))) {
        return false;
    }

    segment0.ro -= worldPush.camOffset;
    segment1.ro -= worldPush.camOffset;
    root.sourcePositionMaterial = sourcePositionMaterial;
    root.sourcePositionMaterial.xyz -= worldPush.camOffset;
    root.sourceNormalRoughness = sourceNormalRoughness;
    root.receiverPositionMaterial = receiverPositionMaterial;
    root.receiverPositionMaterial.xyz -= worldPush.camOffset;
    root.receiverNormalRoughness = receiverNormalRoughness;
    root.segment0 = segment0;
    root.segment1 = segment1;
    return true;
}

// Same-frame validation of the non-persistent ABI-10 scratch mapping. This dispatch runs only after
// path_shifted_radiance has written the consumed slot. It replays the original source seeds through
// the production path tracer, reconstructs receiver PDF/Jacobian/visibility/radiance, and records one
// terminal counter. It never writes either reservoir slot or the output image.
public void pathMappingReplayValidate() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW
            || (pc.historyFlags & PATH_MAPPING_REPLAY_PASS_FLAG) == 0u
            || pc.pathReservoirAddr == 0u
            || pc.pathReservoirPreviousAddr == 0u
            || worldPush.pathShiftDiagnosticCounterAddr == 0u
            || worldPush.pathShiftSourceRootAddr == 0u
            || worldPush.pathShiftReceiverGuideAddr == 0u
            || worldPush.pathShiftGuideScratchAddr == 0u
            || worldPush.pathShiftGuideRootScratchAddr == 0u) {
        return;
    }

    bool crossFrame = (pc.historyFlags & PATH_CROSS_FRAME_MAPPING_REPLAY_PASS_FLAG) != 0u;
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    DevicePtr<uint> counters =
            DevicePtr<uint>(worldPush.pathShiftDiagnosticCounterAddr);
    uint mappedPixelIndex = pixelIndex;
    if (crossFrame) {
        InterlockedAdd(counters[PATH_CROSS_FRAME_ATTEMPTED], 1u);
        float2 previousPixel = float2(pixel) + 0.5 + gMotion[int2(pixel)];
        int2 previousTexel = int2(floor(previousPixel));
        if (any(previousTexel < int2(0, 0))
                || any(previousTexel >= int2(dimensions))) {
            InterlockedAdd(
                    counters[PATH_CROSS_FRAME_RECEIVER_REPROJECTION_REJECT], 1u);
            return;
        }
        mappedPixelIndex = uint(previousTexel.y) * dimensions.x + uint(previousTexel.x);
    }
    PathReservoir mapped =
            ConstPtr<PathReservoir>(pc.pathReservoirPreviousAddr)[mappedPixelIndex];
    uint mappingControl = asuint(mapped.reconnectionThroughput.w);
    if ((mappingControl & PATH_MAPPING_KIND_MASK)
            != PATH_MAPPING_DIFFUSE_RECONNECTION) {
        if (crossFrame) {
            InterlockedAdd(counters[PATH_CROSS_FRAME_MAPPED_EMPTY], 1u);
        }
        return;
    }
    InterlockedAdd(counters[crossFrame
            ? PATH_CROSS_FRAME_ELIGIBLE : PATH_MAPPING_REPLAY_ELIGIBLE], 1u);

    uint pixelCount = dimensions.x * dimensions.y;
    uint sourceIndex = mapped.identity.y;
    if (!pathMappingControlValid(mapped)
            || mapped.replayControl.x != PATH_REPLAY_VERSION
            || mapped.replayControl.y < 1u
            || mapped.replayControl.y > MAX_PATH_SEGMENTS
            || (mapped.replayControl.y < MAX_PATH_SEGMENTS
                    && any(mapped.replay.zw != uint2(0u)))
             || asuint(mapped.proposalShift.w) != mapped.replayControl.z
             || sourceIndex >= pixelCount) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_ABI_REJECT : PATH_MAPPING_REPLAY_ABI_REJECT], 1u);
        return;
    }

    uint expectedSegments = mapped.replayControl.y;
    PathSourceRoot sourceRoot =
            ConstPtr<PathSourceRoot>(worldPush.pathShiftSourceRootAddr)[mappedPixelIndex];
    bool sourceRootFinite = all(isfinite(sourceRoot.sourcePositionMaterial))
            && all(isfinite(sourceRoot.sourceNormalRoughness))
            && all(isfinite(sourceRoot.receiverPositionMaterial))
            && all(isfinite(sourceRoot.receiverNormalRoughness))
            && all(isfinite(sourceRoot.segment0.ro))
            && all(isfinite(sourceRoot.segment1.ro));
    bool sourceRootChainValid = expectedSegments == 1u
            ? sourceRoot.segment0.nextRecord == PATH_NO_NEXT
            : sourceRoot.segment0.nextRecord != PATH_NO_NEXT
                    && sourceRoot.segment1.nextRecord == PATH_NO_NEXT;
    if (!sourceRootFinite || !sourceRootChainValid) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_SOURCE_ROOT_REJECT
                : PATH_MAPPING_REPLAY_SOURCE_ROOT_REJECT], 1u);
        return;
    }

    uint2 previousSourcePixel = uint2(
            sourceIndex % dimensions.x, sourceIndex / dimensions.x);
    float3 retainedRebase = worldPush.camOffset
            - (crossFrame ? worldPush.camDelta : float3(0.0));
    float4 retainedSourcePositionMaterial = sourceRoot.sourcePositionMaterial;
    retainedSourcePositionMaterial.xyz += retainedRebase;
    float4 retainedReceiverPositionMaterial = sourceRoot.receiverPositionMaterial;
    retainedReceiverPositionMaterial.xyz += retainedRebase;

    int2 receiverTexel = int2(pixel);
    float4 receiverPositionMaterial = gRestirPositionMaterial[receiverTexel];
    float4 receiverNormalRoughness = gRestirNormalRoughness[receiverTexel];
    if (crossFrame
            && (!pathReplayPositionMatches(retainedReceiverPositionMaterial.xyz,
                        receiverPositionMaterial.xyz)
                    || !pathReplayColorMatches(sourceRoot.receiverNormalRoughness.xyz,
                        receiverNormalRoughness.xyz)
                    || !pathReplayFloatMatches(sourceRoot.receiverNormalRoughness.w,
                        receiverNormalRoughness.w)
                    || uint(max(retainedReceiverPositionMaterial.w, 0.0))
                            != uint(max(receiverPositionMaterial.w, 0.0)))) {
        InterlockedAdd(counters[PATH_CROSS_FRAME_RECEIVER_SURFACE_REJECT], 1u);
        return;
    }

    int2 sourceTexel = int2(previousSourcePixel);
    float4 currentSourcePositionMaterial = float4(0.0);
    float4 currentSourceNormalRoughness = float4(0.0);
    bool sourceReprojected = !crossFrame;
    if (crossFrame) {
        float3 currentSourceCamRelative = sourceRoot.sourcePositionMaterial.xyz
                - worldPush.camDelta;
        float4 sourceClip = mul(worldPush.curViewProj,
                float4(currentSourceCamRelative, 1.0));
        if (sourceClip.w > 0.0 && all(isfinite(sourceClip))) {
            float2 sourceNdc = sourceClip.xy / sourceClip.w;
            int2 projectedSourceTexel = int2(floor(
                    (sourceNdc * 0.5 + 0.5) * float2(dimensions)));
            for (int oy = -1; oy <= 1 && !sourceReprojected; ++oy) {
                for (int ox = -1; ox <= 1 && !sourceReprojected; ++ox) {
                    int2 candidateTexel = projectedSourceTexel + int2(ox, oy);
                    if (any(candidateTexel < int2(0, 0))
                            || any(candidateTexel >= int2(dimensions))) continue;
                    int2 candidatePreviousTexel = int2(floor(float2(candidateTexel)
                            + 0.5 + gMotion[candidateTexel]));
                    if (any(candidatePreviousTexel != int2(previousSourcePixel))) continue;
                    float4 candidatePosition = gRestirPositionMaterial[candidateTexel];
                    float4 candidateNormal = gRestirNormalRoughness[candidateTexel];
                    if (pathReplayPositionMatches(retainedSourcePositionMaterial.xyz,
                                candidatePosition.xyz)
                            && pathReplayColorMatches(sourceRoot.sourceNormalRoughness.xyz,
                                candidateNormal.xyz)
                            && pathReplayFloatMatches(sourceRoot.sourceNormalRoughness.w,
                                candidateNormal.w)
                            && uint(max(retainedSourcePositionMaterial.w, 0.0))
                                    == uint(max(candidatePosition.w, 0.0))) {
                        sourceTexel = candidateTexel;
                        currentSourcePositionMaterial = candidatePosition;
                        currentSourceNormalRoughness = candidateNormal;
                        sourceReprojected = true;
                    }
                }
            }
        }
        if (!sourceReprojected) {
            InterlockedAdd(
                    counters[PATH_CROSS_FRAME_SOURCE_REPROJECTION_REJECT], 1u);
            return;
        }
    } else {
        currentSourcePositionMaterial = gRestirPositionMaterial[sourceTexel];
        currentSourceNormalRoughness = gRestirNormalRoughness[sourceTexel];
        if (!pathReplayPositionMatches(retainedSourcePositionMaterial.xyz,
                    currentSourcePositionMaterial.xyz)
                || !pathReplayColorMatches(sourceRoot.sourceNormalRoughness.xyz,
                    currentSourceNormalRoughness.xyz)
                || !pathReplayFloatMatches(sourceRoot.sourceNormalRoughness.w,
                    currentSourceNormalRoughness.w)
                || uint(max(retainedSourcePositionMaterial.w, 0.0))
                        != uint(max(currentSourcePositionMaterial.w, 0.0))) {
            InterlockedAdd(counters[PATH_MAPPING_REPLAY_SOURCE_ROOT_REJECT], 1u);
            return;
        }
    }

    PathReplayAggregate replay = pathReplayAggregateEmpty();
    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
        if (leaf >= expectedSegments) break;
        PackedPathSegment packed = leaf == 0u
                ? sourceRoot.segment0 : sourceRoot.segment1;
        // The sidecar stores capture-frame camera-relative origins. Cross-frame replay reconstructs
        // static world geometry in the current terrain rebase and rejects object motion separately.
        packed.ro += retainedRebase;
        PathSegment segment = unpackPathSegment(packed);
        uint replaySeed = leaf == 0u ? mapped.replay.x : mapped.replay.z;
        uint proposalReplaySeed = leaf == 0u ? mapped.replay.y : mapped.replay.w;
        PathTraceResult traced = tracePathSeeded(
                segment, uint2(sourceTexel), true, replaySeed, proposalReplaySeed);
        pathReplayAccumulate(replay, traced, leaf);
    }
    if (!pathMappingSourceReplayMatches(mapped, replay)) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_SOURCE_REPLAY_REJECT
                : PATH_MAPPING_REPLAY_SOURCE_REJECT], 1u);
        return;
    }
    float4 sourcePositionMaterial = retainedSourcePositionMaterial;
    float4 sourceNormalRoughness = sourceRoot.sourceNormalRoughness;
    float receiverNormalLength = length(receiverNormalRoughness.xyz);
    float sourceNormalLength = length(sourceNormalRoughness.xyz);
    uint receiverMaterialIdentity =
            uint(max(receiverPositionMaterial.w, 0.0));
    uint sourceMaterialIdentity = uint(max(sourcePositionMaterial.w, 0.0));
    uint receiverMaterial = receiverMaterialModel(receiverMaterialIdentity);
    PathReservoir current =
            ConstPtr<PathReservoir>(pc.pathReservoirAddr)[pixelIndex];
    uint currentReconnectionMetadata = asuint(current.reconnectionVertex.w);
    bool currentEdgeValid =
            (currentReconnectionMetadata & PATH_RECONNECTION_VALID) != 0u;
    bool currentEdgeDepth =
            ((currentReconnectionMetadata >> PATH_RECONNECTION_DEPTH_SHIFT) & 0xFu) == 1u;
    bool currentEdgeEvent = pathReconnectionEvent(currentReconnectionMetadata)
            == PATH_RECONNECTION_EVENT_DIFFUSE;
    bool currentEdgeMapping = pathMappingControlValid(current)
            && pathMappingKind(current) == PATH_MAPPING_IDENTITY;
    bool currentEdgePdf = current.reconnectionNormalPdf.w > 0.0;
    bool currentEdgeFinite = all(isfinite(current.reconnectionVertex.xyz))
            && all(isfinite(current.reconnectionThroughput.xyz));
    bool currentEdgeAvailable = currentEdgeValid && currentEdgeDepth && currentEdgeEvent
            && currentEdgeMapping && currentEdgePdf && currentEdgeFinite;
    if (receiverNormalLength < 0.5 || sourceNormalLength < 0.5
            || (receiverMaterial != MATERIAL_OPAQUE
                    && receiverMaterial != MATERIAL_PARTICLE)
            || receiverMaterialIdentity != sourceMaterialIdentity) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_RECEIVER_SURFACE_REJECT
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        return;
    }
    if (crossFrame) {
        // Validate the deterministic primary receiver contract independently of endpoint selection.
        // Only positive stored diffuse edges participate in the numerical comparison; empty/unsupported
        // current records remain an explicit category and never borrow the guide for actual remapping.
        InterlockedAdd(counters[PATH_RECEIVER_GUIDE_ATTEMPTED], 1u);
        ConstPtr<float4> receiverGuides =
                ConstPtr<float4>(worldPush.pathShiftReceiverGuideAddr);
        float4 receiverGuide = receiverGuides[pixelIndex * 2u];
        float4 receiverRayOrigin = receiverGuides[pixelIndex * 2u + 1u];
        bool guideValid = all(isfinite(receiverGuide))
                && all(isfinite(receiverRayOrigin))
                && all(receiverGuide.xyz >= float3(0.0))
                && receiverGuide.w > 0.0 && receiverGuide.w <= 1.0
                && receiverRayOrigin.w == 1.0;
        if (!guideValid) {
            InterlockedAdd(counters[PATH_RECEIVER_GUIDE_INVALID], 1u);
        } else if (!currentEdgeAvailable) {
            // The exact material/origin substrate is valid even when endpoint selection publishes
            // no canonical edge. This is the population the eventual receiver remap must serve.
            InterlockedAdd(counters[PATH_RECEIVER_GUIDE_NO_STORED_EDGE], 1u);
            InterlockedAdd(counters[PATH_GUIDE_REMAP_ELIGIBLE], 1u);

            // Shadow A/B only: evaluate whether the replayed source and endpoint-independent receiver
            // state provide finite pre-visibility remap terms. Strict admission below is unchanged.
            float3 guideReceiverNormal =
                    receiverNormalRoughness.xyz / receiverNormalLength;
            float3 guideSharedVertex = replay.reconnectionVertex;
            float guideSharedNormalLength = length(replay.reconnectionNormal);
            float3 guideSourceEdge = guideSharedVertex - sourcePositionMaterial.xyz;
            float3 guideReceiverEdge = guideSharedVertex - receiverPositionMaterial.xyz;
            float guideSourceDistance = length(guideSourceEdge);
            float guideReceiverDistance = length(guideReceiverEdge);
            bool guideGeometryValid = replay.reconnectionValid != 0u
                    && replay.reconnectionDepth == 1u
                    && replay.reconnectionEventKind == PATH_RECONNECTION_EVENT_DIFFUSE
                    && guideSharedNormalLength >= 0.5
                    && guideSourceDistance > 1.0e-4
                    && guideReceiverDistance > 1.0e-4;
            float guideGeometryJacobian = 0.0;
            if (guideGeometryValid) {
                float3 guideSharedNormal =
                        replay.reconnectionNormal / guideSharedNormalLength;
                float guideSourceCosine = abs(dot(guideSharedNormal,
                        -guideSourceEdge / guideSourceDistance));
                float guideReceiverCosine = abs(dot(guideSharedNormal,
                        guideReceiverEdge / guideReceiverDistance));
                guideGeometryJacobian = guideReceiverCosine
                        * guideSourceDistance * guideSourceDistance
                        / max(guideSourceCosine
                                * guideReceiverDistance * guideReceiverDistance, 1.0e-8);
                guideGeometryValid = guideSourceCosine > 0.0
                        && guideReceiverCosine > 0.0
                        && isfinite(guideGeometryJacobian)
                        && guideGeometryJacobian > 0.0;
            }
            if (!guideGeometryValid) {
                InterlockedAdd(counters[PATH_GUIDE_REMAP_GEOMETRY_REJECT], 1u);
            } else {
                float3 guideSampledEdge = guideSharedVertex - receiverRayOrigin.xyz;
                float guideReceiverPdf = receiverGuide.w
                        * pathMappingDiffuseDensity(guideReceiverNormal, guideSampledEdge);
                float guideSourcePdf = replay.reconnectionSourcePdf;
                float guidePssJacobian = guideGeometryJacobian
                        * guideReceiverPdf / guideSourcePdf;
                bool guidePdfValid = isfinite(guideReceiverPdf)
                        && guideReceiverPdf > 0.0
                        && isfinite(guideSourcePdf) && guideSourcePdf > 0.0
                        && isfinite(guidePssJacobian) && guidePssJacobian > 0.0;
                if (!guidePdfValid) {
                    InterlockedAdd(counters[PATH_GUIDE_REMAP_PDF_REJECT], 1u);
                } else {
                    float3 guideSourceRadiance = replay.canonicalRadiance;
                    float3 guideSourceThroughput = replay.reconnectionThroughput;
                    float3 guideReceiverThroughput = receiverGuide.xyz / receiverGuide.w;
                    bool guideInvalidSourceChannel = any(guideSourceRadiance > float3(0.0)
                            && guideSourceThroughput <= float3(0.0));
                    float3 guideThroughputRatio = float3(
                            pathMappingThroughputRatio(guideSourceThroughput.x,
                                    guideReceiverThroughput.x),
                            pathMappingThroughputRatio(guideSourceThroughput.y,
                                    guideReceiverThroughput.y),
                            pathMappingThroughputRatio(guideSourceThroughput.z,
                                    guideReceiverThroughput.z));
                    float3 guideShiftedRadiance =
                            guideSourceRadiance * guideThroughputRatio;
                    bool guideThroughputValid = !guideInvalidSourceChannel
                            && all(isfinite(guideSourceRadiance))
                            && all(isfinite(guideSourceThroughput))
                            && all(isfinite(guideReceiverThroughput))
                            && all(isfinite(guideThroughputRatio))
                            && all(isfinite(guideShiftedRadiance))
                            && all(guideSourceRadiance >= float3(0.0))
                            && all(guideSourceThroughput >= float3(0.0))
                            && all(guideReceiverThroughput >= float3(0.0))
                            && all(guideThroughputRatio >= float3(0.0))
                            && all(guideShiftedRadiance >= float3(0.0));
                    if (!guideThroughputValid) {
                        InterlockedAdd(counters[PATH_GUIDE_REMAP_THROUGHPUT_REJECT], 1u);
                    } else {
                        InterlockedAdd(counters[PATH_GUIDE_REMAP_READY], 1u);
                        InterlockedAdd(counters[PATH_GUIDE_VISIBILITY_ELIGIBLE], 1u);

                        // Visibility shadow A/B only. Use the exact Pass-B biased receiver origin
                        // and the production shadow SBT, but publish counters rather than radiance.
                        float3 guideToShared = guideSharedVertex - receiverRayOrigin.xyz;
                        float guideVisibilityDistance = length(guideToShared);
                        if (!isfinite(guideVisibilityDistance)
                                || guideVisibilityDistance <= 1.0e-4) {
                            InterlockedAdd(counters[PATH_GUIDE_VISIBILITY_INVALID], 1u);
                        } else {
                            VisibilityResult guideVisibility = visibility(receiverRayOrigin.xyz,
                                    guideToShared / guideVisibilityDistance,
                                    guideVisibilityDistance * 0.999);
                            float3 guideTransmittance = guideVisibility.transmittance;
                            bool guideVisibilityValid = all(isfinite(guideTransmittance))
                                    && all(guideTransmittance >= float3(0.0))
                                    && all(guideTransmittance <= float3(1.0));
                            uint guideVisibilityResult = !guideVisibilityValid
                                    ? PATH_GUIDE_VISIBILITY_INVALID
                                    : max(guideTransmittance.x,
                                            max(guideTransmittance.y,
                                                    guideTransmittance.z)) <= 0.0
                                            ? PATH_GUIDE_VISIBILITY_OCCLUDED
                                            : all(guideTransmittance == float3(1.0))
                                                    ? PATH_GUIDE_VISIBILITY_CLEAR
                                                    : PATH_GUIDE_VISIBILITY_TINTED;
                            InterlockedAdd(counters[guideVisibilityResult], 1u);
                            if (guideVisibilityValid) {
                                InterlockedAdd(counters[PATH_GUIDE_TARGET_ELIGIBLE], 1u);
                                float3 guideVisibleRadiance =
                                        guideShiftedRadiance * guideTransmittance;
                                float guideShiftedTarget = pathLuminance(guideVisibleRadiance);
                                bool guideTargetValid = all(isfinite(guideVisibleRadiance))
                                        && all(guideVisibleRadiance >= float3(0.0))
                                        && isfinite(guideShiftedTarget)
                                        && guideShiftedTarget >= 0.0;
                                uint guideTargetResult = !guideTargetValid
                                        ? PATH_GUIDE_TARGET_INVALID
                                        : guideShiftedTarget > 0.0
                                                ? PATH_GUIDE_TARGET_POSITIVE
                                                : PATH_GUIDE_TARGET_ZERO;
                                InterlockedAdd(counters[guideTargetResult], 1u);
                                if (guideTargetValid) {
                                    // Counter-only GRIS readiness audit. This deliberately forms no
                                    // selection probability and writes no reservoir or history.
                                    InterlockedAdd(counters[PATH_GUIDE_WEIGHT_ELIGIBLE], 1u);
                                    float guideSourceFinalWeight = mapped.weights.z;
                                    float guideSourceEffectiveCount = mapped.weights.y;
                                    float guideSourceCount = min(guideSourceEffectiveCount,
                                            PATH_TEMPORAL_MAX_SOURCE_COUNT);
                                    bool guideWeightTermsValid =
                                            isfinite(guideSourceFinalWeight)
                                            && guideSourceFinalWeight >= 0.0
                                            && isfinite(guideSourceEffectiveCount)
                                            && guideSourceEffectiveCount >= 0.0;
                                    float guideMergeWeight = guideShiftedTarget
                                            * guideSourceFinalWeight * guideSourceCount
                                            * guidePssJacobian;
                                    bool guideWeightValid = guideWeightTermsValid
                                            && isfinite(guideMergeWeight)
                                            && guideMergeWeight >= 0.0;
                                    uint guideWeightResult = !guideWeightValid
                                            ? PATH_GUIDE_WEIGHT_INVALID
                                            : guideMergeWeight > 0.0
                                                    ? PATH_GUIDE_WEIGHT_POSITIVE
                                                    : PATH_GUIDE_WEIGHT_ZERO;
                                    InterlockedAdd(counters[guideWeightResult], 1u);
                                    if (guideWeightValid) {
                                        // Selection-readiness audit only: evaluate the probability
                                        // without consuming RNG or mutating any reservoir state.
                                        InterlockedAdd(
                                                counters[PATH_GUIDE_SELECTION_ELIGIBLE], 1u);
                                        float guideCurrentWeightSum = current.weights.x;
                                        bool guideCurrentWeightValid =
                                                isfinite(guideCurrentWeightSum)
                                                && guideCurrentWeightSum >= 0.0;
                                        float guideNextWeightSum = min(guideCurrentWeightSum
                                                + guideMergeWeight, 1.0e30);
                                        float guideSelectionProbability = guideMergeWeight > 0.0
                                                && guideNextWeightSum > 0.0
                                                ? guideMergeWeight / guideNextWeightSum : 0.0;
                                        bool guideSelectionArithmeticValid =
                                                isfinite(guideNextWeightSum)
                                                && guideNextWeightSum >= 0.0
                                                && isfinite(guideSelectionProbability)
                                                && guideSelectionProbability >= 0.0;
                                        uint guideSelectionResult = !guideCurrentWeightValid
                                                ? PATH_GUIDE_SELECTION_CURRENT_REJECT
                                                : !guideSelectionArithmeticValid
                                                        ? PATH_GUIDE_SELECTION_ARITHMETIC_REJECT
                                                        : guideSelectionProbability > 1.0
                                                                ? PATH_GUIDE_SELECTION_PROBABILITY_HIGH_REJECT
                                                                : guideSelectionProbability > 0.0
                                                                        ? PATH_GUIDE_SELECTION_POSITIVE
                                                                        : PATH_GUIDE_SELECTION_ZERO;
                                        InterlockedAdd(counters[guideSelectionResult], 1u);

                                        // Counter-only A/B against the rejected capped-denominator
                                        // formula above. The stable ratio uses uncapped relative
                                        // weights and performs no RNG draw or reservoir mutation.
                                        InterlockedAdd(
                                                counters[PATH_GUIDE_STABLE_SELECTION_ELIGIBLE], 1u);
                                        float guideStableSelectionProbability =
                                                guideCurrentWeightValid
                                                ? pathStableSelectionProbability(
                                                        guideCurrentWeightSum, guideMergeWeight)
                                                : 0.0;
                                        bool guideStableSelectionValid = guideCurrentWeightValid
                                                && isfinite(guideStableSelectionProbability)
                                                && guideStableSelectionProbability >= 0.0
                                                && guideStableSelectionProbability <= 1.0;
                                        uint guideStableSelectionResult =
                                                !guideStableSelectionValid
                                                ? PATH_GUIDE_STABLE_SELECTION_INVALID
                                                : guideStableSelectionProbability > 0.0
                                                        ? PATH_GUIDE_STABLE_SELECTION_POSITIVE
                                                        : PATH_GUIDE_STABLE_SELECTION_ZERO;
                                        InterlockedAdd(
                                                counters[guideStableSelectionResult], 1u);

                                        // A local hash-derived draw keeps this counter-only gate
                                        // independent of both stored replay RNG streams.
                                        InterlockedAdd(
                                                counters[PATH_GUIDE_STABLE_BERNOULLI_ELIGIBLE], 1u);
                                        uint guideBernoulliSeed = pathHash(pixelIndex
                                                ^ worldPush.frameIndex * 747796405u
                                                ^ 0xA511E9B3u);
                                        float guideBernoulliRandom =
                                                float(guideBernoulliSeed >> 8u)
                                                * (1.0 / 16777216.0);
                                        bool guideBernoulliValid = guideStableSelectionValid
                                                && isfinite(guideBernoulliRandom)
                                                && guideBernoulliRandom >= 0.0
                                                && guideBernoulliRandom < 1.0;
                                        bool guideSourceSelected = guideBernoulliValid
                                                && guideBernoulliRandom
                                                        < guideStableSelectionProbability;
                                        uint guideBernoulliResult = !guideBernoulliValid
                                                ? PATH_GUIDE_STABLE_BERNOULLI_INVALID
                                                : guideSourceSelected
                                                        ? PATH_GUIDE_STABLE_BERNOULLI_SELECTED
                                                        : PATH_GUIDE_STABLE_BERNOULLI_RETAINED;
                                        InterlockedAdd(counters[guideBernoulliResult], 1u);

                                        // Shadow only the arithmetic needed after selection. No
                                        // sample metadata or reservoir state is copied or written.
                                        InterlockedAdd(
                                                counters[PATH_GUIDE_POST_SELECTION_ELIGIBLE], 1u);
                                        float guidePostWeightSum = min(guideCurrentWeightSum
                                                + guideMergeWeight, 1.0e30);
                                        float guidePostEffectiveCount = min(current.weights.y
                                                + guideSourceCount, 16777216.0);
                                        float guidePostSelectedTarget = guideSourceSelected
                                                ? guideShiftedTarget
                                                : current.sampleRadianceTarget.w;
                                        bool guidePostBaseValid = guideBernoulliValid
                                                && isfinite(current.weights.y)
                                                && current.weights.y >= 0.0
                                                && isfinite(guidePostWeightSum)
                                                && guidePostWeightSum >= 0.0
                                                && isfinite(guidePostEffectiveCount)
                                                && guidePostEffectiveCount >= 0.0;
                                        bool guidePostSampleValid =
                                                isfinite(guidePostSelectedTarget)
                                                && guidePostSelectedTarget >= 0.0;
                                        float guidePostFinalWeight = 0.0;
                                        uint guidePostSelectionResult;
                                        if (!guidePostBaseValid) {
                                            guidePostSelectionResult =
                                                    PATH_GUIDE_POST_SELECTION_ARITHMETIC_REJECT;
                                        } else if (!guidePostSampleValid) {
                                            guidePostSelectionResult =
                                                    PATH_GUIDE_POST_SELECTION_SAMPLE_REJECT;
                                        } else if (guidePostWeightSum == 0.0) {
                                            guidePostSelectionResult =
                                                    PATH_GUIDE_POST_SELECTION_EMPTY;
                                        } else if (guidePostSelectedTarget == 0.0) {
                                            guidePostSelectionResult =
                                                    PATH_GUIDE_POST_SELECTION_SAMPLE_REJECT;
                                        } else {
                                            float guidePostDenominator = guidePostEffectiveCount
                                                    * guidePostSelectedTarget;
                                            guidePostFinalWeight = guidePostWeightSum
                                                    / guidePostDenominator;
                                            bool guidePostArithmeticValid =
                                                    guidePostEffectiveCount > 0.0
                                                    && isfinite(guidePostDenominator)
                                                    && guidePostDenominator > 0.0
                                                    && isfinite(guidePostFinalWeight)
                                                    && guidePostFinalWeight > 0.0;
                                            guidePostSelectionResult =
                                                    !guidePostArithmeticValid
                                                    ? PATH_GUIDE_POST_SELECTION_ARITHMETIC_REJECT
                                                    : guideSourceSelected
                                                            ? PATH_GUIDE_POST_SELECTION_SELECTED_READY
                                                            : PATH_GUIDE_POST_SELECTION_RETAINED_READY;
                                        }
                                        InterlockedAdd(
                                                counters[guidePostSelectionResult], 1u);

                                        // Build the complete future sample in registers before
                                        // copying it to the isolated, cleared guide scratch. A
                                        // source selection receives the current receiver mapping;
                                        // a retained selection preserves every current sample bit.
                                        InterlockedAdd(
                                                counters[PATH_GUIDE_SAMPLE_COPY_ELIGIBLE], 1u);
                                        uint guideSampleCopyResult;
                                        PathReservoir guideStoredScratch = pathReservoirEmpty();
                                        PathSourceRoot guideStoredRoot;
                                        bool guideStoredRootValid = false;
                                        if (guidePostSelectionResult
                                                == PATH_GUIDE_POST_SELECTION_EMPTY) {
                                            guideSampleCopyResult = PATH_GUIDE_SAMPLE_COPY_EMPTY;
                                        } else if (guidePostSelectionResult
                                                == PATH_GUIDE_POST_SELECTION_SAMPLE_REJECT) {
                                            guideSampleCopyResult =
                                                    PATH_GUIDE_SAMPLE_COPY_METADATA_REJECT;
                                        } else if (guidePostSelectionResult
                                                == PATH_GUIDE_POST_SELECTION_ARITHMETIC_REJECT) {
                                            guideSampleCopyResult =
                                                    PATH_GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT;
                                        } else {
                                            PathReservoir guideScratch = current;
                                            if (guideSourceSelected) {
                                                uint guideCurrentSourceIndex =
                                                        uint(sourceTexel.y) * dimensions.x
                                                        + uint(sourceTexel.x);
                                                guideScratch.sampleRadianceTarget = float4(
                                                        guideVisibleRadiance,
                                                        guideShiftedTarget);
                                                guideScratch.proposalShift = float4(
                                                        mapped.proposalShift.x,
                                                        guidePssJacobian,
                                                        max(mapped.proposalShift.z, 1.0e-5),
                                                        asfloat(mapped.replayControl.z));
                                                guideScratch.proposalComponents =
                                                        mapped.proposalComponents;
                                                guideScratch.canonicalRadianceTarget = float4(
                                                        guideVisibleRadiance,
                                                        guideShiftedTarget);
                                                guideScratch.identity = uint4(
                                                        mapped.identity.x,
                                                        guideCurrentSourceIndex,
                                                        replay.topologyKey,
                                                        pathPackMetadata(pc.historyGeneration,
                                                                replay.depth,
                                                                replay.transportMask));
                                                guideScratch.replay = mapped.replay;
                                                guideScratch.replayControl = mapped.replayControl;
                                                guideScratch.reconnectionVertex = float4(
                                                        replay.reconnectionVertex,
                                                        asfloat(pathPackReconnectionMetadata(
                                                                replay.reconnectionDepth,
                                                                replay.reconnectionEventKind,
                                                                replay.reconnectionValid != 0u)));
                                                guideScratch.reconnectionNormalPdf = float4(
                                                        replay.reconnectionNormal,
                                                        guideReceiverPdf);
                                                guideScratch.reconnectionThroughput = float4(
                                                        guideReceiverThroughput,
                                                        asfloat(
                                                                PATH_MAPPING_DIFFUSE_RECONNECTION));
                                                // The original path remains the replay root, but its
                                                // stable screen key and camera-relative positions
                                                // advance to the independently reprojected source in
                                                // this frame. Never overwrite the persistent root
                                                // buffer that other cross-frame invocations read.
                                                guideStoredRoot = sourceRoot;
                                                guideStoredRoot.sourcePositionMaterial =
                                                        currentSourcePositionMaterial;
                                                guideStoredRoot.sourcePositionMaterial.xyz -=
                                                        worldPush.camOffset;
                                                guideStoredRoot.sourceNormalRoughness =
                                                        currentSourceNormalRoughness;
                                                guideStoredRoot.receiverPositionMaterial =
                                                        receiverPositionMaterial;
                                                guideStoredRoot.receiverPositionMaterial.xyz -=
                                                        worldPush.camOffset;
                                                guideStoredRoot.receiverNormalRoughness =
                                                        receiverNormalRoughness;
                                                guideStoredRoot.segment0.ro -= worldPush.camDelta;
                                                guideStoredRoot.segment1.ro -= worldPush.camDelta;
                                                guideStoredRootValid = true;
                                            } else {
                                                guideStoredRootValid = pathCaptureCurrentGuideRoot(
                                                        current, dimensions,
                                                        receiverPositionMaterial,
                                                        receiverNormalRoughness,
                                                        guideStoredRoot);
                                            }
                                            guideScratch.weights = float4(
                                                    guidePostWeightSum,
                                                    guidePostEffectiveCount,
                                                    guidePostFinalWeight,
                                                    min(guidePostEffectiveCount * 0.25, 1.0));

                                            bool guideScratchWeightsValid =
                                                    all(isfinite(guideScratch.weights))
                                                    && all(guideScratch.weights
                                                            > float4(0.0));
                                            bool guideScratchMetadataValid;
                                            if (guideSourceSelected) {
                                                uint guideScratchReconnection = asuint(
                                                        guideScratch.reconnectionVertex.w);
                                                guideScratchMetadataValid =
                                                        pathReservoirSampleMetadataReady(
                                                                guideScratch)
                                                        && pathReplayColorMatches(
                                                                guideScratch
                                                                        .sampleRadianceTarget.xyz,
                                                                guideVisibleRadiance)
                                                        && pathReplayFloatMatches(
                                                                guideScratch
                                                                        .sampleRadianceTarget.w,
                                                                guideShiftedTarget)
                                                        && all(asuint(guideScratch
                                                                        .canonicalRadianceTarget)
                                                                == asuint(guideScratch
                                                                        .sampleRadianceTarget))
                                                        && guideScratch.identity.y
                                                                == uint(sourceTexel.y)
                                                                        * dimensions.x
                                                                        + uint(sourceTexel.x)
                                                        && guideScratch.replayControl.x
                                                                == PATH_REPLAY_VERSION
                                                        && asuint(guideScratch.proposalShift.w)
                                                                == guideScratch.replayControl.z
                                                        && pathMappingControlValid(guideScratch)
                                                        && pathMappingKind(guideScratch)
                                                                == PATH_MAPPING_DIFFUSE_RECONNECTION
                                                        && (guideScratchReconnection
                                                                & PATH_RECONNECTION_VALID) != 0u
                                                        && pathReconnectionEvent(
                                                                guideScratchReconnection)
                                                                == PATH_RECONNECTION_EVENT_DIFFUSE
                                                        && all(isfinite(guideScratch
                                                                .reconnectionNormalPdf))
                                                        && guideScratch
                                                                .reconnectionNormalPdf.w > 0.0
                                                        && all(isfinite(guideScratch
                                                                .reconnectionThroughput.xyz))
                                                        && all(guideScratch
                                                                .reconnectionThroughput.xyz
                                                                >= float3(0.0));
                                            } else {
                                                guideScratchMetadataValid =
                                                        pathReservoirSampleMetadataReady(
                                                                guideScratch)
                                                        && pathReservoirSampleMetadataMatches(
                                                                guideScratch, current)
                                                        && pathReplayFloatMatches(
                                                                guideScratch
                                                                        .sampleRadianceTarget.w,
                                                                guidePostSelectedTarget);
                                            }
                                            guideSampleCopyResult = !guideScratchWeightsValid
                                                    ? PATH_GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT
                                                    : !guideScratchMetadataValid
                                                            ? PATH_GUIDE_SAMPLE_COPY_METADATA_REJECT
                                                            : guideSourceSelected
                                                                    ? PATH_GUIDE_SAMPLE_COPY_SELECTED
                                                                    : PATH_GUIDE_SAMPLE_COPY_RETAINED;
                                            if (guideSampleCopyResult
                                                    == PATH_GUIDE_SAMPLE_COPY_SELECTED
                                                    || guideSampleCopyResult
                                                            == PATH_GUIDE_SAMPLE_COPY_RETAINED) {
                                                guideStoredScratch = guideScratch;
                                            }
                                        }
                                        InterlockedAdd(counters[guideSampleCopyResult], 1u);
                                        if (guideSampleCopyResult != PATH_GUIDE_SAMPLE_COPY_SELECTED
                                                && guideSampleCopyResult
                                                        != PATH_GUIDE_SAMPLE_COPY_RETAINED) {
                                            guideStoredScratch.identity.w =
                                                    pc.historyGeneration & PATH_GENERATION_MASK;
                                            guideStoredScratch.replayControl = uint4(
                                                    PATH_REPLAY_VERSION, 0u,
                                                    PATH_GUIDE_SCRATCH_SENTINEL_MAGIC,
                                                    guideSampleCopyResult);
                                        }
                                        DevicePtr<PathReservoir>(
                                                worldPush.pathShiftGuideScratchAddr)[pixelIndex] =
                                                guideStoredScratch;
                                        if ((guideSampleCopyResult
                                                        == PATH_GUIDE_SAMPLE_COPY_SELECTED
                                                    || guideSampleCopyResult
                                                        == PATH_GUIDE_SAMPLE_COPY_RETAINED)
                                                && guideStoredRootValid) {
                                            DevicePtr<PathSourceRoot>(worldPush
                                                    .pathShiftGuideRootScratchAddr)[pixelIndex] =
                                                    guideStoredRoot;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            InterlockedAdd(counters[PATH_RECEIVER_GUIDE_STORED_ELIGIBLE], 1u);
            float3 exactReceiverNormal =
                    receiverNormalRoughness.xyz / receiverNormalLength;
            // Use the exact Pass-B biased origin captured before lobe/endpoint selection. The
            // Pass-A guide position is appropriate for surface reprojection, but mixing it with
            // a Pass-B vertex is not an exact sampled-direction reconstruction.
            float3 sampledEdge = current.reconnectionVertex.xyz
                    - receiverRayOrigin.xyz;
            float baseDensity = pathMappingDiffuseDensity(exactReceiverNormal,
                    sampledEdge);
            float expectedPdf = max(receiverGuide.w * baseDensity, 1.0e-20);
            float3 expectedThroughput = receiverGuide.xyz / receiverGuide.w;
            // Throughput stores diffAlb / techniqueMass exactly. Recover the material-only mass
            // from the strongest channel so this check is independent of direction reconstruction.
            float dominantAlbedo = receiverGuide.x;
            float dominantThroughput = current.reconnectionThroughput.x;
            if (receiverGuide.y > dominantAlbedo) {
                dominantAlbedo = receiverGuide.y;
                dominantThroughput = current.reconnectionThroughput.y;
            }
            if (receiverGuide.z > dominantAlbedo) {
                dominantAlbedo = receiverGuide.z;
                dominantThroughput = current.reconnectionThroughput.z;
            }
            float storedMass = dominantAlbedo > 0.0 && dominantThroughput > 0.0
                    ? dominantAlbedo / dominantThroughput : 0.0;
            uint comparison = !pathReplayFloatMatches(storedMass, receiverGuide.w)
                    ? PATH_RECEIVER_GUIDE_MASS_MISMATCH
                    : !pathReplayColorMatches(expectedThroughput,
                            current.reconnectionThroughput.xyz)
                            ? PATH_RECEIVER_GUIDE_THROUGHPUT_MISMATCH
                            : !pathReplayFloatMatches(expectedPdf,
                                    current.reconnectionNormalPdf.w)
                                    ? PATH_RECEIVER_GUIDE_PDF_MISMATCH
                                    : PATH_RECEIVER_GUIDE_ACCEPTED;
            InterlockedAdd(counters[comparison], 1u);
        }
    }
    bool currentSampleUsable = current.weights.y > 0.0
            && current.sampleRadianceTarget.w > 0.0;
    bool sampleRescueCandidate = crossFrame && !currentSampleUsable;
    if (!currentSampleUsable) {
        if (!crossFrame) {
            InterlockedAdd(counters[PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
            return;
        }
        InterlockedAdd(counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_ELIGIBLE], 1u);
    }
    if (!currentEdgeAvailable) {
        if (crossFrame) {
            InterlockedAdd(counters[PATH_CROSS_FRAME_EDGE_BREAKDOWN_ELIGIBLE], 1u);
            uint edgeReason = !currentEdgeValid
                    ? PATH_CROSS_FRAME_EDGE_MISSING_VALID
                    : !currentEdgeDepth
                            ? PATH_CROSS_FRAME_EDGE_DEPTH_REJECT
                            : !currentEdgeEvent
                                    ? PATH_CROSS_FRAME_EDGE_EVENT_REJECT
                                    : !currentEdgeMapping
                                            ? PATH_CROSS_FRAME_EDGE_MAPPING_REJECT
                                            : !currentEdgePdf
                                                    ? PATH_CROSS_FRAME_EDGE_PDF_REJECT
                                                    : PATH_CROSS_FRAME_EDGE_FINITE_REJECT;
            InterlockedAdd(counters[edgeReason], 1u);
        }
        InterlockedAdd(counters[crossFrame
                ? (sampleRescueCandidate
                        ? PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT
                        : PATH_CROSS_FRAME_RECEIVER_EDGE_REJECT)
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        if (sampleRescueCandidate) {
            InterlockedAdd(counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_EDGE_REJECT], 1u);
        }
        return;
    }
    if (current.identity.z != mapped.identity.z) {
        InterlockedAdd(counters[crossFrame
                ? (sampleRescueCandidate
                        ? PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT
                        : PATH_CROSS_FRAME_RECEIVER_TOPOLOGY_REJECT)
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        if (sampleRescueCandidate) {
            InterlockedAdd(
                    counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_TOPOLOGY_REJECT], 1u);
        }
        return;
    }
    if (pathMetadataDepth(current.identity.w)
            != pathMetadataDepth(mapped.identity.w)) {
        InterlockedAdd(counters[crossFrame
                ? (sampleRescueCandidate
                        ? PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT
                        : PATH_CROSS_FRAME_RECEIVER_DEPTH_REJECT)
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        if (sampleRescueCandidate) {
            InterlockedAdd(counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_DEPTH_REJECT], 1u);
        }
        return;
    }
    if (pathMetadataTransport(current.identity.w)
            != pathMetadataTransport(mapped.identity.w)) {
        InterlockedAdd(counters[crossFrame
                ? (sampleRescueCandidate
                        ? PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT
                        : PATH_CROSS_FRAME_RECEIVER_TRANSPORT_REJECT)
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        if (sampleRescueCandidate) {
            InterlockedAdd(
                    counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_TRANSPORT_REJECT], 1u);
        }
        return;
    }

    float3 receiverNormal =
            receiverNormalRoughness.xyz / receiverNormalLength;
    float3 sourceNormal = sourceNormalRoughness.xyz / sourceNormalLength;
    float relativeDepth = abs(length(sourcePositionMaterial.xyz)
            - length(receiverPositionMaterial.xyz))
            / max(length(receiverPositionMaterial.xyz), 1.0);
    float sourceFootprint = max(mapped.proposalShift.z, 1.0e-5);
    float receiverFootprint = max(current.proposalShift.z, 1.0e-5);
    if (dot(receiverNormal, sourceNormal) < PATH_SPATIAL_NORMAL_COS
            || relativeDepth > PATH_SPATIAL_RELATIVE_DEPTH
            || max(sourceFootprint, receiverFootprint)
                    / min(sourceFootprint, receiverFootprint)
                    > PATH_MAX_FOOTPRINT_RATIO) {
        InterlockedAdd(counters[crossFrame
                ? (sampleRescueCandidate
                        ? PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT
                        : PATH_CROSS_FRAME_RECEIVER_FOOTPRINT_REJECT)
                : PATH_MAPPING_REPLAY_RECEIVER_REJECT], 1u);
        if (sampleRescueCandidate) {
            InterlockedAdd(
                    counters[PATH_CROSS_FRAME_SAMPLE_RESCUE_FOOTPRINT_REJECT], 1u);
        }
        return;
    }
    if (sampleRescueCandidate) {
        // This is a policy A/B count only. Keep the strict terminal category and do not trace,
        // select, or persist the hypothetical relaxed sample.
        InterlockedAdd(counters[PATH_CROSS_FRAME_RECEIVER_SAMPLE_REJECT], 1u);
        InterlockedAdd(counters[PATH_CROSS_FRAME_SAMPLE_RESCUED], 1u);
        return;
    }

    float3 sharedVertex = replay.reconnectionVertex;
    float3 sharedNormalValue = replay.reconnectionNormal;
    float sharedNormalLength = length(sharedNormalValue);
    float3 sourceEdge = sharedVertex - sourcePositionMaterial.xyz;
    float3 receiverEdge = sharedVertex - receiverPositionMaterial.xyz;
    float sourceDistance = length(sourceEdge);
    float receiverDistance = length(receiverEdge);
    if (sharedNormalLength < 0.5 || sourceDistance <= 1.0e-4
            || receiverDistance <= 1.0e-4) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_GEOMETRY_REJECT
                : PATH_MAPPING_REPLAY_GEOMETRY_REJECT], 1u);
        return;
    }
    float3 sharedNormal = sharedNormalValue / sharedNormalLength;
    float sourceCosine =
            abs(dot(sharedNormal, -sourceEdge / sourceDistance));
    float receiverCosine =
            abs(dot(sharedNormal, receiverEdge / receiverDistance));
    float geometryJacobian =
            receiverCosine * sourceDistance * sourceDistance
            / max(sourceCosine * receiverDistance * receiverDistance, 1.0e-8);
    if (!isfinite(geometryJacobian) || geometryJacobian <= 0.0) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_GEOMETRY_REJECT
                : PATH_MAPPING_REPLAY_GEOMETRY_REJECT], 1u);
        return;
    }

    float currentDensity = pathMappingDiffuseDensity(receiverNormal,
            current.reconnectionVertex.xyz - receiverPositionMaterial.xyz);
    float shiftedDensity =
            pathMappingDiffuseDensity(receiverNormal, receiverEdge);
    float sourceDensity =
            pathMappingDiffuseDensity(sourceNormal, sourceEdge);
    float currentTechniqueMass =
            current.reconnectionNormalPdf.w / currentDensity;
    float sourceTechniqueMass =
            replay.reconnectionSourcePdf / sourceDensity;
    float receiverPdf = currentTechniqueMass * shiftedDensity;
    float pssJacobian =
            geometryJacobian * receiverPdf / replay.reconnectionSourcePdf;
    if (!isfinite(currentDensity) || !isfinite(shiftedDensity)
            || !isfinite(sourceDensity) || currentDensity <= 0.0
            || shiftedDensity <= 0.0 || sourceDensity <= 0.0
            || !isfinite(currentTechniqueMass)
            || !isfinite(sourceTechniqueMass)
            || currentTechniqueMass <= 0.0 || sourceTechniqueMass <= 0.0
             || !isfinite(receiverPdf) || receiverPdf <= 0.0
             || !isfinite(pssJacobian) || pssJacobian <= 0.0
             || (!crossFrame && (!pathReplayFloatMatches(receiverPdf,
                         mapped.reconnectionNormalPdf.w)
                     || !pathReplayFloatMatches(pssJacobian, mapped.proposalShift.y)
                     || !pathReplayColorMatches(current.reconnectionThroughput.xyz,
                         mapped.reconnectionThroughput.xyz)))) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_PDF_REJECT
                : PATH_MAPPING_REPLAY_PDF_REJECT], 1u);
        return;
    }

    float side = dot(receiverNormal, receiverEdge) >= 0.0 ? 1.0 : -1.0;
    float3 shadowOrigin = receiverPositionMaterial.xyz
            + side * receiverNormal * SURF_BIAS;
    float3 toShared = sharedVertex - shadowOrigin;
    float shiftedDistance = length(toShared);
    if (shiftedDistance <= 1.0e-4) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_GEOMETRY_REJECT
                : PATH_MAPPING_REPLAY_GEOMETRY_REJECT], 1u);
        return;
    }
    VisibilityResult shiftedVisibility = visibility(shadowOrigin,
            toShared / shiftedDistance, shiftedDistance * 0.999);
    if (max(shiftedVisibility.transmittance.x,
            max(shiftedVisibility.transmittance.y,
                    shiftedVisibility.transmittance.z)) <= 0.0) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_VISIBILITY_REJECT
                : PATH_MAPPING_REPLAY_VISIBILITY_REJECT], 1u);
        return;
    }

    float3 sourceRadiance = replay.canonicalRadiance;
    float3 sourceThroughput = replay.reconnectionThroughput;
    // Persistent remapping always rebuilds the direct source -> current-receiver shift. The old
    // receiver throughput/Jacobian are validation metadata, never factors to compose.
    float3 receiverThroughput = crossFrame
            ? current.reconnectionThroughput.xyz : mapped.reconnectionThroughput.xyz;
    bool invalidSourceChannel = any(sourceRadiance > float3(0.0)
            && sourceThroughput <= float3(0.0));
    float3 throughputRatio = float3(
            pathMappingThroughputRatio(
                    sourceThroughput.x, receiverThroughput.x),
            pathMappingThroughputRatio(
                    sourceThroughput.y, receiverThroughput.y),
            pathMappingThroughputRatio(
                    sourceThroughput.z, receiverThroughput.z));
    float3 shiftedRadiance = sourceRadiance * throughputRatio
            * shiftedVisibility.transmittance;
    float shiftedTarget = pathLuminance(shiftedRadiance);
    if (invalidSourceChannel
            || !all(isfinite(sourceRadiance))
            || !all(isfinite(sourceThroughput))
            || !all(isfinite(receiverThroughput))
            || !all(isfinite(throughputRatio))
            || !all(isfinite(shiftedRadiance))
            || any(sourceRadiance < float3(0.0))
            || any(sourceThroughput < float3(0.0))
            || any(receiverThroughput < float3(0.0))
             || any(throughputRatio < float3(0.0))
             || any(shiftedRadiance < float3(0.0))
             || shiftedTarget <= 0.0
             || (!crossFrame && (!pathReplayColorMatches(
                         shiftedRadiance, mapped.sampleRadianceTarget.xyz)
                     || !pathReplayFloatMatches(
                         shiftedTarget, mapped.sampleRadianceTarget.w)
                     || !pathReplayColorMatches(
                         shiftedRadiance, mapped.canonicalRadianceTarget.xyz)
                     || !pathReplayFloatMatches(
                         shiftedTarget, mapped.canonicalRadianceTarget.w)))) {
        InterlockedAdd(counters[crossFrame
                ? PATH_CROSS_FRAME_RADIANCE_REJECT
                : PATH_MAPPING_REPLAY_RADIANCE_REJECT], 1u);
        return;
    }

    InterlockedAdd(counters[crossFrame
            ? PATH_CROSS_FRAME_ACCEPTED : PATH_MAPPING_REPLAY_ACCEPTED], 1u);
}

// One-frame lifetime/replay audit for the paired guide reservoir and source-root scratch. The
// previous pair is read before the host clears the same buffers for the current frame. It validates
// receiver and source reprojection plus exact original-source replay, but never reconstructs a new
// mapping, changes a weight, writes history, or touches the output image.
static const uint PATH_BRANCH_SCRATCH_CAPTURE_CAPACITY = 4096u;
static const uint PATH_BRANCH_SCRATCH_CAPTURE_BYTE_OFFSET = 131072u;
static const uint PATH_SOURCE_ROOT_STRIDE_BYTES = 160u;
static const uint PATH_BRANCH_CANDIDATE_TAG_VALID_BIT = 0x80000000u;
static const uint PATH_BRANCH_CANDIDATE_TAG_MAPPING_SHIFT = 24u;

static uint64_t pathBranchCandidateTagAddress(uint64_t rootAddress, uint pixelCount) {
    return rootAddress + uint64_t(pixelCount) * uint64_t(PATH_SOURCE_ROOT_STRIDE_BYTES);
}

static uint pathBranchCandidateTagControl(uint generation, uint mappingKind) {
    return PATH_BRANCH_CANDIDATE_TAG_VALID_BIT
            | ((mappingKind & 0x0Fu) << PATH_BRANCH_CANDIDATE_TAG_MAPPING_SHIFT)
            | (generation & PATH_GENERATION_MASK);
}

struct PathBranchScratchCapture {
    uint4 metadata; // pixel index, mapping kind, frame index, history generation
    PathReservoir reservoir;
    PathSourceRoot root;
};

static bool pathReservoirBitsMatch(PathReservoir a, PathReservoir b) {
    return all(asuint(a.sampleRadianceTarget) == asuint(b.sampleRadianceTarget))
            && all(asuint(a.proposalShift) == asuint(b.proposalShift))
            && all(asuint(a.proposalComponents) == asuint(b.proposalComponents))
            && all(asuint(a.canonicalRadianceTarget) == asuint(b.canonicalRadianceTarget))
            && all(asuint(a.weights) == asuint(b.weights))
            && all(a.identity == b.identity)
            && all(a.replay == b.replay)
            && all(a.replayControl == b.replayControl)
            && all(asuint(a.reconnectionVertex) == asuint(b.reconnectionVertex))
            && all(asuint(a.reconnectionNormalPdf) == asuint(b.reconnectionNormalPdf))
            && all(asuint(a.reconnectionThroughput) == asuint(b.reconnectionThroughput));
}

static bool packedPathSegmentBitsMatch(PackedPathSegment a, PackedPathSegment b) {
    return all(asuint(a.ro) == asuint(b.ro))
            && a.rd == b.rd
            && a.throughput == b.throughput
            && a.currentExtinction == b.currentExtinction
            && a.outerExtinction == b.outerExtinction
            && a.mediumIors == b.mediumIors
            && a.rayCone == b.rayCone
            && a.seed == b.seed
            && a.pathFlags == b.pathFlags
            && a.nextRecord == b.nextRecord;
}

static bool pathSourceRootBitsMatch(PathSourceRoot a, PathSourceRoot b) {
    return all(asuint(a.sourcePositionMaterial) == asuint(b.sourcePositionMaterial))
            && all(asuint(a.sourceNormalRoughness) == asuint(b.sourceNormalRoughness))
            && all(asuint(a.receiverPositionMaterial) == asuint(b.receiverPositionMaterial))
            && all(asuint(a.receiverNormalRoughness) == asuint(b.receiverNormalRoughness))
            && packedPathSegmentBitsMatch(a.segment0, b.segment0)
            && packedPathSegmentBitsMatch(a.segment1, b.segment1);
}

static void pathBranchScratchStorageValidate() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW
            || (pc.historyFlags & PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_PASS_FLAG) == 0u
            || worldPush.pathShiftDiagnosticCounterAddr == 0u
            || worldPush.pathShiftDiagnosticPairAddr == 0u
            || worldPush.pathShiftBranchReservoirCurrentAddr == 0u
            || worldPush.pathShiftBranchRootCurrentAddr == 0u) {
        return;
    }
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint2 dispatchPixel = DispatchRaysIndex().xy;
    uint sampleIndex = dispatchPixel.y * dimensions.x + dispatchPixel.x;
    DevicePtr<uint> counters = DevicePtr<uint>(
            worldPush.pathShiftDiagnosticCounterAddr);
    uint capturedCount = min(counters[PATH_GUIDE_BRANCH_SCRATCH_CAPTURE_CURSOR],
            PATH_BRANCH_SCRATCH_CAPTURE_CAPACITY);
    if (sampleIndex >= capturedCount) return;

    InterlockedAdd(counters[PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_ATTEMPTED], 1u);
    ConstPtr<PathBranchScratchCapture> captures = ConstPtr<PathBranchScratchCapture>(
            worldPush.pathShiftDiagnosticPairAddr
                    + uint64_t(PATH_BRANCH_SCRATCH_CAPTURE_BYTE_OFFSET));
    PathBranchScratchCapture expected = captures[sampleIndex];
    uint pixelCount = dimensions.x * dimensions.y;
    uint currentGeneration = pc.historyGeneration & PATH_GENERATION_MASK;
    bool metadataValid = expected.metadata.x < pixelCount
            && (expected.metadata.y == PATH_MAPPING_IDENTITY
                    || expected.metadata.y == PATH_MAPPING_DIFFUSE_RECONNECTION)
            && expected.metadata.z == worldPush.frameIndex
            && expected.metadata.w == currentGeneration;
    if (!metadataValid) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_METADATA_REJECT], 1u);
        return;
    }

    PathReservoir stored = ConstPtr<PathReservoir>(
            worldPush.pathShiftBranchReservoirCurrentAddr)[expected.metadata.x];
    PathSourceRoot root = ConstPtr<PathSourceRoot>(
            worldPush.pathShiftBranchRootCurrentAddr)[expected.metadata.x];
    bool reservoirMatches = pathReservoirBitsMatch(stored, expected.reservoir);
    bool rootMatches = pathSourceRootBitsMatch(root, expected.root);
    InterlockedAdd(counters[reservoirMatches
            ? PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_RESERVOIR_MATCH
            : PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_RESERVOIR_MISMATCH], 1u);
    InterlockedAdd(counters[rootMatches
            ? PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_ROOT_MATCH
            : PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_ROOT_MISMATCH], 1u);
    if (!reservoirMatches || !rootMatches) {
        InterlockedAdd(counters[PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_PAIR_REJECT], 1u);
        return;
    }
    InterlockedAdd(counters[PATH_GUIDE_BRANCH_CANDIDATE_TAG_WRITE_ELIGIBLE], 1u);
    DevicePtr<uint2> candidateTags = DevicePtr<uint2>(pathBranchCandidateTagAddress(
            worldPush.pathShiftBranchRootCurrentAddr, pixelCount));
    candidateTags[expected.metadata.x] = uint2(worldPush.frameIndex,
            pathBranchCandidateTagControl(currentGeneration, expected.metadata.y));
    InterlockedAdd(counters[expected.metadata.y == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_SELECTED_ACCEPTED
            : PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_RETAINED_ACCEPTED], 1u);
    InterlockedAdd(counters[expected.metadata.y == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? PATH_GUIDE_BRANCH_CANDIDATE_TAG_SELECTED_WRITTEN
            : PATH_GUIDE_BRANCH_CANDIDATE_TAG_RETAINED_WRITTEN], 1u);
}

static void pathBranchScratchReplayValidate() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW
            || (pc.historyFlags & PATH_GUIDE_PREVIOUS_REPLAY_PASS_FLAG) == 0u
            || (pc.historyFlags & PATH_GUIDE_BRANCH_PREVIOUS_AVAILABLE_FLAG) == 0u
            || worldPush.pathShiftDiagnosticCounterAddr == 0u
            || worldPush.pathShiftBranchReservoirPreviousAddr == 0u
            || worldPush.pathShiftBranchRootPreviousAddr == 0u) {
        return;
    }
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    DevicePtr<uint> counters =
            DevicePtr<uint>(worldPush.pathShiftDiagnosticCounterAddr);
    InterlockedAdd(counters[PATH_GUIDE_BRANCH_SCRATCH_REPLAY_ATTEMPTED], 1u);

    float2 previousPixel = float2(pixel) + 0.5 + gMotion[int2(pixel)];
    int2 previousTexel = int2(floor(previousPixel));
    if (any(previousTexel < int2(0, 0))
            || any(previousTexel >= int2(dimensions))) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_REPLAY_RECEIVER_REPROJECTION_REJECT], 1u);
        return;
    }
    uint previousPixelIndex = uint(previousTexel.y) * dimensions.x
            + uint(previousTexel.x);
    uint pixelCount = dimensions.x * dimensions.y;
    uint currentGeneration = pc.historyGeneration & PATH_GENERATION_MASK;
    uint2 candidateTag = ConstPtr<uint2>(pathBranchCandidateTagAddress(
            worldPush.pathShiftBranchRootPreviousAddr, pixelCount))[previousPixelIndex];
    if ((candidateTag.y & PATH_BRANCH_CANDIDATE_TAG_VALID_BIT) == 0u) {
        InterlockedAdd(counters[PATH_GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_EMPTY], 1u);
        return;
    }
    PathReservoir stored = ConstPtr<PathReservoir>(
            worldPush.pathShiftBranchReservoirPreviousAddr)[previousPixelIndex];
    PathSourceRoot root = ConstPtr<PathSourceRoot>(
            worldPush.pathShiftBranchRootPreviousAddr)[previousPixelIndex];
    if (all(stored.replayControl == uint4(0u)) && stored.identity.w == 0u) {
        InterlockedAdd(counters[PATH_GUIDE_BRANCH_SCRATCH_REPLAY_EMPTY], 1u);
        return;
    }
    uint sourceIndex = stored.identity.y;
    uint mappingKind = pathMappingKind(stored);
    bool candidateTagValid = candidateTag.x == worldPush.frameIndex - 1u
            && candidateTag.y == pathBranchCandidateTagControl(
                    currentGeneration, mappingKind);
    if (!candidateTagValid) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_METADATA_REJECT], 1u);
        return;
    }
    InterlockedAdd(counters[mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? PATH_GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_SELECTED_ADMITTED
            : PATH_GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_RETAINED_ADMITTED], 1u);
    bool metadataValid = all(isfinite(stored.weights))
            && all(stored.weights > float4(0.0))
            && pathReservoirSampleMetadataReady(stored)
            && (stored.identity.w & PATH_GENERATION_MASK) == currentGeneration
            && sourceIndex < pixelCount
            && (mappingKind == PATH_MAPPING_IDENTITY
                    || mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION);
    bool rootFinite = all(isfinite(root.sourcePositionMaterial))
            && all(isfinite(root.sourceNormalRoughness))
            && all(isfinite(root.receiverPositionMaterial))
            && all(isfinite(root.receiverNormalRoughness))
            && all(isfinite(root.segment0.ro))
            && all(isfinite(root.segment1.ro));
    bool rootChainValid = stored.replayControl.y == 1u
            ? root.segment0.nextRecord == PATH_NO_NEXT
            : stored.replayControl.y == MAX_PATH_SEGMENTS
                    && root.segment0.nextRecord != PATH_NO_NEXT
                    && root.segment1.nextRecord == PATH_NO_NEXT;
    if (!metadataValid || !rootFinite || !rootChainValid) {
        InterlockedAdd(counters[PATH_GUIDE_BRANCH_SCRATCH_REPLAY_METADATA_REJECT], 1u);
        return;
    }

    float3 retainedRebase = worldPush.camOffset - worldPush.camDelta;
    float4 receiverPositionMaterial = gRestirPositionMaterial[int2(pixel)];
    float4 receiverNormalRoughness = gRestirNormalRoughness[int2(pixel)];
    float4 retainedReceiverPositionMaterial = root.receiverPositionMaterial;
    retainedReceiverPositionMaterial.xyz += retainedRebase;
    if (!pathReplayPositionMatches(retainedReceiverPositionMaterial.xyz,
                receiverPositionMaterial.xyz)
            || !pathReplayColorMatches(root.receiverNormalRoughness.xyz,
                receiverNormalRoughness.xyz)
            || !pathReplayFloatMatches(root.receiverNormalRoughness.w,
                receiverNormalRoughness.w)
            || uint(max(retainedReceiverPositionMaterial.w, 0.0))
                    != uint(max(receiverPositionMaterial.w, 0.0))) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_REPLAY_RECEIVER_SURFACE_REJECT], 1u);
        return;
    }

    uint2 previousSourcePixel = uint2(sourceIndex % dimensions.x,
            sourceIndex / dimensions.x);
    float4 retainedSourcePositionMaterial = root.sourcePositionMaterial;
    retainedSourcePositionMaterial.xyz += retainedRebase;
    float3 currentSourceCamRelative = root.sourcePositionMaterial.xyz
            - worldPush.camDelta;
    float4 sourceClip = mul(worldPush.curViewProj,
            float4(currentSourceCamRelative, 1.0));
    bool sourceMotionFound = false;
    bool sourceReprojected = false;
    int2 sourceTexel = int2(previousSourcePixel);
    if (sourceClip.w > 0.0 && all(isfinite(sourceClip))) {
        float2 sourceNdc = sourceClip.xy / sourceClip.w;
        int2 projectedSourceTexel = int2(floor(
                (sourceNdc * 0.5 + 0.5) * float2(dimensions)));
        for (int oy = -1; oy <= 1 && !sourceReprojected; ++oy) {
            for (int ox = -1; ox <= 1 && !sourceReprojected; ++ox) {
                int2 candidateTexel = projectedSourceTexel + int2(ox, oy);
                if (any(candidateTexel < int2(0, 0))
                        || any(candidateTexel >= int2(dimensions))) continue;
                int2 candidatePreviousTexel = int2(floor(float2(candidateTexel)
                        + 0.5 + gMotion[candidateTexel]));
                if (any(candidatePreviousTexel != int2(previousSourcePixel))) continue;
                sourceMotionFound = true;
                float4 candidatePosition = gRestirPositionMaterial[candidateTexel];
                float4 candidateNormal = gRestirNormalRoughness[candidateTexel];
                if (pathReplayPositionMatches(retainedSourcePositionMaterial.xyz,
                            candidatePosition.xyz)
                        && pathReplayColorMatches(root.sourceNormalRoughness.xyz,
                            candidateNormal.xyz)
                        && pathReplayFloatMatches(root.sourceNormalRoughness.w,
                            candidateNormal.w)
                        && uint(max(retainedSourcePositionMaterial.w, 0.0))
                                == uint(max(candidatePosition.w, 0.0))) {
                    sourceTexel = candidateTexel;
                    sourceReprojected = true;
                }
            }
        }
    }
    if (!sourceMotionFound) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_REPROJECTION_REJECT], 1u);
        return;
    }
    if (!sourceReprojected) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_SURFACE_REJECT], 1u);
        return;
    }

    PathReplayAggregate replay = pathReplayAggregateEmpty();
    uint expectedSegments = stored.replayControl.y;
    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
        if (leaf >= expectedSegments) break;
        PackedPathSegment packed = leaf == 0u ? root.segment0 : root.segment1;
        packed.ro += retainedRebase;
        PathSegment segment = unpackPathSegment(packed);
        uint replaySeed = leaf == 0u ? stored.replay.x : stored.replay.z;
        uint proposalReplaySeed = leaf == 0u ? stored.replay.y : stored.replay.w;
        PathTraceResult traced = tracePathSeeded(
                segment, uint2(sourceTexel), true, replaySeed, proposalReplaySeed);
        pathReplayAccumulate(replay, traced, leaf);
    }
    bool replayMatches = mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? pathMappingSourceReplayMatches(stored, replay)
            : pathReplayCompare(stored, replay) == 0u;
    if (!replayMatches) {
        InterlockedAdd(counters[
                PATH_GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_REPLAY_REJECT], 1u);
        return;
    }
    InterlockedAdd(counters[mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? PATH_GUIDE_BRANCH_SCRATCH_REPLAY_SELECTED_ACCEPTED
            : PATH_GUIDE_BRANCH_SCRATCH_REPLAY_RETAINED_ACCEPTED], 1u);
}

public void pathGuidePreviousReplayValidate() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW
            || (pc.historyFlags & PATH_GUIDE_PREVIOUS_REPLAY_PASS_FLAG) == 0u
            || worldPush.pathShiftDiagnosticCounterAddr == 0u
            || worldPush.pathShiftGuideScratchAddr == 0u
            || worldPush.pathShiftGuideRootScratchAddr == 0u
            || pc.pathReservoirAddr == 0u) {
        return;
    }

    pathBranchScratchReplayValidate();

    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    uint pixelCount = dimensions.x * dimensions.y;
    DevicePtr<uint> counters =
            DevicePtr<uint>(worldPush.pathShiftDiagnosticCounterAddr);
    InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_REPLAY_ATTEMPTED], 1u);

    if ((pc.historyFlags & PATH_GUIDE_PREVIOUS_AVAILABLE_FLAG) == 0u) {
        InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_REPLAY_LIFECYCLE_REJECT], 1u);
        return;
    }

    float2 previousPixel = float2(pixel) + 0.5 + gMotion[int2(pixel)];
    int2 previousTexel = int2(floor(previousPixel));
    if (any(previousTexel < int2(0, 0))
            || any(previousTexel >= int2(dimensions))) {
        InterlockedAdd(counters[
                PATH_GUIDE_PREVIOUS_REPLAY_RECEIVER_REPROJECTION_REJECT], 1u);
        return;
    }
    uint previousPixelIndex = uint(previousTexel.y) * dimensions.x
            + uint(previousTexel.x);
    PathReservoir stored = ConstPtr<PathReservoir>(
            worldPush.pathShiftGuideScratchAddr)[previousPixelIndex];
    if (all(stored.replayControl == uint4(0u)) && stored.identity.w == 0u) {
        InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_REPLAY_EMPTY], 1u);
        return;
    }

    uint currentGeneration = pc.historyGeneration & PATH_GENERATION_MASK;
    if (stored.replayControl.y == 0u) {
        uint terminal = stored.replayControl.w;
        bool sentinelValid = stored.replayControl.x == PATH_REPLAY_VERSION
                && stored.replayControl.z == PATH_GUIDE_SCRATCH_SENTINEL_MAGIC
                && stored.identity.w == currentGeneration
                && (terminal == PATH_GUIDE_SAMPLE_COPY_EMPTY
                        || terminal == PATH_GUIDE_SAMPLE_COPY_METADATA_REJECT
                        || terminal == PATH_GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT);
        InterlockedAdd(counters[sentinelValid
                ? PATH_GUIDE_PREVIOUS_REPLAY_EMPTY
                : PATH_GUIDE_PREVIOUS_REPLAY_METADATA_REJECT], 1u);
        return;
    }

    uint sourceIndex = stored.identity.y;
    uint mappingKind = pathMappingKind(stored);
    bool weightsValid = all(isfinite(stored.weights))
            && all(stored.weights > float4(0.0));
    bool metadataValid = weightsValid && pathReservoirSampleMetadataReady(stored)
            && (stored.identity.w & PATH_GENERATION_MASK) == currentGeneration
            && sourceIndex < pixelCount
            && (mappingKind == PATH_MAPPING_IDENTITY
                    || mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION);
    PathSourceRoot root = ConstPtr<PathSourceRoot>(
            worldPush.pathShiftGuideRootScratchAddr)[previousPixelIndex];
    bool rootFinite = all(isfinite(root.sourcePositionMaterial))
            && all(isfinite(root.sourceNormalRoughness))
            && all(isfinite(root.receiverPositionMaterial))
            && all(isfinite(root.receiverNormalRoughness))
            && all(isfinite(root.segment0.ro))
            && all(isfinite(root.segment1.ro));
    bool rootChainValid = stored.replayControl.y == 1u
            ? root.segment0.nextRecord == PATH_NO_NEXT
            : stored.replayControl.y == MAX_PATH_SEGMENTS
                    && root.segment0.nextRecord != PATH_NO_NEXT
                    && root.segment1.nextRecord == PATH_NO_NEXT;
    if (!metadataValid || !rootFinite || !rootChainValid) {
        InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_REPLAY_METADATA_REJECT], 1u);
        return;
    }

    float3 retainedRebase = worldPush.camOffset - worldPush.camDelta;
    float4 retainedReceiverPositionMaterial = root.receiverPositionMaterial;
    retainedReceiverPositionMaterial.xyz += retainedRebase;
    float4 receiverPositionMaterial = gRestirPositionMaterial[int2(pixel)];
    float4 receiverNormalRoughness = gRestirNormalRoughness[int2(pixel)];
    if (!pathReplayPositionMatches(retainedReceiverPositionMaterial.xyz,
                receiverPositionMaterial.xyz)
            || !pathReplayColorMatches(root.receiverNormalRoughness.xyz,
                receiverNormalRoughness.xyz)
            || !pathReplayFloatMatches(root.receiverNormalRoughness.w,
                receiverNormalRoughness.w)
            || uint(max(retainedReceiverPositionMaterial.w, 0.0))
                    != uint(max(receiverPositionMaterial.w, 0.0))) {
        InterlockedAdd(counters[
                PATH_GUIDE_PREVIOUS_REPLAY_RECEIVER_SURFACE_REJECT], 1u);
        return;
    }

    uint2 previousSourcePixel = uint2(sourceIndex % dimensions.x,
            sourceIndex / dimensions.x);
    float4 retainedSourcePositionMaterial = root.sourcePositionMaterial;
    retainedSourcePositionMaterial.xyz += retainedRebase;
    float3 currentSourceCamRelative = root.sourcePositionMaterial.xyz
            - worldPush.camDelta;
    float4 sourceClip = mul(worldPush.curViewProj,
            float4(currentSourceCamRelative, 1.0));
    bool sourceMotionFound = false;
    bool sourceReprojected = false;
    int2 sourceTexel = int2(previousSourcePixel);
    if (sourceClip.w > 0.0 && all(isfinite(sourceClip))) {
        float2 sourceNdc = sourceClip.xy / sourceClip.w;
        int2 projectedSourceTexel = int2(floor(
                (sourceNdc * 0.5 + 0.5) * float2(dimensions)));
        for (int oy = -1; oy <= 1 && !sourceReprojected; ++oy) {
            for (int ox = -1; ox <= 1 && !sourceReprojected; ++ox) {
                int2 candidateTexel = projectedSourceTexel + int2(ox, oy);
                if (any(candidateTexel < int2(0, 0))
                        || any(candidateTexel >= int2(dimensions))) continue;
                int2 candidatePreviousTexel = int2(floor(float2(candidateTexel)
                        + 0.5 + gMotion[candidateTexel]));
                if (any(candidatePreviousTexel != int2(previousSourcePixel))) continue;
                sourceMotionFound = true;
                float4 candidatePosition = gRestirPositionMaterial[candidateTexel];
                float4 candidateNormal = gRestirNormalRoughness[candidateTexel];
                if (pathReplayPositionMatches(retainedSourcePositionMaterial.xyz,
                            candidatePosition.xyz)
                        && pathReplayColorMatches(root.sourceNormalRoughness.xyz,
                            candidateNormal.xyz)
                        && pathReplayFloatMatches(root.sourceNormalRoughness.w,
                            candidateNormal.w)
                        && uint(max(retainedSourcePositionMaterial.w, 0.0))
                                == uint(max(candidatePosition.w, 0.0))) {
                    sourceTexel = candidateTexel;
                    sourceReprojected = true;
                }
            }
        }
    }
    if (!sourceMotionFound) {
        InterlockedAdd(counters[
                PATH_GUIDE_PREVIOUS_REPLAY_SOURCE_REPROJECTION_REJECT], 1u);
        return;
    }
    if (!sourceReprojected) {
        InterlockedAdd(counters[
                PATH_GUIDE_PREVIOUS_REPLAY_SOURCE_SURFACE_REJECT], 1u);
        return;
    }
    float4 currentSourcePositionMaterial =
            gRestirPositionMaterial[sourceTexel];
    float4 currentSourceNormalRoughness =
            gRestirNormalRoughness[sourceTexel];

    PathReplayAggregate replay = pathReplayAggregateEmpty();
    uint expectedSegments = stored.replayControl.y;
    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
        if (leaf >= expectedSegments) break;
        PackedPathSegment packed = leaf == 0u ? root.segment0 : root.segment1;
        packed.ro += retainedRebase;
        PathSegment segment = unpackPathSegment(packed);
        uint replaySeed = leaf == 0u ? stored.replay.x : stored.replay.z;
        uint proposalReplaySeed = leaf == 0u ? stored.replay.y : stored.replay.w;
        PathTraceResult traced = tracePathSeeded(
                segment, uint2(sourceTexel), true, replaySeed, proposalReplaySeed);
        pathReplayAccumulate(replay, traced, leaf);
    }
    bool replayMatches = mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? pathMappingSourceReplayMatches(stored, replay)
            : pathReplayCompare(stored, replay) == 0u;
    if (!replayMatches) {
        InterlockedAdd(counters[
                PATH_GUIDE_PREVIOUS_REPLAY_SOURCE_REPLAY_REJECT], 1u);
        return;
    }
    if (mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION) {
        // This is the next bounded gate after exact replay: reconstruct only the already-proven
        // receiver-aware diffuse mapping chain from the accepted previous pair. Every result below
        // is a counter-only diagnostic. It does not select, weight, write scratch/history, or feed
        // the estimator.
        InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_REMAP_ELIGIBLE], 1u);
        bool receiverGuideValid = false;
        float4 receiverGuide = float4(0.0);
        float4 receiverRayOrigin = float4(0.0);
        if (worldPush.pathShiftReceiverGuideAddr != 0u) {
            ConstPtr<float4> receiverGuides =
                    ConstPtr<float4>(worldPush.pathShiftReceiverGuideAddr);
            receiverGuide = receiverGuides[pixelIndex * 2u];
            receiverRayOrigin = receiverGuides[pixelIndex * 2u + 1u];
            receiverGuideValid = all(isfinite(receiverGuide))
                    && all(isfinite(receiverRayOrigin))
                    && all(receiverGuide.xyz >= float3(0.0))
                    && receiverGuide.w > 0.0 && receiverGuide.w <= 1.0
                    && receiverRayOrigin.w == 1.0;
        }
        if (!receiverGuideValid) {
            InterlockedAdd(counters[
                    PATH_GUIDE_PREVIOUS_REMAP_RECEIVER_GUIDE_REJECT], 1u);
        } else {
            float receiverNormalLength = length(receiverNormalRoughness.xyz);
            float sourceNormalLength = length(root.sourceNormalRoughness.xyz);
            float3 sourceEdge = replay.reconnectionVertex
                    - retainedSourcePositionMaterial.xyz;
            float3 receiverEdge = replay.reconnectionVertex
                    - receiverPositionMaterial.xyz;
            float sourceDistance = length(sourceEdge);
            float receiverDistance = length(receiverEdge);
            bool geometryValid = replay.reconnectionValid != 0u
                    && replay.reconnectionDepth == 1u
                    && replay.reconnectionEventKind == PATH_RECONNECTION_EVENT_DIFFUSE
                    && receiverNormalLength >= 0.5
                    && sourceNormalLength >= 0.5
                    && length(replay.reconnectionNormal) >= 0.5
                    && sourceDistance > 1.0e-4
                    && receiverDistance > 1.0e-4;
            float geometryJacobian = 0.0;
            if (geometryValid) {
                float3 sharedNormal = replay.reconnectionNormal
                        / length(replay.reconnectionNormal);
                float sourceCosine = abs(dot(sharedNormal,
                        -sourceEdge / sourceDistance));
                float receiverCosine = abs(dot(sharedNormal,
                        receiverEdge / receiverDistance));
                geometryJacobian = receiverCosine * sourceDistance * sourceDistance
                        / max(sourceCosine * receiverDistance * receiverDistance, 1.0e-8);
                geometryValid = sourceCosine > 0.0
                        && receiverCosine > 0.0
                        && isfinite(geometryJacobian)
                        && geometryJacobian > 0.0;
            }
            if (!geometryValid) {
                InterlockedAdd(counters[
                        PATH_GUIDE_PREVIOUS_REMAP_GEOMETRY_REJECT], 1u);
            } else {
                float3 receiverNormal = receiverNormalRoughness.xyz / receiverNormalLength;
                float3 sampledEdge = replay.reconnectionVertex - receiverRayOrigin.xyz;
                float receiverPdf = receiverGuide.w
                        * pathMappingDiffuseDensity(receiverNormal, sampledEdge);
                float sourcePdf = replay.reconnectionSourcePdf;
                float pssJacobian = geometryJacobian * receiverPdf / sourcePdf;
                bool pdfValid = isfinite(receiverPdf) && receiverPdf > 0.0
                        && isfinite(sourcePdf) && sourcePdf > 0.0
                        && isfinite(pssJacobian) && pssJacobian > 0.0;
                if (!pdfValid) {
                    InterlockedAdd(counters[
                            PATH_GUIDE_PREVIOUS_REMAP_PDF_REJECT], 1u);
                } else {
                    float3 sourceRadiance = replay.canonicalRadiance;
                    float3 sourceThroughput = replay.reconnectionThroughput;
                    float3 receiverThroughput = receiverGuide.xyz / receiverGuide.w;
                    bool invalidSourceChannel = any(sourceRadiance > float3(0.0)
                            && sourceThroughput <= float3(0.0));
                    float3 throughputRatio = float3(
                            pathMappingThroughputRatio(sourceThroughput.x,
                                    receiverThroughput.x),
                            pathMappingThroughputRatio(sourceThroughput.y,
                                    receiverThroughput.y),
                            pathMappingThroughputRatio(sourceThroughput.z,
                                    receiverThroughput.z));
                    float3 shiftedRadiance = sourceRadiance * throughputRatio;
                    bool throughputValid = !invalidSourceChannel
                            && all(isfinite(sourceRadiance))
                            && all(isfinite(sourceThroughput))
                            && all(isfinite(receiverThroughput))
                            && all(isfinite(throughputRatio))
                            && all(isfinite(shiftedRadiance))
                            && all(sourceRadiance >= float3(0.0))
                            && all(sourceThroughput >= float3(0.0))
                            && all(receiverThroughput >= float3(0.0))
                            && all(throughputRatio >= float3(0.0))
                            && all(shiftedRadiance >= float3(0.0));
                    if (!throughputValid) {
                        InterlockedAdd(counters[
                                PATH_GUIDE_PREVIOUS_REMAP_THROUGHPUT_REJECT], 1u);
                    } else {
                        InterlockedAdd(counters[
                                PATH_GUIDE_PREVIOUS_REMAP_READY], 1u);
                        InterlockedAdd(counters[
                                PATH_GUIDE_PREVIOUS_VISIBILITY_ELIGIBLE], 1u);
                        float3 toShared = replay.reconnectionVertex - receiverRayOrigin.xyz;
                        float visibilityDistance = length(toShared);
                        if (!isfinite(visibilityDistance) || visibilityDistance <= 1.0e-4) {
                            InterlockedAdd(counters[
                                    PATH_GUIDE_PREVIOUS_VISIBILITY_INVALID], 1u);
                        } else {
                            VisibilityResult visibilityResult = visibility(
                                    receiverRayOrigin.xyz,
                                    toShared / visibilityDistance,
                                    visibilityDistance * 0.999);
                            float3 transmittance = visibilityResult.transmittance;
                            bool visibilityValid = all(isfinite(transmittance))
                                    && all(transmittance >= float3(0.0))
                                    && all(transmittance <= float3(1.0));
                            uint visibilityCategory = !visibilityValid
                                    ? PATH_GUIDE_PREVIOUS_VISIBILITY_INVALID
                                    : max(transmittance.x,
                                            max(transmittance.y, transmittance.z)) <= 0.0
                                            ? PATH_GUIDE_PREVIOUS_VISIBILITY_OCCLUDED
                                            : all(transmittance == float3(1.0))
                                                    ? PATH_GUIDE_PREVIOUS_VISIBILITY_CLEAR
                                                    : PATH_GUIDE_PREVIOUS_VISIBILITY_TINTED;
                            InterlockedAdd(counters[visibilityCategory], 1u);
                            if (visibilityValid) {
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_TARGET_ELIGIBLE], 1u);
                                float3 visibleRadiance = shiftedRadiance * transmittance;
                                float shiftedTarget = pathLuminance(visibleRadiance);
                                bool targetValid = all(isfinite(visibleRadiance))
                                        && all(visibleRadiance >= float3(0.0))
                                        && isfinite(shiftedTarget)
                                        && shiftedTarget >= 0.0;
                                uint targetCategory = !targetValid
                                        ? PATH_GUIDE_PREVIOUS_TARGET_INVALID
                                        : shiftedTarget > 0.0
                                                ? PATH_GUIDE_PREVIOUS_TARGET_POSITIVE
                                                : PATH_GUIDE_PREVIOUS_TARGET_ZERO;
                                InterlockedAdd(counters[targetCategory], 1u);

                                // Compare the reconstructed previous result with the exact lanes
                                // captured in the paired scratch record. These are independent
                                // reproducibility audits: they never alter mapping, selection,
                                // weights, history, or the ordinary estimator.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_STORED_METADATA_ELIGIBLE], 1u);
                                bool storedTargetValid =
                                        all(isfinite(stored.sampleRadianceTarget))
                                        && all(stored.sampleRadianceTarget >= float4(0.0));
                                bool targetMatchesStored = targetValid && storedTargetValid
                                        && pathReplayColorMatches(visibleRadiance,
                                                stored.sampleRadianceTarget.xyz)
                                        && pathReplayFloatMatches(shiftedTarget,
                                                stored.sampleRadianceTarget.w);
                                InterlockedAdd(counters[targetValid && storedTargetValid
                                        ? targetMatchesStored
                                                ? PATH_GUIDE_PREVIOUS_STORED_TARGET_MATCH
                                                : PATH_GUIDE_PREVIOUS_STORED_TARGET_MISMATCH
                                        : PATH_GUIDE_PREVIOUS_STORED_TARGET_INVALID], 1u);

                                bool storedThroughputValid =
                                        all(isfinite(stored.reconnectionThroughput.xyz))
                                        && all(stored.reconnectionThroughput.xyz
                                                >= float3(0.0));
                                bool throughputMatchesStored = storedThroughputValid
                                        && all(isfinite(receiverThroughput))
                                        && all(receiverThroughput >= float3(0.0))
                                        && pathReplayColorMatches(receiverThroughput,
                                                stored.reconnectionThroughput.xyz);
                                InterlockedAdd(counters[storedThroughputValid
                                        ? throughputMatchesStored
                                                ? PATH_GUIDE_PREVIOUS_STORED_THROUGHPUT_MATCH
                                                : PATH_GUIDE_PREVIOUS_STORED_THROUGHPUT_MISMATCH
                                        : PATH_GUIDE_PREVIOUS_STORED_THROUGHPUT_INVALID], 1u);

                                // Counter-only GRIS arithmetic audit. The source final weight and
                                // effective count come from the stored guide reservoir; the target
                                // and PSS Jacobian are the freshly reconstructed current-receiver
                                // values above. No RNG, selection, merge, or reservoir mutation is
                                // performed here.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_WEIGHT_ELIGIBLE], 1u);
                                float sourceFinalWeight = stored.weights.z;
                                float sourceEffectiveCount = min(
                                        stored.weights.y, PATH_TEMPORAL_MAX_SOURCE_COUNT);
                                bool weightTermsValid = targetValid
                                        && isfinite(shiftedTarget)
                                        && shiftedTarget >= 0.0
                                        && isfinite(sourceFinalWeight)
                                        && sourceFinalWeight >= 0.0
                                        && isfinite(sourceEffectiveCount)
                                        && sourceEffectiveCount >= 0.0
                                        && isfinite(pssJacobian)
                                        && pssJacobian > 0.0;
                                float mergeWeight = weightTermsValid
                                        ? shiftedTarget * sourceFinalWeight
                                                * sourceEffectiveCount * pssJacobian
                                        : 0.0;
                                bool mergeWeightPositive = weightTermsValid
                                        && isfinite(mergeWeight)
                                        && mergeWeight > 0.0;
                                bool mergeWeightZero = weightTermsValid
                                        && (!mergeWeightPositive);
                                uint weightCategory = !weightTermsValid
                                        || (!isfinite(mergeWeight) || mergeWeight < 0.0)
                                        ? PATH_GUIDE_PREVIOUS_WEIGHT_INVALID
                                        : mergeWeightPositive
                                                ? PATH_GUIDE_PREVIOUS_WEIGHT_POSITIVE
                                                : mergeWeightZero
                                                        ? PATH_GUIDE_PREVIOUS_WEIGHT_ZERO
                                                        : PATH_GUIDE_PREVIOUS_WEIGHT_INVALID;
                                InterlockedAdd(counters[weightCategory], 1u);

                                // Post-weight arithmetic audit for both possible reservoir
                                // payloads. We intentionally evaluate current-retained and
                                // previous-source final weights independently instead of selecting
                                // one with RNG. No reservoir lane is written.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_POST_WEIGHT_ELIGIBLE], 1u);
                                PathReservoir current = ConstPtr<PathReservoir>(
                                        pc.pathReservoirAddr)[pixelIndex];
                                bool currentTermsValid = all(isfinite(current.weights))
                                        && all(current.weights >= float4(0.0))
                                        && isfinite(current.sampleRadianceTarget.w)
                                        && current.sampleRadianceTarget.w >= 0.0;
                                bool nextInputsValid = currentTermsValid && weightTermsValid
                                        && isfinite(mergeWeight) && mergeWeight >= 0.0;
                                float nextWeightSum = nextInputsValid
                                        ? min(current.weights.x + mergeWeight, 1.0e30)
                                        : 0.0;
                                float nextEffectiveCount = nextInputsValid
                                        ? min(current.weights.y + sourceEffectiveCount,
                                                16777216.0)
                                        : 0.0;
                                bool nextValid = nextInputsValid
                                        && isfinite(nextWeightSum) && nextWeightSum >= 0.0
                                        && isfinite(nextEffectiveCount)
                                        && nextEffectiveCount >= 0.0;
                                InterlockedAdd(counters[nextValid
                                        ? PATH_GUIDE_PREVIOUS_POST_WEIGHT_NEXT_VALID
                                        : PATH_GUIDE_PREVIOUS_POST_WEIGHT_NEXT_INVALID], 1u);

                                bool currentFinalValid = false;
                                bool currentDenominatorValid = false;
                                float currentDenominator = 0.0;
                                float currentFinalWeight = 0.0;
                                if (nextValid) {
                                    if (nextWeightSum == 0.0) {
                                        currentFinalValid = true;
                                        currentDenominatorValid = true;
                                    } else if (nextEffectiveCount > 0.0
                                            && current.sampleRadianceTarget.w > 0.0) {
                                        currentDenominator = nextEffectiveCount
                                                * current.sampleRadianceTarget.w;
                                        currentFinalWeight = nextWeightSum / currentDenominator;
                                        currentDenominatorValid = isfinite(currentDenominator)
                                                && currentDenominator > 0.0
                                                && isfinite(currentFinalWeight)
                                                && currentFinalWeight > 0.0;
                                        currentFinalValid = currentDenominatorValid;
                                    }
                                }
                                InterlockedAdd(counters[currentFinalValid
                                        ? PATH_GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_VALID
                                        : PATH_GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_INVALID], 1u);

                                bool sourceFinalValid = false;
                                bool sourceDenominatorValid = false;
                                float sourceDenominator = 0.0;
                                float sourceFinalWeightResult = 0.0;
                                if (nextValid) {
                                    if (nextWeightSum == 0.0) {
                                        sourceFinalValid = true;
                                        sourceDenominatorValid = true;
                                    } else if (nextEffectiveCount > 0.0
                                            && shiftedTarget > 0.0) {
                                        sourceDenominator = nextEffectiveCount
                                                * shiftedTarget;
                                        sourceFinalWeightResult = nextWeightSum / sourceDenominator;
                                        sourceDenominatorValid = isfinite(sourceDenominator)
                                                && sourceDenominator > 0.0
                                                && isfinite(sourceFinalWeightResult)
                                                && sourceFinalWeightResult > 0.0;
                                        sourceFinalValid = sourceDenominatorValid;
                                    }
                                }
                                InterlockedAdd(counters[sourceFinalValid
                                        ? PATH_GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_VALID
                                        : PATH_GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_INVALID], 1u);

                                // Counter-only selection-probability denominator audit for both
                                // hypothetical payload branches. The probability itself is shared
                                // by reservoir update, while each branch still has an independent
                                // final-weight denominator. No RNG draw or lane mutation occurs.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_SELECTION_PROBABILITY_ELIGIBLE], 1u);
                                bool selectionDenominatorValid = nextValid
                                        && isfinite(nextWeightSum)
                                        && nextWeightSum >= 0.0;
                                InterlockedAdd(counters[selectionDenominatorValid
                                        ? PATH_GUIDE_PREVIOUS_SELECTION_DENOMINATOR_VALID
                                        : PATH_GUIDE_PREVIOUS_SELECTION_DENOMINATOR_INVALID], 1u);
                                float selectionProbability = selectionDenominatorValid
                                        && mergeWeight > 0.0 && nextWeightSum > 0.0
                                        ? mergeWeight / nextWeightSum
                                        : 0.0;
                                bool selectionProbabilityArithmeticValid =
                                        selectionDenominatorValid
                                        && isfinite(selectionProbability)
                                        && selectionProbability >= 0.0;
                                uint selectionProbabilityCategory =
                                        !selectionProbabilityArithmeticValid
                                        ? PATH_GUIDE_PREVIOUS_SELECTION_PROBABILITY_INVALID
                                        : selectionProbability > 1.0
                                                ? PATH_GUIDE_PREVIOUS_SELECTION_PROBABILITY_HIGH_REJECT
                                                : PATH_GUIDE_PREVIOUS_SELECTION_PROBABILITY_VALID;
                                InterlockedAdd(counters[selectionProbabilityCategory], 1u);
                                InterlockedAdd(counters[currentDenominatorValid
                                        ? PATH_GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_VALID
                                        : PATH_GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_INVALID], 1u);
                                InterlockedAdd(counters[sourceDenominatorValid
                                        ? PATH_GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_VALID
                                        : PATH_GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_INVALID], 1u);

                                // Overflow-stable A/B for the exact same weights. It deliberately
                                // bypasses the stored weight-sum cap only for the relative selection
                                // ratio and still performs no RNG draw or reservoir mutation.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_STABLE_SELECTION_ELIGIBLE], 1u);
                                float stableSelectionProbability = nextInputsValid
                                        ? pathStableSelectionProbability(
                                                current.weights.x, mergeWeight)
                                        : 0.0;
                                bool stableSelectionValid = nextInputsValid
                                        && isfinite(stableSelectionProbability)
                                        && stableSelectionProbability >= 0.0
                                        && stableSelectionProbability <= 1.0;
                                uint stableSelectionCategory = !stableSelectionValid
                                        ? PATH_GUIDE_PREVIOUS_STABLE_SELECTION_INVALID
                                        : stableSelectionProbability > 0.0
                                                ? PATH_GUIDE_PREVIOUS_STABLE_SELECTION_POSITIVE
                                                : PATH_GUIDE_PREVIOUS_STABLE_SELECTION_ZERO;
                                InterlockedAdd(counters[stableSelectionCategory], 1u);

                                // Counter-only Bernoulli partition for the stable ratio. The local
                                // hash is independent of both replay streams; this deliberately
                                // records selected/retained/invalid outcomes without copying a
                                // payload, composing a mapping, or writing history/estimator state.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_STABLE_BERNOULLI_ELIGIBLE], 1u);
                                uint stableBernoulliSeed = pathHash(pixelIndex
                                        ^ worldPush.frameIndex * 747796405u
                                        ^ 0xD1B54A35u);
                                float stableBernoulliRandom = float(stableBernoulliSeed >> 8u)
                                        * (1.0 / 16777216.0);
                                bool stableBernoulliValid = stableSelectionValid
                                        && isfinite(stableBernoulliRandom)
                                        && stableBernoulliRandom >= 0.0
                                        && stableBernoulliRandom < 1.0;
                                bool stableBernoulliSelected = stableBernoulliValid
                                        && stableBernoulliRandom < stableSelectionProbability;
                                uint stableBernoulliCategory = !stableBernoulliValid
                                        ? PATH_GUIDE_PREVIOUS_STABLE_BERNOULLI_INVALID
                                        : stableBernoulliSelected
                                                ? PATH_GUIDE_PREVIOUS_STABLE_BERNOULLI_SELECTED
                                                : PATH_GUIDE_PREVIOUS_STABLE_BERNOULLI_RETAINED;
                                InterlockedAdd(counters[stableBernoulliCategory], 1u);

                                // Branch-specific final-state and payload readiness audit. The
                                // selected/retained branch is classified only after the stable
                                // Bernoulli result; no reservoir payload is copied, rewritten, or
                                // allowed to reach history or the ordinary estimator.
                                InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_BRANCH_ELIGIBLE],
                                        stableBernoulliValid ? 1u : 0u);
                                bool selectedFinalStateValid = stableBernoulliSelected
                                        && nextValid
                                        && sourceFinalValid
                                        && isfinite(sourceFinalWeightResult)
                                        && sourceFinalWeightResult > 0.0
                                        && isfinite(nextWeightSum)
                                        && nextWeightSum >= 0.0
                                        && isfinite(nextEffectiveCount)
                                        && nextEffectiveCount > 0.0
                                        && isfinite(shiftedTarget)
                                        && shiftedTarget > 0.0;
                                bool retainedFinalStateValid = stableBernoulliValid
                                        && !stableBernoulliSelected
                                        && nextValid
                                        && currentFinalValid
                                        && isfinite(currentFinalWeight)
                                        && currentFinalWeight > 0.0
                                        && isfinite(nextWeightSum)
                                        && nextWeightSum >= 0.0
                                        && isfinite(nextEffectiveCount)
                                        && nextEffectiveCount > 0.0
                                        && isfinite(current.sampleRadianceTarget.w)
                                        && current.sampleRadianceTarget.w > 0.0;
                                InterlockedAdd(counters[selectedFinalStateValid
                                        ? PATH_GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_VALID
                                        : PATH_GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_INVALID],
                                        stableBernoulliSelected ? 1u : 0u);
                                InterlockedAdd(counters[retainedFinalStateValid
                                        ? PATH_GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_VALID
                                        : PATH_GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_INVALID],
                                        stableBernoulliValid && !stableBernoulliSelected
                                                ? 1u : 0u);

                                bool selectedPayloadValid = selectedFinalStateValid
                                        && pathReservoirSampleMetadataReady(stored)
                                        && all(isfinite(stored.weights))
                                        && all(stored.weights >= float4(0.0))
                                        && all(isfinite(replay.canonicalRadiance))
                                        && all(replay.canonicalRadiance >= float3(0.0))
                                        && all(isfinite(replay.reconnectionThroughput))
                                        && all(replay.reconnectionThroughput >= float3(0.0));
                                bool retainedPayloadValid = retainedFinalStateValid
                                        && pathReservoirSampleMetadataReady(current)
                                        && all(isfinite(current.weights))
                                        && all(current.weights >= float4(0.0));
                                InterlockedAdd(counters[selectedPayloadValid
                                        ? PATH_GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_VALID
                                        : PATH_GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_INVALID],
                                        stableBernoulliSelected ? 1u : 0u);
                                InterlockedAdd(counters[retainedPayloadValid
                                        ? PATH_GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_VALID
                                        : PATH_GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_INVALID],
                                        stableBernoulliValid && !stableBernoulliSelected
                                                ? 1u : 0u);

                                // Audit the exact lane policy for a future branch record without
                                // constructing or copying either PathReservoir. A selected source
                                // must rewrite receiver-dependent lanes, preserve the original
                                // replay/proposal path, and install one direct mapping descriptor.
                                // A retained current sample must preserve its opaque payload. Only
                                // the shared weight lane is branch-dependent in both cases.
                                InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_LANE_ELIGIBLE],
                                        stableBernoulliValid ? 1u : 0u);
                                uint currentSourceIndex = uint(sourceTexel.y) * dimensions.x
                                        + uint(sourceTexel.x);
                                bool selectedRewriteReady = selectedPayloadValid
                                        && all(isfinite(visibleRadiance))
                                        && all(visibleRadiance >= float3(0.0))
                                        && isfinite(shiftedTarget) && shiftedTarget > 0.0
                                        && isfinite(stored.proposalShift.x)
                                        && stored.proposalShift.x > 0.0
                                        && isfinite(replay.footprint) && replay.footprint > 0.0
                                        && isfinite(pssJacobian) && pssJacobian > 0.0
                                        && currentSourceIndex < pixelCount
                                        && replay.canonicalValid != 0u
                                        && replay.reconnectionValid != 0u
                                        && replay.reconnectionDepth == 1u
                                        && replay.reconnectionEventKind
                                                == PATH_RECONNECTION_EVENT_DIFFUSE
                                        && all(isfinite(replay.reconnectionVertex))
                                        && all(isfinite(replay.reconnectionNormal))
                                        && isfinite(receiverPdf) && receiverPdf > 0.0
                                        && all(isfinite(receiverThroughput))
                                        && all(receiverThroughput >= float3(0.0))
                                        && pathPackMetadata(pc.historyGeneration,
                                                replay.depth, replay.transportMask)
                                                == pathPackMetadata(currentGeneration,
                                                        replay.depth, replay.transportMask);
                                InterlockedAdd(counters[selectedRewriteReady
                                        ? PATH_GUIDE_PREVIOUS_LANE_SELECTED_REWRITE_READY
                                        : PATH_GUIDE_PREVIOUS_LANE_SELECTED_REWRITE_REJECT],
                                        stableBernoulliSelected ? 1u : 0u);

                                bool selectedPreserveReady = selectedPayloadValid
                                        && pathMappingSourceReplayMatches(stored, replay)
                                        && stored.replayControl.x == PATH_REPLAY_VERSION
                                        && stored.replayControl.y == replay.replaySegmentCount
                                        && stored.replayControl.z == replay.replayState
                                        && stored.replayControl.w == replay.proposalReplayState
                                        && stored.identity.z == replay.topologyKey
                                        && pathReplayFloatMatches(stored.proposalShift.z,
                                                replay.footprint);
                                InterlockedAdd(counters[selectedPreserveReady
                                        ? PATH_GUIDE_PREVIOUS_LANE_SELECTED_PRESERVE_READY
                                        : PATH_GUIDE_PREVIOUS_LANE_SELECTED_PRESERVE_REJECT],
                                        stableBernoulliSelected ? 1u : 0u);

                                bool retainedPreserveReady = retainedPayloadValid
                                        && pathMappingKind(current) == PATH_MAPPING_IDENTITY
                                        && current.replayControl.x == PATH_REPLAY_VERSION
                                        && current.replayControl.y >= 1u
                                        && current.replayControl.y <= MAX_PATH_SEGMENTS
                                        && current.identity.y < pixelCount
                                        && (current.identity.w & PATH_GENERATION_MASK)
                                                == currentGeneration;
                                InterlockedAdd(counters[retainedPreserveReady
                                        ? PATH_GUIDE_PREVIOUS_LANE_RETAINED_PRESERVE_READY
                                        : PATH_GUIDE_PREVIOUS_LANE_RETAINED_PRESERVE_REJECT],
                                        stableBernoulliValid && !stableBernoulliSelected
                                                ? 1u : 0u);

                                float chosenFinalWeight = stableBernoulliSelected
                                        ? sourceFinalWeightResult : currentFinalWeight;
                                bool chosenWeightsReady = stableBernoulliValid
                                        && nextValid
                                        && isfinite(nextWeightSum) && nextWeightSum > 0.0
                                        && isfinite(nextEffectiveCount)
                                        && nextEffectiveCount > 0.0
                                        && isfinite(chosenFinalWeight)
                                        && chosenFinalWeight > 0.0
                                        && isfinite(min(nextEffectiveCount * 0.25, 1.0))
                                        && min(nextEffectiveCount * 0.25, 1.0) > 0.0;
                                InterlockedAdd(counters[chosenWeightsReady
                                        ? PATH_GUIDE_PREVIOUS_LANE_WEIGHTS_READY
                                        : PATH_GUIDE_PREVIOUS_LANE_WEIGHTS_REJECT],
                                        stableBernoulliValid ? 1u : 0u);

                                // Assemble the complete hypothetical next record in registers and
                                // pass both branches through the common metadata validator. This is
                                // deliberately the last gate before any device write: selected paths
                                // rewrite only receiver-dependent lanes while retaining the original
                                // proposal/replay root; retained paths keep the current opaque payload
                                // bit-for-bit. Neither branch is stored or exposed to the estimator.
                                InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_RECORD_ELIGIBLE],
                                        stableBernoulliValid ? 1u : 0u);
                                PathReservoir branchRecord = current;
                                if (stableBernoulliSelected) {
                                    branchRecord.sampleRadianceTarget = float4(
                                            visibleRadiance, shiftedTarget);
                                    branchRecord.proposalShift = float4(
                                            stored.proposalShift.x,
                                            pssJacobian,
                                            stored.proposalShift.z,
                                            stored.proposalShift.w);
                                    branchRecord.proposalComponents = stored.proposalComponents;
                                    branchRecord.canonicalRadianceTarget = float4(
                                            visibleRadiance, shiftedTarget);
                                    branchRecord.identity = uint4(
                                            stored.identity.x,
                                            currentSourceIndex,
                                            replay.topologyKey,
                                            pathPackMetadata(currentGeneration,
                                                    replay.depth, replay.transportMask));
                                    branchRecord.replay = stored.replay;
                                    branchRecord.replayControl = stored.replayControl;
                                    branchRecord.reconnectionVertex = float4(
                                            replay.reconnectionVertex,
                                            asfloat(pathPackReconnectionMetadata(
                                                    replay.reconnectionDepth,
                                                    replay.reconnectionEventKind,
                                                    replay.reconnectionValid != 0u)));
                                    branchRecord.reconnectionNormalPdf = float4(
                                            replay.reconnectionNormal, receiverPdf);
                                    branchRecord.reconnectionThroughput = float4(
                                            receiverThroughput,
                                            asfloat(PATH_MAPPING_DIFFUSE_RECONNECTION));
                                }
                                branchRecord.weights = float4(
                                        nextWeightSum,
                                        nextEffectiveCount,
                                        chosenFinalWeight,
                                        min(nextEffectiveCount * 0.25, 1.0));

                                bool branchCommonMetadataReady = chosenWeightsReady
                                        && pathReservoirSampleMetadataReady(branchRecord)
                                        && all(isfinite(branchRecord.weights))
                                        && all(branchRecord.weights > float4(0.0));
                                bool selectedRecordReady = stableBernoulliSelected
                                        && selectedRewriteReady
                                        && selectedPreserveReady
                                        && branchCommonMetadataReady
                                        && pathMappingSourceReplayMatches(branchRecord, replay)
                                        && pathMappingKind(branchRecord)
                                                == PATH_MAPPING_DIFFUSE_RECONNECTION
                                        && branchRecord.identity.y == currentSourceIndex
                                        && all(asuint(branchRecord.sampleRadianceTarget)
                                                == asuint(branchRecord.canonicalRadianceTarget));
                                bool retainedRecordReady = stableBernoulliValid
                                        && !stableBernoulliSelected
                                        && retainedPreserveReady
                                        && branchCommonMetadataReady
                                        && pathReservoirSampleMetadataMatches(
                                                branchRecord, current)
                                        && pathMappingKind(branchRecord)
                                                == PATH_MAPPING_IDENTITY;
                                InterlockedAdd(counters[selectedRecordReady
                                        ? PATH_GUIDE_PREVIOUS_RECORD_SELECTED_READY
                                        : PATH_GUIDE_PREVIOUS_RECORD_SELECTED_REJECT],
                                        stableBernoulliSelected ? 1u : 0u);
                                InterlockedAdd(counters[retainedRecordReady
                                        ? PATH_GUIDE_PREVIOUS_RECORD_RETAINED_READY
                                        : PATH_GUIDE_PREVIOUS_RECORD_RETAINED_REJECT],
                                        stableBernoulliValid && !stableBernoulliSelected
                                                ? 1u : 0u);

                                // Pair the register-only reservoir with the exact source-root
                                // provenance that a future ping-pong write would carry. Selected
                                // history advances the old root into the current camera-relative
                                // frame; retained history captures a fresh current queue chain.
                                // This remains a register/counter audit and never writes either
                                // scratch buffer or committed history.
                                InterlockedAdd(counters[PATH_GUIDE_PREVIOUS_PAIR_ELIGIBLE],
                                        stableBernoulliValid ? 1u : 0u);
                                PathSourceRoot branchRoot = {};
                                bool branchRootReady = false;
                                uint branchSourceIndex = branchRecord.identity.y;
                                uint2 branchSourcePixel = uint2(
                                        branchSourceIndex % dimensions.x,
                                        branchSourceIndex / dimensions.x);
                                float4 branchSourcePositionMaterial =
                                        gRestirPositionMaterial[int2(branchSourcePixel)];
                                float4 branchSourceNormalRoughness =
                                        gRestirNormalRoughness[int2(branchSourcePixel)];
                                if (stableBernoulliSelected && selectedRecordReady) {
                                    branchRoot = root;
                                    branchRoot.sourcePositionMaterial =
                                            currentSourcePositionMaterial;
                                    branchRoot.sourcePositionMaterial.xyz -=
                                            worldPush.camOffset;
                                    branchRoot.sourceNormalRoughness =
                                            currentSourceNormalRoughness;
                                    branchRoot.receiverPositionMaterial =
                                            receiverPositionMaterial;
                                    branchRoot.receiverPositionMaterial.xyz -=
                                            worldPush.camOffset;
                                    branchRoot.receiverNormalRoughness =
                                            receiverNormalRoughness;
                                    branchRoot.segment0.ro -= worldPush.camDelta;
                                    branchRoot.segment1.ro -= worldPush.camDelta;
                                    branchRootReady = true;
                                } else if (!stableBernoulliSelected && retainedRecordReady) {
                                    branchRootReady = pathCaptureCurrentGuideRoot(
                                            branchRecord, dimensions,
                                            receiverPositionMaterial,
                                            receiverNormalRoughness,
                                            branchRoot);
                                }
                                uint branchSegments = branchRecord.replayControl.y;
                                bool branchRootFinite = all(isfinite(
                                        branchRoot.sourcePositionMaterial))
                                        && all(isfinite(branchRoot.sourceNormalRoughness))
                                        && all(isfinite(branchRoot.receiverPositionMaterial))
                                        && all(isfinite(branchRoot.receiverNormalRoughness))
                                        && all(isfinite(branchRoot.segment0.ro))
                                        && all(isfinite(branchRoot.segment1.ro));
                                bool branchRootChainValid = branchSegments == 1u
                                        ? branchRoot.segment0.nextRecord == PATH_NO_NEXT
                                        : branchSegments == MAX_PATH_SEGMENTS
                                                && branchRoot.segment0.nextRecord
                                                        != PATH_NO_NEXT
                                                && branchRoot.segment1.nextRecord
                                                        == PATH_NO_NEXT;
                                bool branchRootIdentityValid = !stableBernoulliSelected
                                        ? branchRootReady
                                        : branchRootFinite
                                                && pathReplayPositionMatches(
                                                        branchRoot.sourcePositionMaterial.xyz
                                                                + worldPush.camOffset,
                                                        branchSourcePositionMaterial.xyz)
                                                && pathReplayColorMatches(
                                                        branchRoot.sourceNormalRoughness.xyz,
                                                        branchSourceNormalRoughness.xyz)
                                                && pathReplayFloatMatches(
                                                        branchRoot.sourceNormalRoughness.w,
                                                        branchSourceNormalRoughness.w)
                                                && pathReplayPositionMatches(
                                                        branchRoot.receiverPositionMaterial.xyz
                                                                + worldPush.camOffset,
                                                        receiverPositionMaterial.xyz)
                                                && pathReplayColorMatches(
                                                        branchRoot.receiverNormalRoughness.xyz,
                                                        receiverNormalRoughness.xyz)
                                                && pathReplayFloatMatches(
                                                        branchRoot.receiverNormalRoughness.w,
                                                        receiverNormalRoughness.w)
                                                && uint(max(branchRoot.sourcePositionMaterial.w,
                                                        0.0))
                                                        == uint(max(branchSourcePositionMaterial.w,
                                                                0.0))
                                                && uint(max(branchRoot.receiverPositionMaterial.w,
                                                        0.0))
                                                        == uint(max(receiverPositionMaterial.w,
                                                                0.0));
                                bool branchPairReady = stableBernoulliValid
                                        && branchRootReady
                                        && branchRootChainValid
                                        && branchRootIdentityValid;
                                InterlockedAdd(counters[branchPairReady
                                        ? stableBernoulliSelected
                                                ? PATH_GUIDE_PREVIOUS_PAIR_SELECTED_READY
                                                : PATH_GUIDE_PREVIOUS_PAIR_RETAINED_READY
                                        : stableBernoulliSelected
                                                ? PATH_GUIDE_PREVIOUS_PAIR_SELECTED_REJECT
                                                : PATH_GUIDE_PREVIOUS_PAIR_RETAINED_REJECT],
                                        stableBernoulliValid ? 1u : 0u);
                                if (stableBernoulliValid && !branchPairReady) {
                                    InterlockedAdd(counters[!branchRootReady
                                            ? PATH_GUIDE_PREVIOUS_PAIR_ROOT_CAPTURE_REJECT
                                            : !branchRootChainValid
                                                    ? PATH_GUIDE_PREVIOUS_PAIR_ROOT_CHAIN_REJECT
                                                    : PATH_GUIDE_PREVIOUS_PAIR_ROOT_IDENTITY_REJECT],
                                            1u);
                                }

                                // Replay the just-assembled pair directly from its own root before
                                // authorizing any device write. This is still register/counter only:
                                // it neither advances the stored replay seeds nor changes scratch,
                                // history, selection, or the estimator.
                                InterlockedAdd(counters[
                                        PATH_GUIDE_PREVIOUS_PAIR_REPLAY_ELIGIBLE],
                                        branchPairReady ? 1u : 0u);
                                if (branchPairReady) {
                                    PathReplayAggregate branchReplay =
                                            pathReplayAggregateEmpty();
                                    for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
                                        if (leaf >= branchSegments) break;
                                        PackedPathSegment packed = leaf == 0u
                                                ? branchRoot.segment0
                                                : branchRoot.segment1;
                                        packed.ro += worldPush.camOffset;
                                        PathSegment branchSegment =
                                                unpackPathSegment(packed);
                                        uint replaySeed = leaf == 0u
                                                ? branchRecord.replay.x
                                                : branchRecord.replay.z;
                                        uint proposalReplaySeed = leaf == 0u
                                                ? branchRecord.replay.y
                                                : branchRecord.replay.w;
                                        PathTraceResult branchTraced = tracePathSeeded(
                                                branchSegment, branchSourcePixel, true,
                                                replaySeed, proposalReplaySeed);
                                        pathReplayAccumulate(branchReplay, branchTraced, leaf);
                                    }
                                    bool branchReplayMatches = stableBernoulliSelected
                                            ? pathMappingSourceReplayMatches(
                                                    branchRecord, branchReplay)
                                            : pathReplayCompare(branchRecord, branchReplay) == 0u;
                                    InterlockedAdd(counters[stableBernoulliSelected
                                            ? branchReplayMatches
                                                    ? PATH_GUIDE_PREVIOUS_PAIR_REPLAY_SELECTED_ACCEPTED
                                                    : PATH_GUIDE_PREVIOUS_PAIR_REPLAY_SELECTED_REJECT
                                            : branchReplayMatches
                                                    ? PATH_GUIDE_PREVIOUS_PAIR_REPLAY_RETAINED_ACCEPTED
                                                    : PATH_GUIDE_PREVIOUS_PAIR_REPLAY_RETAINED_REJECT],
                                            1u);
                                    if (branchReplayMatches
                                            && worldPush.pathShiftBranchReservoirCurrentAddr != 0u
                                            && worldPush.pathShiftBranchRootCurrentAddr != 0u
                                            && worldPush.pathShiftDiagnosticPairAddr != 0u) {
                                        InterlockedAdd(counters[
                                                PATH_GUIDE_BRANCH_SCRATCH_WRITE_ELIGIBLE], 1u);
                                        uint captureIndex = 0u;
                                        InterlockedAdd(counters[
                                                PATH_GUIDE_BRANCH_SCRATCH_CAPTURE_CURSOR],
                                                1u, captureIndex);
                                        if (captureIndex
                                                < PATH_BRANCH_SCRATCH_CAPTURE_CAPACITY) {
                                            PathBranchScratchCapture capture;
                                            capture.metadata = uint4(pixelIndex,
                                                    pathMappingKind(branchRecord),
                                                    worldPush.frameIndex,
                                                    currentGeneration);
                                            capture.reservoir = branchRecord;
                                            capture.root = branchRoot;
                                            DevicePtr<PathBranchScratchCapture> captures =
                                                    DevicePtr<PathBranchScratchCapture>(worldPush
                                                            .pathShiftDiagnosticPairAddr
                                                            + uint64_t(
                                                                    PATH_BRANCH_SCRATCH_CAPTURE_BYTE_OFFSET));
                                            captures[captureIndex] = capture;
                                        }
                                        DevicePtr<PathReservoir>(worldPush
                                                .pathShiftBranchReservoirCurrentAddr)[pixelIndex]
                                                = branchRecord;
                                        DevicePtr<PathSourceRoot>(worldPush
                                                .pathShiftBranchRootCurrentAddr)[pixelIndex]
                                                = branchRoot;
                                        InterlockedAdd(counters[
                                                PATH_GUIDE_BRANCH_SCRATCH_WRITE_COMPLETED], 1u);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    InterlockedAdd(counters[mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION
            ? PATH_GUIDE_PREVIOUS_REPLAY_SELECTED_ACCEPTED
            : PATH_GUIDE_PREVIOUS_REPLAY_RETAINED_ACCEPTED], 1u);
}

// Storage/lifetime audit for the future guide-driven sample. The cross-frame replay dispatch writes
// one complete PathReservoir or one ABI-tagged terminal sentinel per eligible receiver. A distinct
// same-frame dispatch reads that device memory after a barrier; it never consumes or updates history.
public void pathGuideScratchValidate() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW
            || (pc.historyFlags & PATH_GUIDE_SCRATCH_VALIDATE_PASS_FLAG) == 0u
            || worldPush.pathShiftDiagnosticCounterAddr == 0u
            || worldPush.pathShiftGuideScratchAddr == 0u
            || worldPush.pathShiftGuideRootScratchAddr == 0u) {
        return;
    }

    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    PathReservoir stored =
            ConstPtr<PathReservoir>(worldPush.pathShiftGuideScratchAddr)[pixelIndex];
    DevicePtr<uint> counters =
            DevicePtr<uint>(worldPush.pathShiftDiagnosticCounterAddr);

    // vkCmdFillBuffer produces an exact all-zero untouched record. Intentional terminal sentinels
    // always carry the replay ABI, zero segment count, magic, category, and current generation.
    if (all(stored.replayControl == uint4(0u)) && stored.identity.w == 0u) {
        return;
    }
    InterlockedAdd(counters[PATH_GUIDE_SCRATCH_STORAGE_ELIGIBLE], 1u);

    uint currentGeneration = pc.historyGeneration & PATH_GENERATION_MASK;
    if (stored.replayControl.y == 0u) {
        uint terminal = stored.replayControl.w;
        bool sentinelValid = stored.replayControl.x == PATH_REPLAY_VERSION
                && stored.replayControl.z == PATH_GUIDE_SCRATCH_SENTINEL_MAGIC
                && stored.identity.w == currentGeneration
                && (terminal == PATH_GUIDE_SAMPLE_COPY_EMPTY
                        || terminal == PATH_GUIDE_SAMPLE_COPY_METADATA_REJECT
                        || terminal == PATH_GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT);
        if (!sentinelValid) {
            InterlockedAdd(counters[PATH_GUIDE_SCRATCH_STORAGE_INVALID], 1u);
            return;
        }
        uint storageTerminal = terminal == PATH_GUIDE_SAMPLE_COPY_EMPTY
                ? PATH_GUIDE_SCRATCH_STORAGE_EMPTY
                : terminal == PATH_GUIDE_SAMPLE_COPY_METADATA_REJECT
                        ? PATH_GUIDE_SCRATCH_STORAGE_METADATA_REJECT
                        : PATH_GUIDE_SCRATCH_STORAGE_ARITHMETIC_REJECT;
        InterlockedAdd(counters[storageTerminal], 1u);
        return;
    }

    bool weightsValid = all(isfinite(stored.weights))
            && all(stored.weights > float4(0.0));
    bool metadataValid = pathReservoirSampleMetadataReady(stored)
            && (stored.identity.w & PATH_GENERATION_MASK) == currentGeneration;
    uint mappingKind = pathMappingKind(stored);
    if (!weightsValid || !metadataValid
            || (mappingKind != PATH_MAPPING_IDENTITY
                    && mappingKind != PATH_MAPPING_DIFFUSE_RECONNECTION)) {
        InterlockedAdd(counters[PATH_GUIDE_SCRATCH_STORAGE_INVALID], 1u);
        return;
    }
    bool sourceSelected = mappingKind == PATH_MAPPING_DIFFUSE_RECONNECTION;
    InterlockedAdd(counters[sourceSelected
            ? PATH_GUIDE_SCRATCH_STORAGE_SELECTED
            : PATH_GUIDE_SCRATCH_STORAGE_RETAINED], 1u);

    InterlockedAdd(counters[PATH_GUIDE_ROOT_STORAGE_ELIGIBLE], 1u);
    PathSourceRoot root = ConstPtr<PathSourceRoot>(
            worldPush.pathShiftGuideRootScratchAddr)[pixelIndex];
    bool rootMissing = all(root.sourcePositionMaterial == float4(0.0))
            && all(root.sourceNormalRoughness == float4(0.0))
            && all(root.receiverPositionMaterial == float4(0.0))
            && all(root.receiverNormalRoughness == float4(0.0))
            && all(root.segment0.ro == float3(0.0))
            && all(root.segment1.ro == float3(0.0));
    if (rootMissing) {
        InterlockedAdd(counters[PATH_GUIDE_ROOT_STORAGE_MISSING], 1u);
        return;
    }

    uint sourceIndex = stored.identity.y;
    uint pixelCount = dimensions.x * dimensions.y;
    bool rootFinite = all(isfinite(root.sourcePositionMaterial))
            && all(isfinite(root.sourceNormalRoughness))
            && all(isfinite(root.receiverPositionMaterial))
            && all(isfinite(root.receiverNormalRoughness))
            && all(isfinite(root.segment0.ro))
            && all(isfinite(root.segment1.ro));
    bool rootChainValid = stored.replayControl.y == 1u
            ? root.segment0.nextRecord == PATH_NO_NEXT
            : root.segment0.nextRecord != PATH_NO_NEXT
                    && root.segment1.nextRecord == PATH_NO_NEXT;
    bool rootMetadataValid = rootFinite && rootChainValid && sourceIndex < pixelCount;
    if (rootMetadataValid) {
        uint2 sourcePixel = uint2(sourceIndex % dimensions.x,
                sourceIndex / dimensions.x);
        float4 sourcePositionMaterial = root.sourcePositionMaterial;
        sourcePositionMaterial.xyz += worldPush.camOffset;
        float4 receiverPositionMaterial = root.receiverPositionMaterial;
        receiverPositionMaterial.xyz += worldPush.camOffset;
        float4 currentSourcePositionMaterial =
                gRestirPositionMaterial[int2(sourcePixel)];
        float4 currentSourceNormalRoughness =
                gRestirNormalRoughness[int2(sourcePixel)];
        float4 currentReceiverPositionMaterial =
                gRestirPositionMaterial[int2(pixel)];
        float4 currentReceiverNormalRoughness =
                gRestirNormalRoughness[int2(pixel)];
        rootMetadataValid = pathReplayPositionMatches(sourcePositionMaterial.xyz,
                    currentSourcePositionMaterial.xyz)
                && pathReplayColorMatches(root.sourceNormalRoughness.xyz,
                    currentSourceNormalRoughness.xyz)
                && pathReplayFloatMatches(root.sourceNormalRoughness.w,
                    currentSourceNormalRoughness.w)
                && uint(max(sourcePositionMaterial.w, 0.0))
                        == uint(max(currentSourcePositionMaterial.w, 0.0))
                && pathReplayPositionMatches(receiverPositionMaterial.xyz,
                    currentReceiverPositionMaterial.xyz)
                && pathReplayColorMatches(root.receiverNormalRoughness.xyz,
                    currentReceiverNormalRoughness.xyz)
                && pathReplayFloatMatches(root.receiverNormalRoughness.w,
                    currentReceiverNormalRoughness.w)
                && uint(max(receiverPositionMaterial.w, 0.0))
                        == uint(max(currentReceiverPositionMaterial.w, 0.0));
    }
    InterlockedAdd(counters[rootMetadataValid
            ? sourceSelected
                    ? PATH_GUIDE_ROOT_STORAGE_SELECTED_READY
                    : PATH_GUIDE_ROOT_STORAGE_RETAINED_READY
            : PATH_GUIDE_ROOT_STORAGE_METADATA_REJECT], 1u);
}

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    if ((pc.historyFlags & PATH_GUIDE_BRANCH_SCRATCH_VALIDATE_PASS_FLAG) != 0u) {
        pathBranchScratchStorageValidate();
        return;
    }
    if ((pc.historyFlags & PATH_GUIDE_PREVIOUS_REPLAY_PASS_FLAG) != 0u) {
        pathGuidePreviousReplayValidate();
        return;
    }
    if ((pc.historyFlags & PATH_GUIDE_SCRATCH_VALIDATE_PASS_FLAG) != 0u) {
        pathGuideScratchValidate();
        return;
    }
    if ((pc.historyFlags & PATH_MAPPING_REPLAY_PASS_FLAG) != 0u) {
        pathMappingReplayValidate();
        return;
    }
    bool pathDebug = pc.debugView == PATH_DEBUG_VIEW;
    bool pathTemporalDebug = pc.debugView == PATH_TEMPORAL_DEBUG_VIEW;
    bool pathReplayDebug = pc.debugView == PATH_REPLAY_DEBUG_VIEW;
    bool pathTemporalMergeDebug = pc.debugView == PATH_TEMPORAL_MERGE_DEBUG_VIEW;
    bool pathSpatialDebug = pc.debugView == PATH_SPATIAL_DEBUG_VIEW;
    bool pathSpatialPolicyDebug = pc.debugView == PATH_SPATIAL_POLICY_DEBUG_VIEW;
    bool pathReconnectionDebug = pc.debugView == PATH_RECONNECTION_DEBUG_VIEW;
    bool pathShiftedRadianceDebug = pc.debugView == PATH_SHIFTED_RADIANCE_DEBUG_VIEW;
    bool canonicalMode = (pc.historyFlags & PATH_CANONICAL_MODE_FLAG) != 0u;
    bool pathCapture = (pc.historyFlags & 4u) != 0u && pc.pathReservoirAddr != 0;
    if (pc.debugView != 0u && !pathDebug && !pathTemporalDebug && !pathReplayDebug
            && !pathTemporalMergeDebug && !pathSpatialDebug && !pathSpatialPolicyDebug
            && !pathReconnectionDebug && !pathShiftedRadianceDebug) {
        return; // pass A owns guide visualization output
    }
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    if ((pathDebug || pathTemporalDebug || pathReplayDebug || pathTemporalMergeDebug
            || pathSpatialDebug || pathSpatialPolicyDebug || pathReconnectionDebug
            || pathShiftedRadianceDebug)
            && !pathCapture) {
        outImage[pix] = float4(1.0, 0.0, 1.0, 1.0); // setting disabled / storage unavailable
        return;
    }
    ConstPtr<PackedPathSegment> queue = ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
    float3 frameRadiance = float3(0.0, 0.0, 0.0);
    if (!pathCapture) {
        uint recordIndex = pixelIndex;
        for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
            PackedPathSegment packed = queue[recordIndex];
            PathSegment segment = unpackPathSegment(packed);
            for (uint s = 0u; s < spp; ++s) {
                frameRadiance += tracePath(segment, uint2(pix), s + leaf * spp, false).radiance;
            }
            if (packed.nextRecord == PATH_NO_NEXT) {
                break;
            }
            recordIndex = packed.nextRecord;
        }
        outImage[pix] = float4(frameRadiance / float(spp), 1.0);
        return;
    }

    PathReservoir pathReservoir = pathReservoirEmpty();
    uint selectionSeed = pathHash(pixelIndex ^ worldPush.frameIndex * 747796405u ^ 2891336453u);
    float3 canonicalDebugColor = float3(0.0);
    for (uint s = 0u; s < spp; ++s) {
        float3 sampleRadiance = float3(0.0);
        uint sampleDepth = 0u;
        uint sampleTopology = pathHash(0x7F4A7C15u);
        uint4 sampleReplaySeeds = uint4(0u);
        uint sampleReplayState = 0u;
        uint sampleProposalReplayState = 0u;
        uint sampleReplaySegmentCount = 0u;
        uint sampleTransportMask = 0u;
        float sampleFootprint = RAY_CONE_MIN_WIDTH;
        float sampleLightProposalPdf = 1.0;
        uint sampleLightProposalEvents = 0u;
        float sampleContinuationProposalPdf = 1.0;
        uint sampleContinuationProposalEvents = 0u;
        float sampleRouletteProposalPdf = 1.0;
        uint sampleRouletteProposalEvents = 0u;
        float3 sampleCanonicalRadiance = float3(0.0);
        uint sampleCanonicalValid = 0u;
        float sampleCanonicalContinuationProposalPdf = 1.0;
        uint sampleCanonicalContinuationProposalEvents = 0u;
        float sampleCanonicalRouletteProposalPdf = 1.0;
        uint sampleCanonicalRouletteProposalEvents = 0u;
        float3 sampleReconnectionVertex = float3(0.0);
        uint sampleReconnectionValid = 0u;
        uint sampleReconnectionDepth = 0u;
        float3 sampleReconnectionNormal = float3(0.0);
        float sampleReconnectionSourcePdf = 0.0;
        uint sampleReconnectionEventKind = PATH_RECONNECTION_EVENT_NONE;
        float3 sampleReconnectionThroughput = float3(0.0);
        uint recordIndex = pixelIndex;
        for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
            PackedPathSegment packed = queue[recordIndex];
            PathSegment segment = unpackPathSegment(packed);
            PathTraceResult traced = tracePath(segment, uint2(pix), s + leaf * spp, true);
            if (leaf == 0u) {
                sampleReplaySeeds.xy = uint2(traced.replaySeed, traced.proposalReplaySeed);
            } else {
                sampleReplaySeeds.zw = uint2(traced.replaySeed, traced.proposalReplaySeed);
            }
            sampleReplaySegmentCount = leaf + 1u;
            sampleRadiance += traced.radiance;
            sampleDepth = max(sampleDepth, traced.depth);
            sampleTopology = pathTopologyStep(sampleTopology,
                    traced.topologyKey ^ leaf * 668265263u);
            sampleReplayState = pathHash(sampleReplayState ^ traced.replayState ^ leaf * 374761393u);
            sampleProposalReplayState = pathHash(sampleProposalReplayState
                    ^ traced.proposalReplayState ^ leaf * 1103515245u);
            sampleTransportMask |= traced.transportMask;
            sampleFootprint = max(sampleFootprint, traced.footprint);
            sampleLightProposalPdf = pathCombineProposal(
                    sampleLightProposalPdf, traced.lightProposalPdf);
            sampleLightProposalEvents += traced.lightProposalEvents;
            sampleContinuationProposalPdf = pathCombineProposal(
                    sampleContinuationProposalPdf, traced.continuationProposalPdf);
            sampleContinuationProposalEvents += traced.continuationProposalEvents;
            sampleRouletteProposalPdf = pathCombineProposal(
                    sampleRouletteProposalPdf, traced.rouletteProposalPdf);
            sampleRouletteProposalEvents += traced.rouletteProposalEvents;
            if (sampleCanonicalValid == 0u && traced.canonicalValid != 0u) {
                sampleCanonicalRadiance = traced.canonicalRadiance;
                sampleCanonicalValid = 1u;
                sampleCanonicalContinuationProposalPdf = traced.canonicalContinuationProposalPdf;
                sampleCanonicalContinuationProposalEvents = traced.canonicalContinuationProposalEvents;
                sampleCanonicalRouletteProposalPdf = traced.canonicalRouletteProposalPdf;
                sampleCanonicalRouletteProposalEvents = traced.canonicalRouletteProposalEvents;
                sampleReconnectionVertex = traced.reconnectionVertex;
                sampleReconnectionValid = traced.reconnectionValid;
                sampleReconnectionDepth = traced.reconnectionDepth;
                sampleReconnectionNormal = traced.reconnectionNormal;
                sampleReconnectionSourcePdf = traced.reconnectionSourcePdf;
                sampleReconnectionEventKind = traced.reconnectionEventKind;
                sampleReconnectionThroughput = traced.reconnectionThroughput;
            }
            if (packed.nextRecord == PATH_NO_NEXT) {
                break;
            }
            recordIndex = packed.nextRecord;
        }
        frameRadiance += sampleRadiance;
        PathCandidate candidate;
        candidate.radiance = sampleRadiance;
        candidate.depth = sampleDepth;
        candidate.topologyKey = sampleTopology;
        candidate.replaySeeds = sampleReplaySeeds;
        candidate.replayState = sampleReplayState;
        candidate.proposalReplayState = sampleProposalReplayState;
        candidate.replaySegmentCount = sampleReplaySegmentCount;
        candidate.transportMask = sampleTransportMask;
        candidate.footprint = sampleFootprint;
        candidate.lightProposalPdf = sampleLightProposalPdf;
        candidate.lightProposalEvents = sampleLightProposalEvents;
        candidate.continuationProposalPdf = sampleContinuationProposalPdf;
        candidate.rouletteProposalPdf = sampleRouletteProposalPdf;
        candidate.continuationProposalEvents = sampleContinuationProposalEvents;
        candidate.rouletteProposalEvents = sampleRouletteProposalEvents;
        candidate.canonicalRadiance = sampleCanonicalRadiance;
        candidate.canonicalValid = sampleCanonicalValid;
        candidate.canonicalContinuationProposalPdf = sampleCanonicalContinuationProposalPdf;
        candidate.canonicalRouletteProposalPdf = sampleCanonicalRouletteProposalPdf;
        candidate.canonicalContinuationProposalEvents = sampleCanonicalContinuationProposalEvents;
        candidate.canonicalRouletteProposalEvents = sampleCanonicalRouletteProposalEvents;
        candidate.reconnectionVertex = sampleReconnectionVertex;
        candidate.reconnectionValid = sampleReconnectionValid;
        candidate.reconnectionDepth = sampleReconnectionDepth;
        candidate.reconnectionNormal = sampleReconnectionNormal;
        candidate.reconnectionSourcePdf = sampleReconnectionSourcePdf;
        candidate.reconnectionEventKind = sampleReconnectionEventKind;
        candidate.reconnectionThroughput = sampleReconnectionThroughput;
        uint sampleId = pathHash(worldPush.frameIndex * 2654435761u
                ^ pixelIndex * 2246822519u ^ s * 3266489917u);
        uint canonicalStatus = pathReservoirUpdate(pathReservoir, candidate, sampleId, pixelIndex,
                pc.historyGeneration, selectionSeed, canonicalMode);
        if (canonicalMode) {
            // Averaging categorical colors across SPP exposes the local admission ratio without
            // allocating another persistent buffer or changing the PathReservoir ABI.
            canonicalDebugColor += pathCanonicalCandidateDebugColor(canonicalStatus);
        }
    }
    uint temporalMergeStatus = 0u;
    if (pathTemporalMergeDebug) {
        if (pathReservoir.weights.y <= 0.0 || pathReservoir.sampleRadianceTarget.w <= 0.0) {
            temporalMergeStatus |= PATH_TEMPORAL_CURRENT_EMPTY;
        }
        bool historyAvailable = (pc.historyFlags & 8u) != 0u
                && pc.pathReservoirPreviousAddr != 0u;
        if (!historyAvailable) {
            temporalMergeStatus |= PATH_TEMPORAL_HISTORY_EMPTY;
        } else {
            // T-049 intentionally uses identity temporal mapping. Motion-vector reprojection stays
            // in the admission pass until a replay/reconnection shift is defined for it.
            PathReservoir previous =
                    ConstPtr<PathReservoir>(pc.pathReservoirPreviousAddr)[pixelIndex];
            if (previous.weights.y <= 0.0 || previous.sampleRadianceTarget.w <= 0.0) {
                temporalMergeStatus |= PATH_TEMPORAL_HISTORY_EMPTY;
            } else if (previous.replayControl.x != PATH_REPLAY_VERSION
                    || previous.replayControl.y < 1u
                    || previous.replayControl.y > MAX_PATH_SEGMENTS
                    || (previous.replayControl.y < MAX_PATH_SEGMENTS
                            && any(previous.replay.zw != uint2(0u)))) {
                temporalMergeStatus |= PATH_TEMPORAL_REPLAY_REJECT;
            } else {
                PathReplayAggregate replay = pathReplayAggregateEmpty();
                uint expectedSegments = previous.replayControl.y;
                uint recordIndex = previous.identity.y;
                bool chainComplete = true;
                for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
                    if (leaf >= expectedSegments || recordIndex == PATH_NO_NEXT) {
                        break;
                    }
                    PackedPathSegment packed = queue[recordIndex];
                    PathSegment segment = unpackPathSegment(packed);
                    uint replaySeed = leaf == 0u
                            ? previous.replay.x : previous.replay.z;
                    uint proposalReplaySeed = leaf == 0u
                            ? previous.replay.y : previous.replay.w;
                    PathTraceResult traced = tracePathSeeded(
                            segment, uint2(pix), true, replaySeed, proposalReplaySeed);
                    pathReplayAccumulate(replay, traced, leaf);
                    if (leaf + 1u == expectedSegments) {
                        chainComplete = packed.nextRecord == PATH_NO_NEXT;
                        break;
                    }
                    if (packed.nextRecord == PATH_NO_NEXT) {
                        chainComplete = false;
                        break;
                    }
                    recordIndex = packed.nextRecord;
                }
                if (!chainComplete || replay.replaySegmentCount != expectedSegments
                        || pathReplayCompare(previous, replay) != 0u) {
                    temporalMergeStatus |= PATH_TEMPORAL_REPLAY_REJECT;
                } else {
                    PathCandidate shifted = pathCandidateFromReplay(replay, previous);
                    temporalMergeStatus |= pathReservoirTemporalMerge(pathReservoir, previous,
                            shifted, pixelIndex, pc.historyGeneration, selectionSeed);
                }
            }
        }
    }
    DevicePtr<PathReservoir>(pc.pathReservoirAddr)[pixelIndex] = pathReservoir;
    if (pathTemporalMergeDebug) {
        outImage[pix] = float4(pathTemporalMergeDebugColor(temporalMergeStatus), 1.0);
    }
    if (pathReplayDebug) {
        uint replayMismatch = PATH_REPLAY_EMPTY;
        if (pathReservoir.weights.y > 0.0) {
            replayMismatch = PATH_REPLAY_ABI_MISMATCH;
        }
        if (pathReservoir.weights.y > 0.0
                && pathReservoir.replayControl.x == PATH_REPLAY_VERSION
                && pathReservoir.replayControl.y >= 1u
                && pathReservoir.replayControl.y <= MAX_PATH_SEGMENTS
                && (pathReservoir.replayControl.y == MAX_PATH_SEGMENTS
                        || all(pathReservoir.replay.zw == uint2(0u)))) {
            PathReplayAggregate replay = pathReplayAggregateEmpty();
            uint expectedSegments = pathReservoir.replayControl.y;
            uint recordIndex = pathReservoir.identity.y;
            bool chainComplete = true;
            for (uint leaf = 0u; leaf < MAX_PATH_SEGMENTS; ++leaf) {
                if (leaf >= expectedSegments || recordIndex == PATH_NO_NEXT) {
                    break;
                }
                PackedPathSegment packed = queue[recordIndex];
                PathSegment segment = unpackPathSegment(packed);
                uint replaySeed = leaf == 0u
                        ? pathReservoir.replay.x : pathReservoir.replay.z;
                uint proposalReplaySeed = leaf == 0u
                        ? pathReservoir.replay.y : pathReservoir.replay.w;
                PathTraceResult traced = tracePathSeeded(
                        segment, uint2(pix), true, replaySeed, proposalReplaySeed);
                pathReplayAccumulate(replay, traced, leaf);
                if (leaf + 1u == expectedSegments) {
                    chainComplete = packed.nextRecord == PATH_NO_NEXT;
                    break;
                }
                if (packed.nextRecord == PATH_NO_NEXT) {
                    chainComplete = false;
                    break;
                }
                recordIndex = packed.nextRecord;
            }
            if (!chainComplete || replay.replaySegmentCount != expectedSegments) {
                replayMismatch = PATH_REPLAY_SEGMENT_MISMATCH;
            } else {
                replayMismatch = pathReplayCompare(pathReservoir, replay);
            }
        }
        outImage[pix] = float4(pathReplayDebugColor(replayMismatch), 1.0);
    }
    // View 14 is written by path_temporal_validate after this pass. Leaving outImage untouched here
    // prevents an intermediate radiance image from leaking into the diagnostic if that pass changes.
    if (!pathTemporalDebug && !pathReplayDebug && !pathTemporalMergeDebug
            && !pathSpatialDebug && !pathSpatialPolicyDebug) {
        outImage[pix] = float4(pathDebug
                ? (canonicalMode
                        ? canonicalDebugColor / float(spp)
                        : pathReservoirDebug(pathReservoir, worldPush.maxBounces))
                : frameRadiance / float(spp), 1.0);
    }
}
