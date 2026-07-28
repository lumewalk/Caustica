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
            if (!canonicalValid) {
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
        if (captureMetadata) {
            topologyKey = pathTopologyStep(topologyKey,
                    (material & 0xFFu) | ((uint(hitDepth) & 0xFFu) << 8u));
        }
        bool restirAtFirstInterface = (worldPush.flags & 2u) != 0u && hitDepth == 0
                && pc.directReservoirAddr != 0;
        if (restirAtFirstInterface) {
            uint2 restirDimensions = DispatchRaysDimensions().xy;
            uint restirIndex = pix.y * restirDimensions.x + pix.x;
            DirectReservoir restirReservoir = ConstPtr<DirectReservoir>(pc.directReservoirAddr)[restirIndex];
            uint restirFlags = reservoirFlags(restirReservoir);
            restirAtFirstInterface = (restirFlags & RESERVOIR_FLAG_VALID) != 0u
                    && restirReservoir.metadata.y == worldPush.lightEpoch
                    && restirReservoir.weights.y > 0.0
                    && restirReservoir.weights.z > 0.0;
        }
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

            if (bounce >= maxBounces) {
                break;
            }
            throughput *= albedo;
            ro = hitPos + n * SURF_BIAS;
            rd = cosineDir(n, seed);
            pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                    max(abs(dot(n, rd)) * INV_PI, 1.0e-20));
            if (!canonicalValid) {
                pathAccumulateProposal(canonicalContinuationProposalPdf,
                        canonicalContinuationProposalEvents,
                        max(abs(dot(n, rd)) * INV_PI, 1.0e-20));
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
        if (!canonicalValid && emission > 0.0) {
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
                throughput *= fresnelSchlick(clamp(dot(n, v), 0.0, 1.0), F0) / ps;
                transportMask |= PATH_TRANSPORT_DELTA;
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xB101u);
            } else {
                float3 h = sampleGGXVNDF(n, v, rough, seed);
                l = reflect(rd, h); // rd = -v: reflect the incoming ray about the microfacet normal
                float ndl2 = dot(n, l);
                if (ndl2 <= 0.0) {
                    break; // microfacet reflects below the surface: terminate this path
                }
                float directionalPdf = ggxVndfReflectionPdf(n, v, h, rough);
                pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                        max(ps * directionalPdf, 1.0e-20));
                if (!canonicalValid) {
                    pathAccumulateProposal(canonicalContinuationProposalPdf,
                            canonicalContinuationProposalEvents,
                            max(ps * directionalPdf, 1.0e-20));
                }
                // VNDF + separable Smith ⇒ weight = F · G2/G1(v) = F · G1(l); divide by the lobe pdf ps.
                float3 F = fresnelSchlick(max(0.0, dot(v, h)), F0);
                throughput *= F * ggxG1(ndl2, rough) / ps;
                transportMask |= PATH_TRANSPORT_GLOSSY;
                if (captureMetadata) topologyKey = pathTopologyStep(topologyKey, 0xB102u);
                rayConeSpread = max(rayConeSpread, rough * RAY_CONE_GLOSSY_SPREAD_SCALE);
            }
            ro = p;
            rd = l;
            showCelestial = true; // specular lobe: this ray is a (glossy) mirror — show the disc
        } else {
            throughput *= diffAlb / (1.0 - ps);
            ro = p;
            rd = cosineDir(n, seed);
            pathAccumulateProposal(continuationProposalPdf, continuationProposalEvents,
                    max((1.0 - ps) * abs(dot(n, rd)) * INV_PI, 1.0e-20));
            if (!canonicalValid) {
                pathAccumulateProposal(canonicalContinuationProposalPdf,
                        canonicalContinuationProposalEvents,
                        max((1.0 - ps) * abs(dot(n, rd)) * INV_PI, 1.0e-20));
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

[shader("raygeneration")]
void main() {
    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    bool pathDebug = pc.debugView == PATH_DEBUG_VIEW;
    bool pathTemporalDebug = pc.debugView == PATH_TEMPORAL_DEBUG_VIEW;
    bool pathCapture = (pc.historyFlags & 4u) != 0u && pc.pathReservoirAddr != 0;
    if (pc.debugView != 0u && !pathDebug && !pathTemporalDebug) {
        return; // pass A owns guide visualization output
    }
    uint2 dispatchIndex = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 pix = int2(dispatchIndex);
    uint spp = max(worldPush.spp, 1u);
    uint pixelIndex = dispatchIndex.y * dimensions.x + dispatchIndex.x;
    if ((pathDebug || pathTemporalDebug) && !pathCapture) {
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
        uint sampleId = pathHash(worldPush.frameIndex * 2654435761u
                ^ pixelIndex * 2246822519u ^ s * 3266489917u);
        pathReservoirUpdate(pathReservoir, candidate, sampleId, pixelIndex,
                pc.historyGeneration, selectionSeed);
    }
    DevicePtr<PathReservoir>(pc.pathReservoirAddr)[pixelIndex] = pathReservoir;
    // View 14 is written by path_temporal_validate after this pass. Leaving outImage untouched here
    // prevents an intermediate radiance image from leaking into the diagnostic if that pass changes.
    if (!pathTemporalDebug) {
        outImage[pix] = float4(pathDebug
                ? pathReservoirDebug(pathReservoir, worldPush.maxBounces)
                : frameRadiance / float(spp), 1.0);
    }
}
