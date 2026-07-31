// Debug-only first-edge reconnection evaluation for ReSTIR PT spatial reuse.
//
// This ray-generation pass runs after path capture and the compute admission/Jacobian pass. It repeats
// the strict view-19 policy for one neighboring reservoir, restricts the first event to diffuse (the
// receiver guides do not yet expose enough F0/metalness data for an exact glossy throughput ratio),
// traces receiver -> source second-hit visibility through the production shadow SBT, and evaluates a
// shifted canonical radiance. It never updates reservoirs, weights, history, or the active estimator.

import world_common;
import world_core;
import trace;
import sampling_core;
import segment;
import path_reservoir_common;

static const float PATH_SHIFT_INV_PI = 0.31830988618;
static const float PATH_SHIFT_MAX_HALF = 65000.0;

static float pathShiftDiffuseDensity(float3 normal, float3 edge) {
    float edgeLength = length(edge);
    if (edgeLength <= 1.0e-5) return 0.0;
    return abs(dot(normal, edge / edgeLength)) * PATH_SHIFT_INV_PI;
}

static float pathShiftDiffuseThroughputRatio(float sourceThroughput,
                                              float receiverThroughput) {
    if (receiverThroughput <= 0.0 || sourceThroughput <= 0.0) return 0.0;
    return receiverThroughput / sourceThroughput;
}

[shader("raygeneration")]
void main() {
    if (pc.debugView != PATH_SHIFTED_RADIANCE_DEBUG_VIEW) return;

    worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    int2 texel = int2(pixel);
    uint pixelIndex = pixel.y * dimensions.x + pixel.x;
    PathReservoir current = ConstPtr<PathReservoir>(pc.pathReservoirAddr)[pixelIndex];

    float4 receiverPositionMaterial = gRestirPositionMaterial[texel];
    float4 receiverNormalRoughness = gRestirNormalRoughness[texel];
    float receiverNormalLength = length(receiverNormalRoughness.xyz);
    uint receiverMaterialIdentity = uint(max(receiverPositionMaterial.w, 0.0));
    uint receiverMaterial = receiverMaterialModel(receiverMaterialIdentity);
    bool receiverPresent = receiverNormalLength >= 0.5
            && receiverMaterial <= MATERIAL_DIELECTRIC
            && current.weights.y > 0.0
            && current.sampleRadianceTarget.w > 0.0;
    bool receiverSupportsDiffuseShift = receiverMaterial == MATERIAL_OPAQUE
            || receiverMaterial == MATERIAL_PARTICLE;

    // States 0..8 mirror view 19. State 9 is a valid but occluded shift, 10 is a
    // source-radiance/albedo support mismatch, 11 is a valid zero target, 12 is a positive shifted
    // target, 13 is invalid arithmetic, and 14 exceeds the R16F debug-output range.
    uint state = receiverPresent ? 1u : 0u;
    float3 shiftedRadiance = float3(0.0);
    float shiftedSourceLuminance = 0.0;
    float shiftedReceiverPdf = 0.0;
    float shiftedPssJacobian = 0.0;
    float shiftedSourceFinalWeight = 0.0;
    float shiftedMergeWeight = 0.0;
    float shiftedCurrentWeightSum = 0.0;
    float shiftedSelectionProbability = 0.0;
    PathReservoir shiftedSource = pathReservoirEmpty();
    float3 shiftedReceiverFirstThroughput = float3(0.0);
    float4 shiftedSourcePositionMaterial = float4(0.0);
    float4 shiftedSourceNormalRoughness = float4(0.0);
    if (receiverPresent) {
        float3 receiverNormal = receiverNormalRoughness.xyz / receiverNormalLength;
        uint currentReconnectionMetadata = asuint(current.reconnectionVertex.w);
        uint currentEvent = pathReconnectionEvent(currentReconnectionMetadata);
        float currentSourcePdf = current.reconnectionNormalPdf.w;
        float3 currentSecondHit = current.reconnectionVertex.xyz;
        bool currentEdgeAvailable =
                (currentReconnectionMetadata & PATH_RECONNECTION_VALID) != 0u
                && ((currentReconnectionMetadata >> PATH_RECONNECTION_DEPTH_SHIFT) & 0xFu) == 1u
                && currentSourcePdf > 0.0
                && all(isfinite(currentSecondHit));
        if (!receiverSupportsDiffuseShift) {
            // Pass A consumes water/dielectric interfaces before the canonical wavefront path begins.
            // Their guide receiver and stored first event therefore describe different vertices and
            // cannot use the opaque/particle diffuse throughput-ratio contract.
            state = 4u;
        } else if (!currentEdgeAvailable) {
            state = 2u;
        } else if (currentEvent != PATH_RECONNECTION_EVENT_DIFFUSE) {
            // Glossy remains unsupported until receiver/source F0 and metalness are available.
            state = 4u;
        } else {
            uint seed = pathHash(pixel.x * 2246822519u
                    ^ pixel.y * 3266489917u ^ pc.frameIndex * 668265263u);
            float rotation = float(seed) * (6.28318530718 / 4294967296.0);
            bool evaluated = false;
            for (uint sample = 0u;
                    sample < PATH_SPATIAL_SAMPLE_COUNT && !evaluated; ++sample) {
                float radiusFraction = sqrt((float(sample) + 0.5)
                        / float(PATH_SPATIAL_SAMPLE_COUNT));
                float angle = rotation + float(sample) * 2.39996322973;
                float radius = 2.0 + radiusFraction * (PATH_SPATIAL_RADIUS - 2.0);
                int2 offset = int2(round(float2(cos(angle), sin(angle)) * radius));
                int2 sourceTexel = texel + offset;
                if (any(sourceTexel < int2(0, 0))
                        || any(sourceTexel >= int2(dimensions))) continue;

                float4 sourcePositionMaterial = gRestirPositionMaterial[sourceTexel];
                float4 sourceNormalRoughness = gRestirNormalRoughness[sourceTexel];
                float sourceNormalLength = length(sourceNormalRoughness.xyz);
                uint sourceMaterialIdentity = uint(max(sourcePositionMaterial.w, 0.0));
                if (sourceNormalLength < 0.5
                        || sourceMaterialIdentity != receiverMaterialIdentity) continue;
                float3 sourceNormal = sourceNormalRoughness.xyz / sourceNormalLength;
                float relativeDepth = abs(length(sourcePositionMaterial.xyz)
                        - length(receiverPositionMaterial.xyz))
                        / max(length(receiverPositionMaterial.xyz), 1.0);
                if (dot(receiverNormal, sourceNormal) < PATH_SPATIAL_NORMAL_COS
                        || relativeDepth > PATH_SPATIAL_RELATIVE_DEPTH) continue;

                uint sourceIndex = uint(sourceTexel.y) * dimensions.x + uint(sourceTexel.x);
                PathReservoir source =
                        ConstPtr<PathReservoir>(pc.pathReservoirAddr)[sourceIndex];
                if (source.weights.y <= 0.0 || source.sampleRadianceTarget.w <= 0.0
                        || source.canonicalRadianceTarget.w <= 0.0) continue;
                if (pathMetadataDepth(source.identity.w) != pathMetadataDepth(current.identity.w)
                        || source.identity.z != current.identity.z
                        || pathMetadataTransport(source.identity.w)
                                != pathMetadataTransport(current.identity.w)) {
                    state = 3u;
                    continue;
                }
                float sourceFootprint = max(source.proposalShift.z, 1.0e-5);
                float receiverFootprint = max(current.proposalShift.z, 1.0e-5);
                if (max(sourceFootprint, receiverFootprint)
                        / min(sourceFootprint, receiverFootprint) > PATH_MAX_FOOTPRINT_RATIO) {
                    state = 3u;
                    continue;
                }

                uint sourceReconnectionMetadata = asuint(source.reconnectionVertex.w);
                uint sourceEvent = pathReconnectionEvent(sourceReconnectionMetadata);
                float sourcePdf = source.reconnectionNormalPdf.w;
                float3 sharedVertex = source.reconnectionVertex.xyz;
                float3 sharedNormalValue = source.reconnectionNormalPdf.xyz;
                float sharedNormalLength = length(sharedNormalValue);
                bool sourceEdgeAvailable =
                        (sourceReconnectionMetadata & PATH_RECONNECTION_VALID) != 0u
                        && ((sourceReconnectionMetadata >> PATH_RECONNECTION_DEPTH_SHIFT) & 0xFu)
                                == 1u
                        && pathMappingControlValid(source)
                        // The first descriptor is intentionally one-hop. Reusing an already shifted
                        // source would require composition of both mappings and Jacobians.
                        && pathMappingKind(source) == PATH_MAPPING_IDENTITY
                        && sourceEvent == PATH_RECONNECTION_EVENT_DIFFUSE
                        && sourceEvent == currentEvent
                        && sourcePdf > 0.0
                        && all(isfinite(sharedVertex))
                        && all(isfinite(sharedNormalValue))
                        && sharedNormalLength >= 0.5;
                if (!sourceEdgeAvailable) {
                    state = 4u;
                    continue;
                }

                float3 sharedNormal = sharedNormalValue / sharedNormalLength;
                float3 sourceEdge = sharedVertex - sourcePositionMaterial.xyz;
                float3 receiverEdge = sharedVertex - receiverPositionMaterial.xyz;
                float sourceDistance = length(sourceEdge);
                float receiverDistance = length(receiverEdge);
                if (sourceDistance <= 1.0e-4 || receiverDistance <= 1.0e-4) {
                    state = 5u;
                    continue;
                }
                float sourceCosine = abs(dot(sharedNormal, -sourceEdge / sourceDistance));
                float receiverCosine = abs(dot(sharedNormal, receiverEdge / receiverDistance));
                float geometryJacobian = receiverCosine * sourceDistance * sourceDistance
                        / max(sourceCosine * receiverDistance * receiverDistance, 1.0e-8);

                float currentDensity = pathShiftDiffuseDensity(receiverNormal,
                        currentSecondHit - receiverPositionMaterial.xyz);
                float shiftedDensity = pathShiftDiffuseDensity(receiverNormal, receiverEdge);
                float sourceDensity = pathShiftDiffuseDensity(sourceNormal, sourceEdge);
                if (!isfinite(currentDensity) || !isfinite(shiftedDensity)
                        || !isfinite(sourceDensity) || currentDensity <= 0.0
                        || shiftedDensity <= 0.0 || sourceDensity <= 0.0) {
                    state = 6u;
                    continue;
                }

                float currentTechniqueMass = currentSourcePdf / currentDensity;
                float sourceTechniqueMass = sourcePdf / sourceDensity;
                float receiverPdf = currentTechniqueMass * shiftedDensity;
                float pssJacobian = geometryJacobian * receiverPdf / sourcePdf;
                if (!isfinite(currentTechniqueMass) || !isfinite(sourceTechniqueMass)
                        || !isfinite(receiverPdf) || currentTechniqueMass <= 0.0
                        || sourceTechniqueMass <= 0.0 || receiverPdf <= 0.0) {
                    state = 7u;
                    continue;
                }
                if (!isfinite(geometryJacobian) || !isfinite(pssJacobian)
                        || geometryJacobian <= 0.0 || pssJacobian <= 0.0) {
                    state = 8u;
                    continue;
                }
                shiftedReceiverPdf = receiverPdf;
                shiftedPssJacobian = pssJacobian;

                float3 sourceRadiance = source.canonicalRadianceTarget.xyz;
                shiftedSourceLuminance = pathLuminance(sourceRadiance);
                float3 sourceFirstThroughput = source.reconnectionThroughput.xyz;
                float3 receiverFirstThroughput = current.reconnectionThroughput.xyz;
                shiftedSource = source;
                shiftedSourcePositionMaterial = sourcePositionMaterial;
                shiftedSourceNormalRoughness = sourceNormalRoughness;
                shiftedReceiverFirstThroughput = receiverFirstThroughput;
                bool invalidSourceChannel = any(sourceRadiance > float3(0.0)
                        && sourceFirstThroughput <= float3(0.0));
                float3 throughputRatio = float3(
                        pathShiftDiffuseThroughputRatio(sourceFirstThroughput.x,
                                receiverFirstThroughput.x),
                        pathShiftDiffuseThroughputRatio(sourceFirstThroughput.y,
                                receiverFirstThroughput.y),
                        pathShiftDiffuseThroughputRatio(sourceFirstThroughput.z,
                                receiverFirstThroughput.z));
                float side = dot(receiverNormal, receiverEdge) >= 0.0 ? 1.0 : -1.0;
                float3 shadowOrigin = receiverPositionMaterial.xyz
                        + side * receiverNormal * SURF_BIAS;
                float3 toShared = sharedVertex - shadowOrigin;
                float shiftedDistance = length(toShared);
                if (shiftedDistance <= 1.0e-4) {
                    state = 5u;
                    continue;
                }
                VisibilityResult shiftedVisibility = visibility(shadowOrigin,
                        toShared / shiftedDistance, shiftedDistance * 0.999);
                if (max(shiftedVisibility.transmittance.x,
                        max(shiftedVisibility.transmittance.y,
                            shiftedVisibility.transmittance.z)) <= 0.0) {
                    state = 9u;
                    evaluated = true;
                    continue;
                }

                shiftedRadiance = sourceRadiance * throughputRatio
                        * shiftedVisibility.transmittance;
                float shiftedTarget = pathLuminance(shiftedRadiance);
                float sourceCount = min(source.weights.y, PATH_TEMPORAL_MAX_SOURCE_COUNT);
                shiftedSourceFinalWeight = source.weights.z;
                shiftedMergeWeight = shiftedTarget * shiftedSourceFinalWeight
                        * sourceCount * shiftedPssJacobian;
                shiftedCurrentWeightSum = max(current.weights.x, 0.0);
                float shiftedNextWeightSum = min(
                        shiftedCurrentWeightSum + shiftedMergeWeight, 1.0e30);
                shiftedSelectionProbability = shiftedMergeWeight > 0.0
                        && shiftedNextWeightSum > 0.0
                        ? shiftedMergeWeight / shiftedNextWeightSum : 0.0;
                if (invalidSourceChannel) {
                    state = 10u;
                } else if (!all(isfinite(sourceFirstThroughput))
                        || !all(isfinite(receiverFirstThroughput))
                        || any(sourceFirstThroughput < float3(0.0))
                        || any(receiverFirstThroughput < float3(0.0))
                        || !all(isfinite(throughputRatio))
                        || !all(isfinite(shiftedRadiance))
                        || any(throughputRatio < float3(0.0))
                        || any(shiftedRadiance < float3(0.0))) {
                    state = 13u;
                } else if (any(shiftedRadiance > float3(PATH_SHIFT_MAX_HALF))) {
                    state = 14u;
                } else if (pathLuminance(shiftedRadiance) <= 0.0) {
                    state = 11u;
                } else {
                    state = 12u;
                }
                evaluated = true;
            }
        }
    }

    // View-20 counters share the already allocated host-visible spatial diagnostic buffers. The
    // state histogram is exact; luminance samples are hash-subsampled and bounded to 4096 pairs.
    if (worldPush.pathShiftDiagnosticCounterAddr != 0u) {
        DevicePtr<uint> counters = DevicePtr<uint>(worldPush.pathShiftDiagnosticCounterAddr);
        InterlockedAdd(counters[state], 1u);
        if ((state == 12u || state == 14u)
                && worldPush.pathShiftDiagnosticPairAddr != 0u
                && isfinite(shiftedSourceLuminance)
                && shiftedSourceLuminance >= 0.0
                && isfinite(shiftedReceiverPdf)
                && shiftedReceiverPdf > 0.0
                        && isfinite(shiftedPssJacobian)
                        && shiftedPssJacobian > 0.0
                        && isfinite(shiftedSourceFinalWeight)
                        && shiftedSourceFinalWeight > 0.0
                        && isfinite(shiftedMergeWeight)
                        && shiftedMergeWeight >= 0.0
                        && isfinite(shiftedCurrentWeightSum)
                        && shiftedCurrentWeightSum >= 0.0
                        && isfinite(shiftedSelectionProbability)
                        && shiftedSelectionProbability >= 0.0
                        && shiftedSelectionProbability <= 1.0
                        && all(isfinite(shiftedRadiance))
                && all(shiftedRadiance >= float3(0.0))) {
            InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_SELECTION_ELIGIBLE], 1u);
            uint spatialSelectionSeed = pathHash(pixelIndex
                    ^ pc.frameIndex * 2246822519u ^ 0xA511E9B3u);
            bool sourceSelected = rndf(spatialSelectionSeed) < shiftedSelectionProbability;
            if (sourceSelected) {
                InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_SOURCE_SELECTED], 1u);
            }
            if (pc.pathReservoirPreviousAddr != 0u) {
                PathReservoir scratch = current;
                PathSourceRoot sourceRoot;
                bool sourceRootValid = !sourceSelected;
                if (sourceSelected && worldPush.pathShiftSourceRootAddr != 0u
                        && pc.pathQueueAddr != 0u
                        && shiftedSource.identity.y < dimensions.x * dimensions.y
                        && shiftedSource.replayControl.y >= 1u
                        && shiftedSource.replayControl.y <= MAX_PATH_SEGMENTS) {
                    ConstPtr<PackedPathSegment> queue =
                            ConstPtr<PackedPathSegment>(pc.pathQueueAddr);
                    PackedPathSegment segment0 = queue[shiftedSource.identity.y];
                    PackedPathSegment segment1 = segment0;
                    bool chainComplete = false;
                    if (shiftedSource.replayControl.y == 1u) {
                        chainComplete = segment0.nextRecord == PATH_NO_NEXT;
                    } else if (segment0.nextRecord != PATH_NO_NEXT
                            && segment0.nextRecord < dimensions.x * dimensions.y * 2u) {
                        segment1 = queue[segment0.nextRecord];
                        chainComplete = segment1.nextRecord == PATH_NO_NEXT;
                    }
                    if (chainComplete
                            && all(isfinite(segment0.ro))
                             && all(isfinite(segment1.ro))
                             && all(isfinite(shiftedSourcePositionMaterial))
                             && all(isfinite(shiftedSourceNormalRoughness))
                             && all(isfinite(receiverPositionMaterial))
                             && all(isfinite(receiverNormalRoughness))) {
                        // Persist camera-relative origins. A later cross-frame pass can reconstruct
                        // current rebased coordinates with currentCamOffset - cameraDelta without
                        // retaining the capture frame's terrain origin.
                        segment0.ro -= worldPush.camOffset;
                        segment1.ro -= worldPush.camOffset;
                        sourceRoot.sourcePositionMaterial = shiftedSourcePositionMaterial;
                         sourceRoot.sourcePositionMaterial.xyz -= worldPush.camOffset;
                         sourceRoot.sourceNormalRoughness = shiftedSourceNormalRoughness;
                         sourceRoot.receiverPositionMaterial = receiverPositionMaterial;
                         sourceRoot.receiverPositionMaterial.xyz -= worldPush.camOffset;
                         sourceRoot.receiverNormalRoughness = receiverNormalRoughness;
                         sourceRoot.segment0 = segment0;
                        sourceRoot.segment1 = segment1;
                        sourceRootValid = true;
                    }
                }
                float sourceCount = min(
                        shiftedSource.weights.y, PATH_TEMPORAL_MAX_SOURCE_COUNT);
                float nextWeightSum = min(
                        shiftedCurrentWeightSum + shiftedMergeWeight, 1.0e30);
                float nextM = min(current.weights.y + sourceCount, 16777216.0);
                if (sourceSelected) {
                    float shiftedTarget = pathLuminance(shiftedRadiance);
                    scratch.sampleRadianceTarget = float4(shiftedRadiance, shiftedTarget);
                    scratch.canonicalRadianceTarget = float4(shiftedRadiance, shiftedTarget);
                    scratch.proposalShift = float4(shiftedSource.proposalShift.x,
                            shiftedPssJacobian, max(shiftedSource.proposalShift.z, 1.0e-5),
                            asfloat(shiftedSource.replayControl.z));
                    scratch.proposalComponents = shiftedSource.proposalComponents;
                    scratch.identity = shiftedSource.identity;
                    scratch.replay = shiftedSource.replay;
                    // The compact descriptor marks a one-hop diffuse reconnection while retaining
                    // the original source seeds and terminal replay states. Generic identity replay
                    // rejects this mapping kind until a receiver-aware replay stage consumes it.
                    scratch.replayControl = shiftedSource.replayControl;
                    scratch.reconnectionVertex = shiftedSource.reconnectionVertex;
                    scratch.reconnectionNormalPdf = float4(
                            shiftedSource.reconnectionNormalPdf.xyz, shiftedReceiverPdf);
                    scratch.reconnectionThroughput = float4(
                            shiftedReceiverFirstThroughput,
                            asfloat(PATH_MAPPING_DIFFUSE_RECONNECTION));
                }
                scratch.weights.x = nextWeightSum;
                scratch.weights.y = nextM;
                float selectedTarget = scratch.sampleRadianceTarget.w;
                scratch.weights.z = selectedTarget > 0.0 && nextM > 0.0
                        ? nextWeightSum / (nextM * selectedTarget) : 0.0;
                scratch.weights.w = selectedTarget > 0.0 ? min(nextM * 0.25, 1.0) : 0.0;
                bool scratchValid = sourceRootValid
                        && all(isfinite(scratch.sampleRadianceTarget))
                        && scratch.sampleRadianceTarget.w > 0.0
                        && all(isfinite(scratch.weights))
                        && scratch.weights.x > 0.0
                        && scratch.weights.y > 0.0
                        && scratch.weights.z > 0.0
                        && scratch.weights.w > 0.0
                        && (!sourceSelected || (scratch.replayControl.x == PATH_REPLAY_VERSION
                                && asuint(scratch.proposalShift.w) == scratch.replayControl.z
                                && pathMappingControlValid(scratch)
                                && pathMappingKind(scratch)
                                        == PATH_MAPPING_DIFFUSE_RECONNECTION
                                && all(isfinite(scratch.reconnectionNormalPdf))
                                && all(isfinite(scratch.reconnectionThroughput))));
                if (scratchValid) {
                    DevicePtr<PathReservoir>(pc.pathReservoirPreviousAddr)[pixelIndex] = scratch;
                    InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_SCRATCH_WRITTEN], 1u);
                    if (sourceSelected) {
                        DevicePtr<PathSourceRoot>(worldPush.pathShiftSourceRootAddr)[pixelIndex] =
                                sourceRoot;
                        InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_SCRATCH_SELECTED], 1u);
                        InterlockedAdd(counters[PATH_SHIFT_SOURCE_ROOT_WRITTEN], 1u);
                    }
                } else {
                    InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_SCRATCH_INVALID], 1u);
                    if (sourceSelected && !sourceRootValid) {
                        InterlockedAdd(counters[PATH_SHIFT_SOURCE_ROOT_INVALID], 1u);
                    }
                }
            }
            uint sampleHash = pathHash(pixelIndex ^ pc.frameIndex * 668265263u);
            if ((sampleHash & PATH_SHIFT_DIAGNOSTIC_PAIR_MASK) == 0u) {
                uint sampleIndex;
                InterlockedAdd(counters[PATH_SHIFT_DIAGNOSTIC_PAIR_CURSOR], 1u, sampleIndex);
                if (sampleIndex < PATH_SPATIAL_DIAGNOSTIC_PAIR_CAPACITY) {
                    DevicePtr<float2> samples =
                            DevicePtr<float2>(worldPush.pathShiftDiagnosticPairAddr);
                    samples[sampleIndex] = float2(pathLuminance(shiftedRadiance),
                            shiftedSourceLuminance);
                    // The second half of this already bounded buffer is reserved for the
                    // receiver-side target/PDF contract; view 17/18 use it only in their own
                    // diagnostic passes.
                    samples[PATH_SPATIAL_DIAGNOSTIC_PAIR_CAPACITY + sampleIndex] =
                            float2(shiftedReceiverPdf, shiftedPssJacobian);
                    samples[PATH_SHIFT_DIAGNOSTIC_MERGE_PAIR_OFFSET + sampleIndex] =
                            float2(shiftedSourceFinalWeight, shiftedMergeWeight);
                    samples[PATH_SHIFT_DIAGNOSTIC_SELECTION_PAIR_OFFSET + sampleIndex] =
                            float2(shiftedCurrentWeightSum, shiftedSelectionProbability);
                }
            }
        }
    }

    float3 color;
    if (state == 0u) color = float3(0.0);                         // receiver/current empty
    else if (state == 1u) color = float3(0.10, 0.10, 0.10);      // no strict diffuse pair
    else if (state == 2u) color = float3(1.00, 0.12, 0.55);      // current edge missing
    else if (state == 3u) color = float3(0.15, 0.35, 0.95);      // strict compatibility reject
    else if (state == 4u) color = float3(0.95, 0.72, 0.04);      // non-diffuse/unsupported event
    else if (state == 5u) color = float3(0.90, 0.06, 0.05);      // invalid geometry
    else if (state == 6u) color = float3(0.05, 0.85, 0.95);      // invalid directional PDF
    else if (state == 7u) color = float3(0.95, 0.30, 0.04);      // invalid technique mass
    else if (state == 8u) color = float3(0.80, 0.80, 0.80);      // invalid Jacobian
    else if (state == 9u) color = float3(0.65, 0.05, 0.90);      // shifted edge occluded
    else if (state == 10u) color = float3(0.95, 0.35, 0.55);     // spectral support mismatch
    else if (state == 11u) color = float3(0.02, 0.16, 0.04);     // valid visible zero target
    else if (state == 12u) color = shiftedRadiance;              // visible positive shifted target
    else if (state == 13u) color = float3(0.10, 0.95, 0.65);     // invalid arithmetic
    else color = float3(1.0);                                    // exceeds R16F output range
    outImage[texel] = float4(color, 1.0);
}
