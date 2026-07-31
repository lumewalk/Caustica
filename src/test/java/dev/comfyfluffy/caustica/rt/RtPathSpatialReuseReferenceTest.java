package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.gen.PathSourceRootData;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtPathSpatialReuseReferenceTest {
    private static RtPathSpatialReuseReference.Surface surface(
            long material, double normalCosine, double relativeDepth, double footprint) {
        return surface(material, normalCosine, relativeDepth, 2, 42L, 2, footprint);
    }

    private static RtPathSpatialReuseReference.Surface surface(
            long material, double normalCosine, double relativeDepth, int depth,
            long topology, int transport, double footprint) {
        return new RtPathSpatialReuseReference.Surface(
                material, normalCosine, relativeDepth, depth, topology, transport, footprint);
    }

    @Test
    void compatibleNeighborRequiresStrictSurfaceAndPathIdentity() {
        var receiver = surface(7L, 0.95, 0.02, 1.0);
        assertEquals(RtPathSpatialReuseReference.Decision.ACCEPTED,
                RtPathSpatialReuseReference.admit(receiver, surface(7L, 0.90, 0.05, 3.9)));
        assertEquals(RtPathSpatialReuseReference.Decision.MATERIAL_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, surface(8L, 0.95, 0.02, 1.0)));
        assertEquals(RtPathSpatialReuseReference.Decision.SURFACE_NORMAL_MISMATCH,
                RtPathSpatialReuseReference.admit(surface(7L, 0.84, 0.02, 1.0), receiver));
        assertEquals(RtPathSpatialReuseReference.Decision.RELATIVE_DEPTH_MISMATCH,
                RtPathSpatialReuseReference.admit(surface(7L, 0.95, 0.11, 1.0), receiver));
        assertEquals(RtPathSpatialReuseReference.Decision.FOOTPRINT_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, surface(7L, 0.95, 0.02, 4.1)));
    }

    @Test
    void limitedTopologyPolicyOnlyRescuesTopologyMismatch() {
        var receiver = surface(7L, 0.95, 0.02, 2, 42L, 2, 1.0);
        var topologyMismatch = surface(7L, 0.95, 0.02, 2, 99L, 2, 1.0);
        var depthMismatch = surface(7L, 0.95, 0.02, 3, 42L, 2, 1.0);
        var transportMismatch = surface(7L, 0.95, 0.02, 2, 42L, 3, 1.0);
        var footprintMismatch = surface(7L, 0.95, 0.02, 2, 42L, 2, 4.1);

        assertEquals(RtPathSpatialReuseReference.Decision.TOPOLOGY_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, topologyMismatch,
                        RtPathSpatialReuseReference.AdmissionPolicy.STRICT));
        assertEquals(RtPathSpatialReuseReference.Decision.ACCEPTED,
                RtPathSpatialReuseReference.admit(receiver, topologyMismatch,
                        RtPathSpatialReuseReference.AdmissionPolicy.LIMITED_TOPOLOGY));
        assertTrue(RtPathSpatialReuseReference.comparePolicies(receiver, topologyMismatch)
                .topologyRescued());

        assertEquals(RtPathSpatialReuseReference.Decision.DEPTH_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, depthMismatch,
                        RtPathSpatialReuseReference.AdmissionPolicy.LIMITED_TOPOLOGY));
        assertEquals(RtPathSpatialReuseReference.Decision.TRANSPORT_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, transportMismatch,
                        RtPathSpatialReuseReference.AdmissionPolicy.LIMITED_TOPOLOGY));
        assertEquals(RtPathSpatialReuseReference.Decision.FOOTPRINT_MISMATCH,
                RtPathSpatialReuseReference.admit(receiver, footprintMismatch,
                        RtPathSpatialReuseReference.AdmissionPolicy.LIMITED_TOPOLOGY));
    }

    @Test
    void deterministicPolicyComparisonReportsOnlyTopologyRescues() {
        var receiver = surface(7L, 0.95, 0.02, 2, 42L, 2, 1.0);
        var counters = new RtPathSpatialReuseReference.PolicyComparisonCounters();
        counters.add(receiver, surface(7L, 0.95, 0.02, 2, 42L, 2, 1.0));
        counters.add(receiver, surface(7L, 0.95, 0.02, 2, 99L, 2, 1.0));
        counters.add(receiver, surface(7L, 0.95, 0.02, 3, 42L, 2, 1.0));
        counters.add(receiver, surface(7L, 0.95, 0.02, 2, 42L, 3, 1.0));
        counters.add(receiver, surface(7L, 0.95, 0.02, 2, 42L, 2, 4.1));
        counters.add(receiver, surface(8L, 0.95, 0.02, 2, 42L, 2, 1.0));
        counters.add(surface(7L, 0.84, 0.02, 2, 42L, 2, 1.0), receiver);

        assertEquals(7L, counters.samples());
        assertEquals(1L, counters.strictAccepted());
        assertEquals(2L, counters.limitedAccepted());
        assertEquals(1L, counters.topologyRescued());
        assertEquals(1.0 / 7.0, counters.strictAcceptanceRatio(), 1.0e-12);
        assertEquals(2.0 / 7.0, counters.limitedAcceptanceRatio(), 1.0e-12);
        assertEquals(1.0 / 7.0, counters.topologyRescueRatio(), 1.0e-12);
    }

    @Test
    void reconnectionJacobianUsesSolidAngleGeometryAndPssPdfRatio() {
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                2.0, 4.0, 0.5, 0.25, 0.2, 0.4);
        assertEquals(0.125, geometry.solidAngleJacobian(), 1.0e-12);
        assertEquals(0.25, geometry.primarySampleJacobian(), 1.0e-12);
    }

    @Test
    void spatialGrisWeightUsesShiftedTargetAndPssJacobianOnce() {
        var weight = new RtPathSpatialReuseReference.SpatialGrisWeight(
                3.0, 2.0, 12.0, 8.0, 0.25);
        assertEquals(8.0, weight.clampedSourceCount(), 1.0e-12);
        assertEquals(12.0, weight.mergeWeight(), 1.0e-12);
        assertEquals(0.75, weight.selectionProbability(4.0), 1.0e-12);
        assertTrue(weight.selectsSource(4.0, 0.74));
        assertFalse(weight.selectsSource(4.0, 0.75));
        var selected = weight.scratchMerge(4.0, 2.0, 5.0, 0.74);
        assertEquals(16.0, selected.weightSum(), 1.0e-12);
        assertEquals(10.0, selected.effectiveCount(), 1.0e-12);
        assertEquals(16.0 / 30.0, selected.finalWeight(), 1.0e-12);
        assertTrue(selected.sourceSelected());
        var retained = weight.scratchMerge(4.0, 2.0, 5.0, 0.75);
        assertEquals(16.0 / 50.0, retained.finalWeight(), 1.0e-12);
        assertFalse(retained.sourceSelected());
    }

    @Test
    void spatialGrisWeightKeepsEmptyOrZeroTargetMergesAtZero() {
        assertEquals(0.0, new RtPathSpatialReuseReference.SpatialGrisWeight(
                0.0, 2.0, 1.0, 8.0, 1.0).mergeWeight(), 1.0e-12);
        assertEquals(0.0, new RtPathSpatialReuseReference.SpatialGrisWeight(
                3.0, 2.0, 0.0, 8.0, 1.0).mergeWeight(), 1.0e-12);
    }

    @Test
    void spatialGrisWeightRejectsInvalidOrOverflowingTerms() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.SpatialGrisWeight(
                        1.0, 1.0, 1.0, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.SpatialGrisWeight(
                        1.0, 1.0, 1.0, 1.0, Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.SpatialGrisWeight(
                Double.MAX_VALUE, Double.MAX_VALUE, 1.0, 1.0, 1.0).mergeWeight());
        var valid = new RtPathSpatialReuseReference.SpatialGrisWeight(
                1.0, 1.0, 1.0, 1.0, 1.0);
        assertThrows(IllegalArgumentException.class, () -> valid.selectionProbability(-1.0));
        assertThrows(IllegalArgumentException.class, () -> valid.selectsSource(1.0, 1.0));
    }

    @Test
    void diffuseShiftDensityReconstructsReceiverPdfAndPssJacobian() {
        var density = new RtPathSpatialReuseReference.DiffuseShiftDensity(
                0.25, 0.50, 0.40, 0.60, 0.80);
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                2.0, 4.0, 0.5, 0.25, 0.2, 0.4);
        assertEquals(2.4, density.currentTechniqueMass(), 1.0e-12);
        assertEquals(2.0, density.sourceTechniqueMass(), 1.0e-12);
        assertEquals(1.2, density.receiverDirectionalPdf(), 1.0e-12);
        assertEquals(0.1875, density.primarySampleJacobian(geometry), 1.0e-12);
        assertEquals(0.5 / Math.PI,
                RtPathSpatialReuseReference.diffuseShiftDirectionalDensity(0.5), 1.0e-12);
        assertEquals(0.5 / Math.PI,
                RtPathSpatialReuseReference.diffuseShiftDirectionalDensity(-0.5), 1.0e-12);
    }

    @Test
    void diffuseShiftDensityRejectsUnsupportedTermsBeforeWeighting() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseShiftDensity(
                        0.0, 0.5, 0.4, 0.6, 0.8));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseShiftDensity(
                        0.25, 0.5, 0.4, 0.6, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.diffuseShiftDirectionalDensity(1.01));
    }

    @Test
    void invalidReconnectionTermsAreRejectedBeforeAReuseDecision() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.ReconnectionGeometry(
                        0.0, 1.0, 1.0, 1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.Surface(
                        1L, 0.9, 0.0, 1, 1L, 1, Double.NaN));
    }

    @Test
    void diffuseShiftReplacesSourceThroughputAndAppliesVisibility() {
        assertEquals(6.0, RtPathSpatialReuseReference.shiftedDiffuseRadiance(
                4.0, 0.25, 0.75, 0.5), 1.0e-12);
        assertEquals(0.0, RtPathSpatialReuseReference.shiftedDiffuseRadiance(
                4.0, 0.25, 0.75, 0.0), 1.0e-12);
        assertEquals(0.5, RtPathSpatialReuseReference.shiftedDiffuseRadiance(
                6.0e-8, 6.0e-8, 0.5, 1.0), 1.0e-12);
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.shiftedDiffuseRadiance(
                        1.0, 0.0, 0.5, 1.0));
    }

    @Test
    void diffuseShiftReceiverExcludesConsumedPrimaryInterfaces() {
        assertTrue(RtPathSpatialReuseReference.supportsDiffuseShiftReceiver(0));
        assertFalse(RtPathSpatialReuseReference.supportsDiffuseShiftReceiver(1));
        assertTrue(RtPathSpatialReuseReference.supportsDiffuseShiftReceiver(2));
        assertFalse(RtPathSpatialReuseReference.supportsDiffuseShiftReceiver(3));
    }

    @Test
    void receiverMaterialIdentityRoundTripsThroughExactR32fIntegerRange() {
        int packed = RtPathSpatialReuseReference.packReceiverMaterialIdentity(
                3, RtPathSpatialReuseReference.RECEIVER_MATERIAL_KEY_MASK);
        assertEquals(0x00FF_FFFF, packed);
        assertEquals(3, RtPathSpatialReuseReference.receiverMaterialModel(packed));
        assertEquals(RtPathSpatialReuseReference.RECEIVER_MATERIAL_KEY_MASK,
                RtPathSpatialReuseReference.receiverMaterialKey(packed));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packReceiverMaterialIdentity(4, 0));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packReceiverMaterialIdentity(0, 0x0040_0000));
    }

    @Test
    void persistentReconnectionLaneSelectsAndPacksTheSecondPathHit() {
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(0));
        assertTrue(RtPathSpatialReuseReference.captureReconnectionAtDepth(1));
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(2));
        assertFalse(RtPathSpatialReuseReference.reconnectionEndpointEligible(0));
        assertTrue(RtPathSpatialReuseReference.reconnectionEndpointEligible(1));
        assertTrue(RtPathSpatialReuseReference.reconnectionEndpointEligible(2));

        int packed = RtPathSpatialReuseReference.packReconnectionMetadata(
                1, RtPathSpatialReuseReference.ReconnectionEvent.GLOSSY, true);
        assertTrue(RtPathSpatialReuseReference.reconnectionValid(packed));
        assertEquals(1, RtPathSpatialReuseReference.reconnectionDepth(packed));
        assertEquals(RtPathSpatialReuseReference.ReconnectionEvent.GLOSSY,
                RtPathSpatialReuseReference.reconnectionEvent(packed));
        assertTrue(RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE.continuous());
        assertFalse(RtPathSpatialReuseReference.ReconnectionEvent.DELTA.continuous());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packReconnectionMetadata(-1, true));
    }

    @Test
    void spatialMappingControlIsCompactStrictAndOneHop() {
        int identity = RtPathSpatialReuseReference.packMappingControl(
                RtPathSpatialReuseReference.MappingKind.IDENTITY);
        int diffuse = RtPathSpatialReuseReference.packMappingControl(
                RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION);

        assertEquals(0, identity);
        assertEquals(1, diffuse);
        assertTrue(RtPathSpatialReuseReference.mappingControlValid(identity));
        assertTrue(RtPathSpatialReuseReference.mappingControlValid(diffuse));
        assertEquals(RtPathSpatialReuseReference.MappingKind.IDENTITY,
                RtPathSpatialReuseReference.mappingKind(identity));
        assertEquals(RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                RtPathSpatialReuseReference.mappingKind(diffuse));
        assertTrue(RtPathSpatialReuseReference.genericIdentityReplayEligible(identity));
        assertFalse(RtPathSpatialReuseReference.genericIdentityReplayEligible(diffuse));
        assertTrue(RtPathSpatialReuseReference.oneHopSpatialSourceEligible(identity));
        assertFalse(RtPathSpatialReuseReference.oneHopSpatialSourceEligible(diffuse));
        assertFalse(RtPathSpatialReuseReference.mappingControlValid(2));
        assertFalse(RtPathSpatialReuseReference.mappingControlValid(0x10));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.mappingKind(0x10));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packMappingControl(null));
    }

    @Test
    void receiverAwareDiffuseReplayReconstructsAndValidatesStoredShift() {
        var replay = new RtPathSpatialReuseReference.DiffuseMappingReplay(
                RtPathReplayReference.REPLAY_VERSION,
                RtPathSpatialReuseReference.packMappingControl(
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION),
                RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE,
                0.2, 1.5,
                new RtPathSpatialReuseReference.Rgb(8.0, 4.0, 2.0),
                new RtPathSpatialReuseReference.Rgb(2.0, 1.0, 0.5),
                new RtPathSpatialReuseReference.Rgb(1.0, 2.0, 1.0),
                new RtPathSpatialReuseReference.Rgb(0.5, 0.5, 0.5));

        var shifted = replay.shiftedRadiance();
        assertEquals(2.0, shifted.r(), 1.0e-12);
        assertEquals(4.0, shifted.g(), 1.0e-12);
        assertEquals(2.0, shifted.b(), 1.0e-12);
        assertEquals(0.2126 * 2.0 + 0.7152 * 4.0 + 0.0722 * 2.0,
                replay.shiftedTarget(), 1.0e-12);
        assertTrue(replay.matchesStoredShift(new RtPathSpatialReuseReference.Rgb(
                2.00001, 3.99999, 2.0)));
        assertFalse(replay.matchesStoredShift(new RtPathSpatialReuseReference.Rgb(
                2.1, 4.0, 2.0)));
    }

    @Test
    void receiverAwareReplayRejectsWrongAbiMappingEventAndSpectralSupport() {
        int diffuse = RtPathSpatialReuseReference.packMappingControl(
                RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION);
        var radiance = new RtPathSpatialReuseReference.Rgb(1.0, 0.0, 0.0);
        var throughput = new RtPathSpatialReuseReference.Rgb(1.0, 1.0, 1.0);
        var visibility = new RtPathSpatialReuseReference.Rgb(1.0, 1.0, 1.0);

        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseMappingReplay(
                        RtPathReplayReference.REPLAY_VERSION - 1, diffuse,
                        RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE,
                        0.2, 1.0, radiance, throughput, throughput, visibility));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseMappingReplay(
                        RtPathReplayReference.REPLAY_VERSION, 0,
                        RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE,
                        0.2, 1.0, radiance, throughput, throughput, visibility));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseMappingReplay(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        RtPathSpatialReuseReference.ReconnectionEvent.GLOSSY,
                        0.2, 1.0, radiance, throughput, throughput, visibility));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseMappingReplay(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE,
                        0.0, 1.0, radiance, throughput, throughput, visibility));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.DiffuseMappingReplay(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        RtPathSpatialReuseReference.ReconnectionEvent.DIFFUSE,
                        0.2, 1.0, radiance,
                        new RtPathSpatialReuseReference.Rgb(0.0, 1.0, 1.0),
                        throughput, visibility));
    }

    @Test
    void persistentDiffuseRemapRecomputesRatherThanComposesJacobian() {
        int diffuse = RtPathSpatialReuseReference.packMappingControl(
                RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION);
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                2.0, 1.0, 0.5, 0.25, 0.4, 0.2);
        var density = new RtPathSpatialReuseReference.DiffuseShiftDensity(
                1.0, 1.0, 1.0, 0.2, 0.4);
        var sourceRadiance = new RtPathSpatialReuseReference.Rgb(8.0, 4.0, 2.0);
        var sourceThroughput = new RtPathSpatialReuseReference.Rgb(2.0, 1.0, 0.5);
        var receiverThroughput = new RtPathSpatialReuseReference.Rgb(1.0, 2.0, 1.0);
        var visibility = new RtPathSpatialReuseReference.Rgb(0.5, 0.5, 0.5);
        var remap = new RtPathSpatialReuseReference.PersistentDiffuseRemap(
                RtPathReplayReference.REPLAY_VERSION, diffuse,
                true, true, true,
                9.0, geometry, density,
                sourceRadiance, sourceThroughput, receiverThroughput, visibility);

        assertEquals(1.0, remap.currentPrimarySampleJacobian(), 1.0e-12);
        assertEquals(2.0, remap.currentReplay().shiftedRadiance().r(), 1.0e-12);
        assertEquals(4.0, remap.currentReplay().shiftedRadiance().g(), 1.0e-12);
        assertEquals(2.0, remap.currentReplay().shiftedRadiance().b(), 1.0e-12);
    }

    @Test
    void persistentDiffuseRemapRejectsUnstableReceiverOrSourceRoot() {
        int diffuse = RtPathSpatialReuseReference.packMappingControl(
                RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION);
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        var density = new RtPathSpatialReuseReference.DiffuseShiftDensity(
                1.0, 1.0, 1.0, 1.0, 1.0);
        var rgb = new RtPathSpatialReuseReference.Rgb(1.0, 1.0, 1.0);

        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.PersistentDiffuseRemap(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        false, true, true, 1.0,
                        geometry, density, rgb, rgb, rgb, rgb));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.PersistentDiffuseRemap(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        true, false, true, 1.0,
                        geometry, density, rgb, rgb, rgb, rgb));
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathSpatialReuseReference.PersistentDiffuseRemap(
                        RtPathReplayReference.REPLAY_VERSION, diffuse,
                        true, true, false, 1.0,
                        geometry, density, rgb, rgb, rgb, rgb));
    }

    @Test
    void crossFrameReceiverPolicyReportsTheFirstExclusiveReject() {
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.NONE,
                receiverPolicy(true, true, true, true, true, true, true).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.SURFACE,
                receiverPolicy(false, false, false, false, false, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.SAMPLE,
                receiverPolicy(true, false, false, false, false, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.EDGE,
                receiverPolicy(true, true, false, false, false, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.TOPOLOGY,
                receiverPolicy(true, true, true, false, false, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.DEPTH,
                receiverPolicy(true, true, true, true, false, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.TRANSPORT,
                receiverPolicy(true, true, true, true, true, false, false).firstReject());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.FOOTPRINT,
                receiverPolicy(true, true, true, true, true, true, false).firstReject());
    }

    @Test
    void crossFrameSampleRescueIsShadowOnlyAndReportsTheNextReject() {
        var rescued = receiverPolicy(true, false, true, true, true, true, true);
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.SAMPLE,
                rescued.firstReject());
        assertTrue(rescued.sampleRescueEligible());
        assertTrue(rescued.sampleRescued());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.NONE,
                rescued.sampleRescueReject());

        var edgeReject = receiverPolicy(true, false, false, false, false, false, false);
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.SAMPLE,
                edgeReject.firstReject());
        assertFalse(edgeReject.sampleRescued());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.EDGE,
                edgeReject.sampleRescueReject());

        var surfaceReject = receiverPolicy(false, false, true, true, true, true, true);
        assertFalse(surfaceReject.sampleRescueEligible());
        assertEquals(RtPathSpatialReuseReference.CrossFrameReceiverReject.SURFACE,
                surfaceReject.sampleRescueReject());
    }

    private static RtPathSpatialReuseReference.CrossFrameReceiverPolicy receiverPolicy(
            boolean surface, boolean sample, boolean edge, boolean topology,
            boolean depth, boolean transport, boolean footprint) {
        return new RtPathSpatialReuseReference.CrossFrameReceiverPolicy(
                surface, sample, edge, topology, depth, transport, footprint);
    }

    @Test
    void persistentSourceRootOriginSurvivesCameraAndTerrainRebase() {
        var stored = RtPathSpatialReuseReference.PersistentSourceRootOrigin.capture(
                15.0, 4.0, -7.0,
                10.0, 2.0, -10.0);
        assertEquals(5.0, stored.cameraRelativeX(), 1.0e-12);
        assertEquals(2.0, stored.cameraRelativeY(), 1.0e-12);
        assertEquals(3.0, stored.cameraRelativeZ(), 1.0e-12);

        var sameFrame = stored.reconstruct(10.0, 2.0, -10.0, 0.0, 0.0, 0.0);
        assertEquals(15.0, sameFrame.cameraRelativeX(), 1.0e-12);
        assertEquals(4.0, sameFrame.cameraRelativeY(), 1.0e-12);
        assertEquals(-7.0, sameFrame.cameraRelativeZ(), 1.0e-12);

        // Camera moved by (+3,+1,-2), while the terrain rebase independently changed the current
        // camera offset to (5,-4,8). The retained world root reconstructs in that new rebase.
        var nextFrame = stored.reconstruct(5.0, -4.0, 8.0, 3.0, 1.0, -2.0);
        assertEquals(7.0, nextFrame.cameraRelativeX(), 1.0e-12);
        assertEquals(-3.0, nextFrame.cameraRelativeY(), 1.0e-12);
        assertEquals(13.0, nextFrame.cameraRelativeZ(), 1.0e-12);

        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.PersistentSourceRootOrigin.capture(
                        Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0));
    }

    @Test
    void pairedMomentsExposeCovarianceAndCorrelationWithoutBatchStorage() {
        var moments = new RtPathSpatialReuseReference.PairMoments();
        moments.add(1.0, 2.0);
        moments.add(2.0, 4.0);
        moments.add(3.0, 6.0);

        assertEquals(3L, moments.count());
        assertEquals(2.0, moments.meanX(), 1.0e-12);
        assertEquals(4.0, moments.meanY(), 1.0e-12);
        assertEquals(1.0, moments.varianceX(), 1.0e-12);
        assertEquals(4.0, moments.varianceY(), 1.0e-12);
        assertEquals(2.0, moments.covariance(), 1.0e-12);
        assertEquals(1.0, moments.correlation(), 1.0e-12);
    }

    @Test
    void covarianceContractRejectsNonFinitePairsAndZeroVarianceCorrelation() {
        var moments = new RtPathSpatialReuseReference.PairMoments();
        assertThrows(IllegalArgumentException.class,
                () -> moments.add(Double.NaN, 1.0));
        moments.add(2.0, 4.0);
        moments.add(2.0, 8.0);
        assertTrue(Double.isNaN(moments.correlation()));
        assertTrue(Double.isNaN(new RtPathSpatialReuseReference.PairMoments().covariance()));
    }

    @Test
    void diagnosticCountersProduceResolutionIndependentCategoryRatios() {
        var counters = new RtPathSpatialReuseReference.DiagnosticCounters();
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.TOPOLOGY_REJECT);
        counters.add(RtPathSpatialReuseReference.DiagnosticCategory.RECEIVER_EMPTY);

        assertEquals(4L, counters.total());
        assertEquals(2L, counters.count(
                RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION));
        assertEquals(0.5, counters.ratio(
                RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION), 1.0e-12);
        assertTrue(Double.isNaN(new RtPathSpatialReuseReference.DiagnosticCounters().ratio(
                RtPathSpatialReuseReference.DiagnosticCategory.SURFACE_REJECT)));
    }

    @Test
    void diagnosticCategoryOrderMatchesGpuAtomicCounterContract() {
        assertEquals(0, RtPathSpatialReuseReference.DiagnosticCategory.RECEIVER_EMPTY.ordinal());
        assertEquals(1, RtPathSpatialReuseReference.DiagnosticCategory.ACCEPTED_RECONNECTION.ordinal());
        assertEquals(2, RtPathSpatialReuseReference.DiagnosticCategory.FOOTPRINT_REJECT.ordinal());
        assertEquals(3, RtPathSpatialReuseReference.DiagnosticCategory.DEPTH_REJECT.ordinal());
        assertEquals(4, RtPathSpatialReuseReference.DiagnosticCategory.TOPOLOGY_REJECT.ordinal());
        assertEquals(5, RtPathSpatialReuseReference.DiagnosticCategory.TRANSPORT_REJECT.ordinal());
        assertEquals(6, RtPathSpatialReuseReference.DiagnosticCategory.COMPATIBLE_NO_RECONNECTION.ordinal());
        assertEquals(7, RtPathSpatialReuseReference.DiagnosticCategory.COMPATIBLE_NEIGHBOR_EMPTY.ordinal());
        assertEquals(8, RtPathSpatialReuseReference.DiagnosticCategory.SURFACE_REJECT.ordinal());
        assertEquals(9, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_CATEGORY_COUNT);
        assertEquals(17, RtPathReservoirHistory.SPATIAL_DEBUG_VIEW);
        assertEquals(18, RtPathReservoirHistory.SPATIAL_POLICY_DEBUG_VIEW);
        assertEquals(19, RtPathReservoirHistory.RECONNECTION_DEBUG_VIEW);
        assertEquals(20, RtPathReservoirHistory.SHIFTED_RADIANCE_DEBUG_VIEW);
        assertEquals(32, RtPathReservoirHistory.MAPPING_REPLAY_PASS_FLAG);
        assertEquals(64, RtPathReservoirHistory.CROSS_FRAME_MAPPING_REPLAY_PASS_FLAG);
        assertEquals(9, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_STRICT_PAIR_CURSOR_INDEX);
        assertEquals(10, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_LIMITED_ADMITTED_INDEX);
        assertEquals(11, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_TOPOLOGY_RESCUED_INDEX);
        assertEquals(12, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_RESCUED_PAIR_CURSOR_INDEX);
        assertEquals(13, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(15, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_STATE_COUNT);
        assertEquals(15, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SELECTION_ELIGIBLE_INDEX);
        assertEquals(16, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SOURCE_SELECTED_INDEX);
        assertEquals(17, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SCRATCH_WRITTEN_INDEX);
        assertEquals(18, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SCRATCH_SELECTED_INDEX);
        assertEquals(19, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SCRATCH_INVALID_INDEX);
        assertEquals(20, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_PAIR_CURSOR_INDEX);
        assertEquals(21, RtPathReservoirHistory.MAPPING_REPLAY_ELIGIBLE_INDEX);
        assertEquals(22, RtPathReservoirHistory.MAPPING_REPLAY_ACCEPTED_INDEX);
        assertEquals(23, RtPathReservoirHistory.MAPPING_REPLAY_ABI_REJECT_INDEX);
        assertEquals(24, RtPathReservoirHistory.MAPPING_REPLAY_SOURCE_REJECT_INDEX);
        assertEquals(25, RtPathReservoirHistory.MAPPING_REPLAY_RECEIVER_REJECT_INDEX);
        assertEquals(26, RtPathReservoirHistory.MAPPING_REPLAY_GEOMETRY_REJECT_INDEX);
        assertEquals(27, RtPathReservoirHistory.MAPPING_REPLAY_PDF_REJECT_INDEX);
        assertEquals(28, RtPathReservoirHistory.MAPPING_REPLAY_VISIBILITY_REJECT_INDEX);
        assertEquals(29, RtPathReservoirHistory.MAPPING_REPLAY_RADIANCE_REJECT_INDEX);
        assertEquals(30, RtPathReservoirHistory.SHIFTED_SOURCE_ROOT_WRITTEN_INDEX);
        assertEquals(31, RtPathReservoirHistory.SHIFTED_SOURCE_ROOT_INVALID_INDEX);
        assertEquals(32, RtPathReservoirHistory.MAPPING_REPLAY_SOURCE_ROOT_REJECT_INDEX);
        assertEquals(33, RtPathReservoirHistory.CROSS_FRAME_ATTEMPTED_INDEX);
        assertEquals(34,
                RtPathReservoirHistory.CROSS_FRAME_RECEIVER_REPROJECTION_REJECT_INDEX);
        assertEquals(35, RtPathReservoirHistory.CROSS_FRAME_MAPPED_EMPTY_INDEX);
        assertEquals(36, RtPathReservoirHistory.CROSS_FRAME_ELIGIBLE_INDEX);
        assertEquals(37, RtPathReservoirHistory.CROSS_FRAME_ACCEPTED_INDEX);
        assertEquals(38, RtPathReservoirHistory.CROSS_FRAME_ABI_REJECT_INDEX);
        assertEquals(39, RtPathReservoirHistory.CROSS_FRAME_SOURCE_ROOT_REJECT_INDEX);
        assertEquals(40,
                RtPathReservoirHistory.CROSS_FRAME_SOURCE_REPROJECTION_REJECT_INDEX);
        assertEquals(41, RtPathReservoirHistory.CROSS_FRAME_SOURCE_REPLAY_REJECT_INDEX);
        assertEquals(42, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_SURFACE_REJECT_INDEX);
        assertEquals(43, RtPathReservoirHistory.CROSS_FRAME_GEOMETRY_REJECT_INDEX);
        assertEquals(44, RtPathReservoirHistory.CROSS_FRAME_PDF_REJECT_INDEX);
        assertEquals(45, RtPathReservoirHistory.CROSS_FRAME_VISIBILITY_REJECT_INDEX);
        assertEquals(46, RtPathReservoirHistory.CROSS_FRAME_RADIANCE_REJECT_INDEX);
        assertEquals(47, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_SAMPLE_REJECT_INDEX);
        assertEquals(48, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_EDGE_REJECT_INDEX);
        assertEquals(49, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_TOPOLOGY_REJECT_INDEX);
        assertEquals(50, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_DEPTH_REJECT_INDEX);
        assertEquals(51, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_TRANSPORT_REJECT_INDEX);
        assertEquals(52, RtPathReservoirHistory.CROSS_FRAME_RECEIVER_FOOTPRINT_REJECT_INDEX);
        assertEquals(53, RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_ELIGIBLE_INDEX);
        assertEquals(54, RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUED_INDEX);
        assertEquals(55, RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_EDGE_REJECT_INDEX);
        assertEquals(56,
                RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_TOPOLOGY_REJECT_INDEX);
        assertEquals(57, RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_DEPTH_REJECT_INDEX);
        assertEquals(58,
                RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_TRANSPORT_REJECT_INDEX);
        assertEquals(59,
                RtPathReservoirHistory.CROSS_FRAME_SAMPLE_RESCUE_FOOTPRINT_REJECT_INDEX);
        assertEquals(60, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(60 * Integer.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_BYTES);
        assertEquals(160, PathSourceRootData.BYTE_SIZE);
        assertEquals(4096, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        assertEquals(4096, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_DENSITY_PAIR_OFFSET);
        assertEquals(8192, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_MERGE_PAIR_OFFSET);
        assertEquals(12288, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SELECTION_PAIR_OFFSET);
        assertEquals(4096 * 8, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_PAIR_FLOAT_COUNT);
        assertEquals(4096 * 8 * Float.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_BYTES);
    }
}
