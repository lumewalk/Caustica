package dev.comfyfluffy.caustica.rt;

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
    void reconnectionJacobianUsesSolidAngleGeometryAndPssPdfRatio() {
        var geometry = new RtPathSpatialReuseReference.ReconnectionGeometry(
                2.0, 4.0, 0.5, 0.25, 0.2, 0.4);
        assertEquals(0.125, geometry.solidAngleJacobian(), 1.0e-12);
        assertEquals(0.25, geometry.primarySampleJacobian(), 1.0e-12);
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
    void persistentReconnectionLaneSelectsAndPacksTheSecondPathHit() {
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(0));
        assertTrue(RtPathSpatialReuseReference.captureReconnectionAtDepth(1));
        assertFalse(RtPathSpatialReuseReference.captureReconnectionAtDepth(2));

        int packed = RtPathSpatialReuseReference.packReconnectionMetadata(1, true);
        assertTrue(RtPathSpatialReuseReference.reconnectionValid(packed));
        assertEquals(1, RtPathSpatialReuseReference.reconnectionDepth(packed));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.packReconnectionMetadata(-1, true));
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
        assertEquals(9, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_CURSOR_INDEX);
        assertEquals(10, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(10 * Integer.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_BYTES);
        assertEquals(4096, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        assertEquals(4096 * 2 * Float.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_BYTES);
    }
}
