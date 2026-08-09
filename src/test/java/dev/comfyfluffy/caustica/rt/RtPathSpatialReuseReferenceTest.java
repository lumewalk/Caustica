package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.gen.PathSourceRootData;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
        assertEquals(0.75, weight.stableSelectionProbability(4.0), 1.0e-12);
        assertTrue(weight.stableSelectsSource(4.0, 0.74));
        assertFalse(weight.stableSelectsSource(4.0, 0.75));
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
        var aboveStoredWeightCap = new RtPathSpatialReuseReference.SpatialGrisWeight(
                1.0e20, 1.0e20, 1.0, 8.0, 1.0);
        assertThrows(IllegalArgumentException.class,
                () -> aboveStoredWeightCap.selectionProbability(0.0));
        assertEquals(1.0, aboveStoredWeightCap.stableSelectionProbability(0.0), 0.0);
    }

    @Test
    void stableSpatialSelectionRatioHandlesExtremeFiniteWeightsWithoutClamping() {
        var large = new RtPathSpatialReuseReference.SpatialGrisWeight(
                1.0e40, 1.0, 1.0, 8.0, 1.0);
        assertEquals(1.0, large.stableSelectionProbability(0.0), 0.0);
        assertEquals(0.5, large.stableSelectionProbability(1.0e40), 0.0);
        assertEquals(1.0, large.stableSelectionProbability(1.0e20), 0.0);

        var small = new RtPathSpatialReuseReference.SpatialGrisWeight(
                1.0e20, 1.0, 1.0, 8.0, 1.0);
        assertEquals(1.0e-20, small.stableSelectionProbability(1.0e40), 1.0e-32);

        var maximum = new RtPathSpatialReuseReference.SpatialGrisWeight(
                Double.MAX_VALUE, 1.0, 1.0, 8.0, 1.0);
        assertEquals(0.5, maximum.stableSelectionProbability(Double.MAX_VALUE), 0.0);
        assertTrue(maximum.stableSelectsSource(0.0, Math.nextDown(1.0)));
        assertThrows(IllegalArgumentException.class,
                () -> maximum.stableSelectionProbability(-1.0));
        assertThrows(IllegalArgumentException.class,
                () -> maximum.stableSelectsSource(0.0, 1.0));

        var zero = new RtPathSpatialReuseReference.SpatialGrisWeight(
                0.0, 1.0, 1.0, 8.0, 1.0);
        assertFalse(zero.stableSelectsSource(Double.MAX_VALUE, 0.0));
    }

    @Test
    void stableShadowMergeAuditsPostSelectionArithmeticWithoutMutation() {
        var weight = new RtPathSpatialReuseReference.SpatialGrisWeight(
                3.0, 2.0, 12.0, 8.0, 0.25);
        var selected = weight.stableShadowMerge(4.0, 2.0, 5.0, 0.74);
        assertEquals(16.0, selected.weightSum(), 1.0e-12);
        assertEquals(10.0, selected.effectiveCount(), 1.0e-12);
        assertEquals(3.0, selected.selectedTarget(), 1.0e-12);
        assertEquals(16.0 / 30.0, selected.finalWeight(), 1.0e-12);
        assertTrue(selected.sourceSelected());
        assertFalse(selected.empty());

        var retained = weight.stableShadowMerge(4.0, 2.0, 5.0, 0.75);
        assertEquals(5.0, retained.selectedTarget(), 1.0e-12);
        assertEquals(16.0 / 50.0, retained.finalWeight(), 1.0e-12);
        assertFalse(retained.sourceSelected());
        assertFalse(retained.empty());

        var zero = new RtPathSpatialReuseReference.SpatialGrisWeight(
                0.0, 2.0, 1.0, 8.0, 1.0);
        var empty = zero.stableShadowMerge(0.0, 0.0, 0.0, 0.0);
        assertEquals(0.0, empty.weightSum(), 0.0);
        assertEquals(0.0, empty.finalWeight(), 0.0);
        assertTrue(empty.empty());

        assertThrows(IllegalArgumentException.class,
                () -> weight.stableShadowMerge(4.0, 2.0, 0.0, 0.75));

        var extreme = new RtPathSpatialReuseReference.SpatialGrisWeight(
                1.0e40, 1.0, 1.0, 8.0, 1.0);
        var capped = extreme.stableShadowMerge(0.0, 0.0, 0.0, 0.99);
        assertEquals(1.0e30, capped.weightSum(), 0.0);
        assertEquals(1.0, capped.effectiveCount(), 0.0);
        assertEquals(1.0e40, capped.selectedTarget(), 0.0);
        assertEquals(1.0e-10, capped.finalWeight(), 1.0e-22);
    }

    @Test
    void stableSampleScratchCopiesOnlyTheSelectedOpaquePayload() {
        var weight = new RtPathSpatialReuseReference.SpatialGrisWeight(
                3.0, 2.0, 12.0, 8.0, 0.25);
        var current = new RtPathSpatialReuseReference.SpatialSamplePayload(0xCA11L, 5.0);
        var shifted = new RtPathSpatialReuseReference.SpatialSamplePayload(0x50A2CEL, 3.0);

        var selected = weight.stableSampleScratch(4.0, 2.0,
                current, shifted, 0.74);
        assertEquals(shifted, selected.sample());
        assertTrue(selected.arithmetic().sourceSelected());
        assertEquals(16.0 / 30.0, selected.arithmetic().finalWeight(), 1.0e-12);

        var retained = weight.stableSampleScratch(4.0, 2.0,
                current, shifted, 0.75);
        assertEquals(current, retained.sample());
        assertFalse(retained.arithmetic().sourceSelected());
        assertEquals(16.0 / 50.0, retained.arithmetic().finalWeight(), 1.0e-12);

        var zero = new RtPathSpatialReuseReference.SpatialGrisWeight(
                0.0, 2.0, 1.0, 8.0, 1.0);
        var empty = zero.stableSampleScratch(0.0, 0.0,
                null, shifted, 0.0);
        assertEquals(null, empty.sample());
        assertTrue(empty.arithmetic().empty());

        assertThrows(IllegalArgumentException.class,
                () -> weight.stableSampleScratch(4.0, 2.0,
                        current, null, 0.74));
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
    void deterministicReceiverGuideNeedsF0AndExactDiffuseTerms() {
        var diffuse = new RtPathSpatialReuseReference.Rgb(0.5, 0.5, 0.5);
        var lowF0 = new RtPathSpatialReuseReference.DiffuseReceiverMaterial(
                diffuse, new RtPathSpatialReuseReference.Rgb(0.04, 0.04, 0.04), false);
        var highF0 = new RtPathSpatialReuseReference.DiffuseReceiverMaterial(
                diffuse, new RtPathSpatialReuseReference.Rgb(0.8, 0.8, 0.8), false);

        var lowGuide = lowF0.exactGuide();
        var highGuide = highF0.exactGuide();
        assertEquals(0.9, lowGuide.techniqueMass(), 1.0e-12);
        assertNotEquals(lowGuide.techniqueMass(), highGuide.techniqueMass());
        assertNotEquals(lowGuide.eventThroughput().r(), highGuide.eventThroughput().r());
        assertEquals(lowGuide.techniqueMass() * 0.5 / Math.PI,
                lowGuide.directionalPdf(0.5), 1.0e-12);
        var particleGuide = RtPathSpatialReuseReference.DiffuseReceiverGuide.particle(diffuse);
        assertEquals(1.0, particleGuide.techniqueMass(), 0.0);
        assertEquals(diffuse, particleGuide.eventThroughput());

        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.ACCEPTED,
                lowGuide.compareStored(lowGuide.eventThroughput(),
                        lowGuide.directionalPdf(0.5), 0.5));
        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.MASS_MISMATCH,
                lowGuide.compareStored(new RtPathSpatialReuseReference.Rgb(0.625, 0.625, 0.625),
                        lowGuide.directionalPdf(0.5), 0.5));
        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.THROUGHPUT_MISMATCH,
                lowGuide.compareStored(
                        new RtPathSpatialReuseReference.Rgb(
                                lowGuide.eventThroughput().r(), 0.1, 0.1),
                        lowGuide.directionalPdf(0.5), 0.5));
        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.PDF_MISMATCH,
                lowGuide.compareStored(lowGuide.eventThroughput(),
                        0.25, 0.5));

        var exactEdge = new RtPathSpatialReuseReference.DiffuseReceiverEdge(
                0.0, 0.01, 0.0, 10.0, 0.02, 0.0, 0.0, 1.0, 0.0);
        var mixedPassEdge = new RtPathSpatialReuseReference.DiffuseReceiverEdge(
                0.0, 0.0, 0.0, 10.0, 0.02, 0.0, 0.0, 1.0, 0.0);
        double exactStoredPdf = lowGuide.directionalPdf(exactEdge.cosine());
        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.ACCEPTED,
                lowGuide.compareStored(lowGuide.eventThroughput(),
                        exactStoredPdf, exactEdge.cosine()));
        assertEquals(RtPathSpatialReuseReference.ReceiverGuideComparison.PDF_MISMATCH,
                lowGuide.compareStored(lowGuide.eventThroughput(),
                        exactStoredPdf, mixedPassEdge.cosine()));

        var deltaOnly = new RtPathSpatialReuseReference.DiffuseReceiverMaterial(
                new RtPathSpatialReuseReference.Rgb(0.0, 0.0, 0.0),
                new RtPathSpatialReuseReference.Rgb(1.0, 1.0, 1.0), true);
        assertThrows(IllegalArgumentException.class, deltaOnly::exactGuide);
    }

    @Test
    void guideOnlyDiffuseRemapDefinesExactPreVisibilityChainAndRejectOrder() {
        var guide = new RtPathSpatialReuseReference.DiffuseReceiverGuide(
                new RtPathSpatialReuseReference.Rgb(0.45, 0.30, 0.15), 0.75);
        var edge = new RtPathSpatialReuseReference.DiffuseReceiverEdge(
                0.0, 0.01, 0.0, 1.0, 1.01, 0.0, 0.0, 1.0, 0.0);
        var remap = new RtPathSpatialReuseReference.GuideOnlyDiffuseRemap(
                guide, edge, 2.0, 1.0, 0.5, 0.25, 0.2,
                new RtPathSpatialReuseReference.Rgb(2.0, 1.0, 0.5),
                new RtPathSpatialReuseReference.Rgb(0.5, 0.5, 0.5));

        assertEquals(guide.directionalPdf(edge.cosine()),
                remap.receiverDirectionalPdf(), 1.0e-12);
        assertEquals(2.0 * remap.receiverDirectionalPdf() / 0.2,
                remap.primarySampleJacobian(), 1.0e-12);
        assertEquals(guide.eventThroughput(), remap.receiverThroughput());
        var shifted = remap.unoccludedShiftedRadiance();
        assertEquals(2.4, shifted.r(), 1.0e-12);
        assertEquals(0.8, shifted.g(), 1.0e-12);
        assertEquals(0.2, shifted.b(), 1.0e-12);
        var tinted = remap.shiftedRadiance(
                new RtPathSpatialReuseReference.Rgb(0.5, 0.25, 0.0));
        assertEquals(1.2, tinted.r(), 1.0e-12);
        assertEquals(0.2, tinted.g(), 1.0e-12);
        assertEquals(0.0, tinted.b(), 1.0e-12);
        assertEquals(0.39816, remap.shiftedTarget(
                new RtPathSpatialReuseReference.Rgb(0.5, 0.25, 0.0)), 1.0e-12);
        assertEquals(0.0, remap.shiftedTarget(
                new RtPathSpatialReuseReference.Rgb(0.0, 0.0, 0.0)), 1.0e-12);
        var tintedTransmission = new RtPathSpatialReuseReference.Rgb(0.5, 0.25, 0.0);
        assertEquals(0.39816 * 2.0 * 8.0 * remap.primarySampleJacobian(),
                remap.mergeWeight(tintedTransmission, 2.0, 12.0, 8.0), 1.0e-12);
        double tintedMergeWeight = remap.mergeWeight(
                tintedTransmission, 2.0, 12.0, 8.0);
        assertEquals(tintedMergeWeight / (4.0 + tintedMergeWeight),
                remap.selectionProbability(
                        tintedTransmission, 2.0, 12.0, 8.0, 4.0), 1.0e-12);
        assertEquals(tintedMergeWeight / (4.0 + tintedMergeWeight),
                remap.stableSelectionProbability(
                        tintedTransmission, 2.0, 12.0, 8.0, 4.0), 1.0e-12);
        double stableProbability = remap.stableSelectionProbability(
                tintedTransmission, 2.0, 12.0, 8.0, 4.0);
        assertTrue(remap.stableSelectsSource(
                tintedTransmission, 2.0, 12.0, 8.0, 4.0,
                Math.nextDown(stableProbability)));
        assertFalse(remap.stableSelectsSource(
                tintedTransmission, 2.0, 12.0, 8.0, 4.0, stableProbability));
        assertEquals(0.0, remap.mergeWeight(
                new RtPathSpatialReuseReference.Rgb(0.0, 0.0, 0.0),
                2.0, 12.0, 8.0), 1.0e-12);
        assertEquals(0.0, remap.selectionProbability(
                new RtPathSpatialReuseReference.Rgb(0.0, 0.0, 0.0),
                2.0, 12.0, 8.0, 4.0), 1.0e-12);
        assertThrows(IllegalArgumentException.class, () -> remap.selectionProbability(
                tintedTransmission, 2.0, 12.0, 8.0, -1.0));
        assertThrows(IllegalArgumentException.class, () -> remap.shiftedRadiance(
                new RtPathSpatialReuseReference.Rgb(1.01, 1.0, 1.0)));

        assertEquals(RtPathSpatialReuseReference.GuideOnlyRemapDecision.GEOMETRY_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyRemapPolicy(false, false, false)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyRemapDecision.PDF_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyRemapPolicy(true, false, false)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyRemapDecision.THROUGHPUT_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyRemapPolicy(true, true, false)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyRemapDecision.READY,
                new RtPathSpatialReuseReference.GuideOnlyRemapPolicy(true, true, true)
                        .firstReject());

        assertEquals(RtPathSpatialReuseReference.GuideOnlyVisibilityDecision.ARITHMETIC_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyVisibilityPolicy(false, true, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyVisibilityDecision.OCCLUDED,
                new RtPathSpatialReuseReference.GuideOnlyVisibilityPolicy(true, false, false)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyVisibilityDecision.CLEAR,
                new RtPathSpatialReuseReference.GuideOnlyVisibilityPolicy(true, true, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyVisibilityDecision.TINTED,
                new RtPathSpatialReuseReference.GuideOnlyVisibilityPolicy(true, true, false)
                        .result());

        assertEquals(RtPathSpatialReuseReference.GuideOnlyShiftedTargetDecision.ARITHMETIC_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyShiftedTargetPolicy(false, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyShiftedTargetDecision.POSITIVE,
                new RtPathSpatialReuseReference.GuideOnlyShiftedTargetPolicy(true, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyShiftedTargetDecision.ZERO,
                new RtPathSpatialReuseReference.GuideOnlyShiftedTargetPolicy(true, false)
                        .result());

        assertEquals(RtPathSpatialReuseReference.GuideOnlyMergeWeightDecision.ARITHMETIC_REJECT,
                new RtPathSpatialReuseReference.GuideOnlyMergeWeightPolicy(false, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyMergeWeightDecision.POSITIVE,
                new RtPathSpatialReuseReference.GuideOnlyMergeWeightPolicy(true, true)
                        .result());
        assertEquals(RtPathSpatialReuseReference.GuideOnlyMergeWeightDecision.ZERO,
                new RtPathSpatialReuseReference.GuideOnlyMergeWeightPolicy(true, false)
                        .result());

        assertEquals(RtPathSpatialReuseReference.GuideOnlySelectionDecision.CURRENT_WEIGHT_REJECT,
                new RtPathSpatialReuseReference.GuideOnlySelectionPolicy(
                        false, false, false, false)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlySelectionDecision.ARITHMETIC_REJECT,
                new RtPathSpatialReuseReference.GuideOnlySelectionPolicy(
                        true, false, false, false)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlySelectionDecision.PROBABILITY_HIGH_REJECT,
                new RtPathSpatialReuseReference.GuideOnlySelectionPolicy(
                        true, true, true, true)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlySelectionDecision.POSITIVE,
                new RtPathSpatialReuseReference.GuideOnlySelectionPolicy(
                        true, true, false, true)
                        .firstReject());
        assertEquals(RtPathSpatialReuseReference.GuideOnlySelectionDecision.ZERO,
                new RtPathSpatialReuseReference.GuideOnlySelectionPolicy(
                        true, true, false, false)
                        .firstReject());
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

    @Test
    void crossFrameReceiverEdgeReportsTheFirstExclusiveReject() {
        var reasons = RtPathSpatialReuseReference.CrossFrameReceiverEdgeReject.values();
        for (int reject = 0; reject < reasons.length; reject++) {
            assertEquals(reasons[reject], receiverEdgePolicy(reject).firstReject());
        }
    }

    private static RtPathSpatialReuseReference.CrossFrameReceiverEdgePolicy receiverEdgePolicy(
            int firstReject) {
        return new RtPathSpatialReuseReference.CrossFrameReceiverEdgePolicy(
                firstReject != 1,
                firstReject != 2,
                firstReject != 3,
                firstReject != 4,
                firstReject != 5,
                firstReject != 6);
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
    void guideSourceRootProvenanceAdvancesSelectedAndCapturesRetainedRoots() {
        var previous = new RtPathSpatialReuseReference.PersistentSourceRootOrigin(
                5.0, 2.0, 3.0);
        var selected = RtPathSpatialReuseReference.GuideSourceRootProvenance.selected(
                23, previous, 3.0, 1.0, -2.0);
        assertEquals(23, selected.sourcePixelIndex());
        assertEquals(2.0, selected.origin().cameraRelativeX(), 1.0e-12);
        assertEquals(1.0, selected.origin().cameraRelativeY(), 1.0e-12);
        assertEquals(5.0, selected.origin().cameraRelativeZ(), 1.0e-12);

        var retained = RtPathSpatialReuseReference.GuideSourceRootProvenance.retained(
                31, 15.0, 4.0, -7.0, 10.0, 2.0, -10.0);
        assertEquals(31, retained.sourcePixelIndex());
        assertEquals(previous, retained.origin());

        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.GuideSourceRootProvenance.selected(
                        0, null, 0.0, 0.0, 0.0));
    }

    @Test
    void guidePreviousReplayReportsTheFirstExclusiveOutcome() {
        var selected = RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION;
        var retained = RtPathSpatialReuseReference.MappingKind.IDENTITY;

        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.LIFECYCLE_REJECT,
                guideReplay(false, true, true, true, true, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.RECEIVER_REPROJECTION_REJECT,
                guideReplay(true, false, true, true, true, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.PREVIOUS_EMPTY,
                guideReplay(true, true, false, true, true, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.METADATA_REJECT,
                guideReplay(true, true, true, false, true, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.RECEIVER_SURFACE_REJECT,
                guideReplay(true, true, true, true, false, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.SOURCE_REPROJECTION_REJECT,
                guideReplay(true, true, true, true, true, false, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.SOURCE_SURFACE_REJECT,
                guideReplay(true, true, true, true, true, true, false, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.SOURCE_REPLAY_REJECT,
                guideReplay(true, true, true, true, true, true, true, false, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.SELECTED_ACCEPTED,
                guideReplay(true, true, true, true, true, true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.GuidePreviousReplayOutcome.RETAINED_ACCEPTED,
                guideReplay(true, true, true, true, true, true, true, true, retained));
        assertThrows(IllegalArgumentException.class,
                () -> guideReplay(true, true, true, true, true, true, true, true, null));
    }

    private static RtPathSpatialReuseReference.GuidePreviousReplayOutcome guideReplay(
            boolean lifecycle, boolean receiverReprojection, boolean previous,
            boolean metadata, boolean receiverSurface, boolean sourceReprojection,
            boolean sourceSurface, boolean sourceReplay,
            RtPathSpatialReuseReference.MappingKind mappingKind) {
        return RtPathSpatialReuseReference.guidePreviousReplayOutcome(
                lifecycle, receiverReprojection, previous, metadata, receiverSurface,
                sourceReprojection, sourceSurface, sourceReplay, mappingKind);
    }

    @Test
    void branchScratchStorageRequiresExactReservoirAndRootBits() {
        var selected = RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION;
        var retained = RtPathSpatialReuseReference.MappingKind.IDENTITY;
        assertEquals(RtPathSpatialReuseReference.BranchScratchStorageOutcome.METADATA_REJECT,
                RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        false, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.BranchScratchStorageOutcome.PAIR_REJECT,
                RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        true, false, true, selected));
        assertEquals(RtPathSpatialReuseReference.BranchScratchStorageOutcome.PAIR_REJECT,
                RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        true, true, false, selected));
        assertEquals(RtPathSpatialReuseReference.BranchScratchStorageOutcome.SELECTED_ACCEPTED,
                RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        true, true, true, selected));
        assertEquals(RtPathSpatialReuseReference.BranchScratchStorageOutcome.RETAINED_ACCEPTED,
                RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        true, true, true, retained));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchScratchStorageOutcome(
                        true, true, true, null));
    }

    @Test
    void branchCandidateTagBindsPreviousFrameGenerationAndMapping() {
        var selected = RtPathSpatialReuseReference.BranchCandidateTag.capture(
                41, 0x123456, RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION);
        assertEquals(41, selected.frameIndex());
        assertEquals(0x81123456, selected.control());
        assertTrue(selected.validFor(
                42, 0x123456, RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION));
        assertFalse(selected.validFor(
                43, 0x123456, RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION));
        assertFalse(selected.validFor(
                42, 0x123457, RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION));
        assertFalse(selected.validFor(
                42, 0x123456, RtPathSpatialReuseReference.MappingKind.IDENTITY));

        var wrapped = RtPathSpatialReuseReference.BranchCandidateTag.capture(
                7, 0xAB123456, RtPathSpatialReuseReference.MappingKind.IDENTITY);
        assertEquals(0x80123456, wrapped.control());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.BranchCandidateTag.capture(0, 0, null));
    }

    @Test
    void branchCandidateRetentionIsBoundedToFourLiveFrameAges() {
        var identity = RtPathSpatialReuseReference.MappingKind.IDENTITY;
        var mapped = RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION;
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.EMPTY,
                retention(false, 20, 20, 7, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.FUTURE_REJECT,
                retention(true, 21, 20, 7, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.GENERATION_REJECT,
                retention(true, 20, 20, 6, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.MAPPING_REJECT,
                retention(true, 20, 20, 7, 7, mapped, identity, mapped));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                retention(true, 20, 20, 7, 7, mapped, mapped, mapped));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE,
                retention(true, 19, 20, 7, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO,
                retention(true, 18, 20, 7, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE,
                retention(true, 17, 20, 7, 7, identity, identity, identity));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.EXPIRED,
                retention(true, 16, 20, 7, 7, identity, identity, identity));
        assertTrue(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT.live());
        assertTrue(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE.live());
        assertFalse(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.EXPIRED.live());
    }

    private static RtPathSpatialReuseReference.BranchCandidateRetentionOutcome retention(
            boolean promoted, long validationFrame, long currentFrame,
            int storedGeneration, int currentGeneration,
            RtPathSpatialReuseReference.MappingKind promotionMapping,
            RtPathSpatialReuseReference.MappingKind metadataMapping,
            RtPathSpatialReuseReference.MappingKind reservoirMapping) {
        return RtPathSpatialReuseReference.branchCandidateRetentionOutcome(
                promoted, validationFrame, currentFrame, storedGeneration, currentGeneration,
                promotionMapping, metadataMapping, reservoirMapping);
    }

    @Test
    void branchReplayCameraProvenanceAdvancesExactlyOncePerFrame() {
        var captured = RtPathSpatialReuseReference.BranchReplayCameraProvenance.capture(40L);
        var ageOne = captured.advance(41L, 1.5, -0.25, 2.0);
        var ageTwo = ageOne.advance(42L, -0.5, 0.75, 1.0);
        assertEquals(1.0, ageTwo.deltaX(), 1.0e-12);
        assertEquals(0.5, ageTwo.deltaY(), 1.0e-12);
        assertEquals(3.0, ageTwo.deltaZ(), 1.0e-12);
        assertEquals(42L, ageTwo.advancedFrame());
        assertThrows(IllegalArgumentException.class,
                () -> captured.advance(42L, 0.0, 0.0, 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> ageTwo.advance(43L, Double.NaN, 0.0, 0.0));
    }

    @Test
    void branchAgeReplayUsesOriginalRootAndRejectsStaleProvenance() {
        var identity = RtPathSpatialReuseReference.MappingKind.IDENTITY;
        var mapped = RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION;
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.EMPTY,
                ageReplay(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.EMPTY,
                        false, null, false));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.METADATA_REJECT,
                ageReplay(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.MAPPING_REJECT,
                        false, null, false));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.CURRENT_SKIP,
                ageReplay(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, identity, true));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.EXPIRED,
                ageReplay(RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.EXPIRED,
                        true, identity, true));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.PROVENANCE_REJECT,
                ageReplay(ageOne, false, mapped, true));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.IDENTITY_ACCEPTED,
                ageReplay(ageOne, true, identity, true));
        assertEquals(RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome.IDENTITY_REJECT,
                ageReplay(ageOne, true, identity, false));
        assertEquals(
                RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome
                        .MAPPED_ORIGINAL_ROOT_ACCEPTED,
                ageReplay(ageOne, true, mapped, true));
        assertEquals(
                RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome
                        .MAPPED_ORIGINAL_ROOT_REJECT,
                ageReplay(ageOne, true, mapped, false));
        assertThrows(IllegalArgumentException.class,
                () -> ageReplay(ageOne, true, null, true));
    }

    private static RtPathSpatialReuseReference.BranchCandidateAgeReplayOutcome ageReplay(
            RtPathSpatialReuseReference.BranchCandidateRetentionOutcome retention,
            boolean provenanceCurrent, RtPathSpatialReuseReference.MappingKind mappingKind,
            boolean replayMatches) {
        return RtPathSpatialReuseReference.branchCandidateAgeReplayOutcome(
                retention, provenanceCurrent, mappingKind, replayMatches);
    }

    @Test
    void branchReceiverReprojectionUsesCumulativeCameraDeltaExactlyOnce() {
        var captured = new RtPathSpatialReuseReference.BranchReceiverPosition(10.0, 20.0, 30.0);
        var currentOffset =
                new RtPathSpatialReuseReference.BranchReceiverPosition(100.0, 200.0, 300.0);
        var provenance = RtPathSpatialReuseReference.BranchReplayCameraProvenance.capture(40L)
                .advance(41L, 1.0, 2.0, 3.0)
                .advance(42L, 4.0, 5.0, 6.0);

        var current = RtPathSpatialReuseReference.branchReceiverCurrentGuidePosition(
                captured, currentOffset, provenance);
        assertEquals(105.0, current.x(), 1.0e-12);
        assertEquals(213.0, current.y(), 1.0e-12);
        assertEquals(321.0, current.z(), 1.0e-12);
    }

    @Test
    void branchReceiverAdmissionClassifiesStrictDirectProjectionInOrder() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome
                        .SOURCE_REPLAY_REJECT,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageOne, false, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.CLIP_REJECT,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageOne, true, false, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.BOUNDS_REJECT,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageOne, true, true, false, true));
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.SURFACE_REJECT,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageOne, true, true, true, false));
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.AGE_ONE_ADMITTED,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageOne, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.AGE_TWO_ADMITTED,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageTwo, true, true, true, true));
        assertEquals(
                RtPathSpatialReuseReference.BranchReceiverAdmissionOutcome.AGE_THREE_ADMITTED,
                RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        ageThree, true, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchReceiverAdmissionOutcome(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, true, true, true));
    }

    @Test
    void branchDirectRemapClassifiesPreVisibilityTermsInOrder() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome
                        .RECEIVER_ADMISSION_REJECT,
                directRemap(ageOne, false, true, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.GUIDE_REJECT,
                directRemap(ageOne, true, false, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.EDGE_REJECT,
                directRemap(ageOne, true, true, false, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.GEOMETRY_REJECT,
                directRemap(ageOne, true, true, true, false, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.PDF_REJECT,
                directRemap(ageOne, true, true, true, true, false, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.THROUGHPUT_REJECT,
                directRemap(ageOne, true, true, true, true, true, false));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.AGE_ONE_READY,
                directRemap(ageOne, true, true, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.AGE_TWO_READY,
                directRemap(ageTwo, true, true, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRemapOutcome.AGE_THREE_READY,
                directRemap(ageThree, true, true, true, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> directRemap(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, true, true, true, true, true));
    }

    private static RtPathSpatialReuseReference.BranchDirectRemapOutcome directRemap(
            RtPathSpatialReuseReference.BranchCandidateRetentionOutcome retention,
            boolean receiverAdmitted, boolean guideValid, boolean edgeValid,
            boolean geometryValid, boolean pdfValid, boolean throughputValid) {
        return RtPathSpatialReuseReference.branchDirectRemapOutcome(
                retention, receiverAdmitted, guideValid, edgeValid,
                geometryValid, pdfValid, throughputValid);
    }

    @Test
    void branchDirectTargetAuditPreservesVisibilityTargetAndAgePartitions() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome
                                .DIRECT_REMAP_REJECT,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.NOT_ELIGIBLE,
                        ageOne),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageOne, false, true, true, true, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome.INVALID,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.NOT_ELIGIBLE,
                        ageOne),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageOne, true, false, true, true, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome.CLEAR,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.POSITIVE,
                        ageOne),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageOne, true, true, true, true, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome.TINTED,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.POSITIVE,
                        ageTwo),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageTwo, true, true, true, false, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome.OCCLUDED,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.ZERO,
                        ageThree),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageThree, true, true, false, false, true, false));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchDirectVisibilityOutcome.CLEAR,
                        RtPathSpatialReuseReference.BranchDirectTargetOutcome.INVALID,
                        ageTwo),
                RtPathSpatialReuseReference.branchDirectTargetAudit(
                        ageTwo, true, true, true, true, false, true));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectTargetAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, true, true, true, true, true));
    }

    @Test
    void branchDirectWeightAuditAppliesTheEightSampleCapWithoutSelection() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(new RtPathSpatialReuseReference.BranchDirectWeightAudit(
                        RtPathSpatialReuseReference.BranchDirectWeightOutcome.TARGET_REJECT,
                        RtPathSpatialReuseReference.BranchDirectSourceCountOutcome.NOT_ELIGIBLE,
                        ageOne, 0.0),
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageOne, false, 2.0, 3.0, 4.0, 5.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectWeightAudit(
                        RtPathSpatialReuseReference.BranchDirectWeightOutcome.POSITIVE,
                        RtPathSpatialReuseReference.BranchDirectSourceCountOutcome.UNCAPPED,
                        ageOne, 120.0),
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageOne, true, 2.0, 3.0, 4.0, 5.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectWeightAudit(
                        RtPathSpatialReuseReference.BranchDirectWeightOutcome.POSITIVE,
                        RtPathSpatialReuseReference.BranchDirectSourceCountOutcome.CAPPED,
                        ageTwo, 240.0),
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageTwo, true, 2.0, 3.0, 16.0, 5.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectWeightAudit(
                        RtPathSpatialReuseReference.BranchDirectWeightOutcome.ZERO,
                        RtPathSpatialReuseReference.BranchDirectSourceCountOutcome.UNCAPPED,
                        ageThree, 0.0),
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageThree, true, 0.0, 3.0, 4.0, 5.0));
        assertEquals(RtPathSpatialReuseReference.BranchDirectWeightOutcome.INVALID,
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageOne, true, 2.0, Double.NaN, 4.0, 5.0).weight());
        assertEquals(RtPathSpatialReuseReference.BranchDirectWeightOutcome.INVALID,
                RtPathSpatialReuseReference.branchDirectWeightAudit(
                        ageOne, true, Double.MAX_VALUE, 2.0, 8.0, 2.0).weight());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectWeightAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, 2.0, 3.0, 4.0, 5.0));
    }

    @Test
    void branchDirectSelectionAuditUsesTheUncappedOverflowStableRatio() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.WEIGHT_REJECT,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE,
                        ageOne, 0.0),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageOne, false, 3.0, 2.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.CURRENT_REJECT,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE,
                        ageTwo, 0.0),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageTwo, true, Double.NaN, 2.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.ZERO,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.ZERO,
                        ageOne, 0.0),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageOne, true, 0.0, 0.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.ONE,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.ZERO,
                        ageTwo, 1.0),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageTwo, true, 0.0, 2.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.OPEN,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.POSITIVE,
                        ageThree, 0.4),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageThree, true, 3.0, 2.0));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.OPEN,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.POSITIVE,
                        ageOne, 0.5),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageOne, true, Double.MAX_VALUE, Double.MAX_VALUE));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectSelectionOutcome.ONE,
                        RtPathSpatialReuseReference.BranchDirectCurrentWeightOutcome.POSITIVE,
                        ageTwo, 1.0),
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageTwo, true, Double.MIN_VALUE, Double.MAX_VALUE));
        assertEquals(RtPathSpatialReuseReference.BranchDirectSelectionOutcome.INVALID,
                RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        ageOne, true, 2.0, Double.POSITIVE_INFINITY).selection());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectSelectionAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, 3.0, 2.0));
    }

    @Test
    void branchDirectBernoulliAuditUsesIndependentDeterministicHashDraws() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        assertEquals(new RtPathSpatialReuseReference.BranchDirectBernoulliAudit(
                        RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.SELECTION_REJECT,
                        RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.NOT_ELIGIBLE,
                        ageOne, 0.0, false, false),
                RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                        ageOne, false, 0.5, 123, 7, 99));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectBernoulliAudit(
                        RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.INVALID,
                        RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.NOT_ELIGIBLE,
                        ageTwo, 0.0, false, false),
                RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                        ageTwo, true, Double.NaN, 123, 7, 99));

        var retainedOpen = RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                ageOne, true, 0.5, 123, 7, 99);
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.RETAINED,
                retainedOpen.outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.OPEN,
                retainedOpen.boundary());
        assertEquals(0.6640617847442627, retainedOpen.random());
        assertFalse(retainedOpen.zeroViolation());
        assertFalse(retainedOpen.oneViolation());

        var selectedOpen = RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                ageTwo, true, 0.5, 123, 8, 99);
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.SELECTED,
                selectedOpen.outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.OPEN,
                selectedOpen.boundary());
        assertEquals(0.24819517135620117, selectedOpen.random());

        var retainedZero = RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                ageThree, true, 0.0, 0, 0, 0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.RETAINED,
                retainedZero.outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.ZERO,
                retainedZero.boundary());
        assertFalse(retainedZero.zeroViolation());
        var selectedOne = RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                ageThree, true, 1.0, 0, 0, 0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliOutcome.SELECTED,
                selectedOne.outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectBernoulliBoundary.ONE,
                selectedOne.boundary());
        assertFalse(selectedOne.oneViolation());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectBernoulliAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, 0.5, 123, 7, 99));
    }

    @Test
    void branchDirectPostSelectionAuditKeepsCapsAndFinalWeightsRegisterOnly() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        var uncapped = RtPathSpatialReuseReference.BranchDirectStoredCapOutcome.UNCAPPED;
        var notEligible = RtPathSpatialReuseReference.BranchDirectStoredCapOutcome.NOT_ELIGIBLE;

        assertEquals(new RtPathSpatialReuseReference.BranchDirectPostSelectionAudit(
                        RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.BERNOULLI_REJECT,
                        notEligible, notEligible, ageOne,
                        0.0, 0.0, 0.0, 0.0, true, false),
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageOne, false, true, 10.0, 2.0, 5.0, 6.0, 4.0, 3.0));

        var selected = RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                ageOne, true, true, 10.0, 2.0, 5.0, 6.0, 4.0, 3.0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.SELECTED_READY,
                selected.outcome());
        assertEquals(uncapped, selected.weightCap());
        assertEquals(uncapped, selected.countCap());
        assertEquals(16.0, selected.nextWeightSum(), 0.0);
        assertEquals(6.0, selected.nextEffectiveCount(), 0.0);
        assertEquals(3.0, selected.selectedTarget(), 0.0);
        assertEquals(16.0 / 18.0, selected.finalWeight(), 1.0e-12);
        assertTrue(selected.sourceSelected());
        assertFalse(selected.empty());

        var retained = RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                ageTwo, true, false, 10.0, 2.0, 5.0, 6.0, 4.0, 3.0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.RETAINED_READY,
                retained.outcome());
        assertEquals(5.0, retained.selectedTarget(), 0.0);
        assertEquals(16.0 / 30.0, retained.finalWeight(), 1.0e-12);
        assertFalse(retained.sourceSelected());

        var empty = RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                ageThree, true, false, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.RETAINED_READY,
                empty.outcome());
        assertEquals(0.0, empty.nextWeightSum(), 0.0);
        assertEquals(4.0, empty.nextEffectiveCount(), 0.0);
        assertEquals(0.0, empty.finalWeight(), 0.0);
        assertTrue(empty.empty());

        var capped = RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                ageOne, true, true, 1.0e30, 16_777_216.0, 2.0,
                Double.MAX_VALUE, 16.0, 2.0);
        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.SELECTED_READY,
                capped.outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectStoredCapOutcome.CAPPED,
                capped.weightCap());
        assertEquals(RtPathSpatialReuseReference.BranchDirectStoredCapOutcome.CAPPED,
                capped.countCap());
        assertEquals(1.0e30, capped.nextWeightSum(), 0.0);
        assertEquals(16_777_216.0, capped.nextEffectiveCount(), 0.0);
        assertEquals(1.0e30 / (16_777_216.0 * 2.0), capped.finalWeight(), 1.0e-12);

        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.CURRENT_REJECT,
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageOne, true, false, Double.NaN, 2.0, 5.0,
                        6.0, 4.0, 3.0).outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.NEXT_REJECT,
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageOne, true, true, 10.0, 2.0, 5.0,
                        Double.POSITIVE_INFINITY, 4.0, 3.0).outcome());
        assertEquals(
                RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.SELECTED_TARGET_REJECT,
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageOne, true, true, 1.0, 1.0, 1.0,
                        1.0, 1.0, 0.0).outcome());
        assertEquals(
                RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.RETAINED_TARGET_REJECT,
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageTwo, true, false, 1.0, 1.0, 0.0,
                        1.0, 1.0, 1.0).outcome());
        assertEquals(
                RtPathSpatialReuseReference.BranchDirectPostSelectionOutcome.SELECTED_FINAL_REJECT,
                RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        ageThree, true, true, 1.0, 16_777_216.0, 1.0,
                        1.0, 0.0, Double.MAX_VALUE).outcome());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectPostSelectionAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, false, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
    }

    @Test
    void branchDirectRecordAuditSeparatesRewritePreserveAndWeightLanes() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        var ready = RtPathSpatialReuseReference.BranchDirectLaneOutcome.READY;
        var reject = RtPathSpatialReuseReference.BranchDirectLaneOutcome.REJECT;
        var notEligible = RtPathSpatialReuseReference.BranchDirectLaneOutcome.NOT_ELIGIBLE;

        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.POST_SELECTION_REJECT,
                        notEligible, notEligible, notEligible, notEligible, ageOne, true),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageOne, false, true, false,
                        true, true, true, true, true));

        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_READY,
                        ready, ready, notEligible, ready, ageOne, true),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageOne, true, true, false,
                        true, true, false, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_REJECT,
                        reject, ready, notEligible, ready, ageTwo, true),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageTwo, true, true, false,
                        false, true, false, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_REJECT,
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageTwo, true, true, false,
                        true, true, false, true, false).outcome());

        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.RETAINED_READY,
                        notEligible, notEligible, ready, ready, ageThree, false),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageThree, true, false, false,
                        false, false, true, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.RETAINED_REJECT,
                        notEligible, notEligible, reject, ready, ageThree, false),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageThree, true, false, false,
                        false, false, false, true, true));

        assertEquals(new RtPathSpatialReuseReference.BranchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.EMPTY_READY,
                        notEligible, notEligible, notEligible, notEligible, ageOne, false),
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageOne, true, false, true,
                        false, false, false, false, false));
        assertEquals(RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_REJECT,
                RtPathSpatialReuseReference.branchDirectRecordAudit(
                        ageOne, true, true, true,
                        false, false, false, false, false).outcome());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectRecordAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        true, false, false,
                        false, false, true, true, true));
    }

    @Test
    void branchDirectPairAuditOrdersSourceKeyAndRootProvenance() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;
        var ready = RtPathSpatialReuseReference.BranchDirectLaneOutcome.READY;
        var reject = RtPathSpatialReuseReference.BranchDirectLaneOutcome.REJECT;
        var notEligible = RtPathSpatialReuseReference.BranchDirectLaneOutcome.NOT_ELIGIBLE;

        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.RECORD_REJECT,
                        notEligible, notEligible, notEligible, ageOne),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageOne,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_REJECT,
                        true, true, true, true, true));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.EMPTY_READY,
                        notEligible, notEligible, notEligible, ageTwo),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageTwo,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.EMPTY_READY,
                        false, false, false, false, false));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome
                                .SELECTED_SOURCE_REJECT,
                        reject, notEligible, notEligible, ageOne),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageOne,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_READY,
                        false, false, false, false, false));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_KEY_REJECT,
                        ready, reject, notEligible, ageTwo),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageTwo,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_READY,
                        true, false, false, false, false));
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_ROOT_REJECT,
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageThree,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_READY,
                        true, true, true, false, true).outcome());
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        ready, ready, ready, ageThree),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageThree,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.SELECTED_READY,
                        true, true, true, true, true));
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairOutcome.RETAINED_ROOT_REJECT,
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageOne,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.RETAINED_READY,
                        false, false, true, true, false).outcome());
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.RETAINED_READY,
                        notEligible, notEligible, ready, ageTwo),
                RtPathSpatialReuseReference.branchDirectPairAudit(
                        ageTwo,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.RETAINED_READY,
                        false, false, true, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectPairAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        RtPathSpatialReuseReference.BranchDirectRecordOutcome.EMPTY_READY,
                        false, false, false, false, false));
    }

    @Test
    void branchDirectPairReplayUsesBranchSpecificComparators() {
        var ageOne = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_ONE;
        var ageTwo = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_TWO;
        var ageThree = RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.AGE_THREE;

        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairReplayAudit(
                        RtPathSpatialReuseReference.BranchDirectPairReplayOutcome.NOT_ELIGIBLE,
                        RtPathSpatialReuseReference.BranchDirectPairReplayComparator.NONE,
                        ageOne, 0),
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageOne, RtPathSpatialReuseReference.BranchDirectPairOutcome.EMPTY_READY,
                        null, 0, false, -1));
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairReplayAudit(
                        RtPathSpatialReuseReference.BranchDirectPairReplayOutcome
                                .SELECTED_ACCEPTED,
                        RtPathSpatialReuseReference.BranchDirectPairReplayComparator
                                .MAPPING_SOURCE,
                        ageTwo, 2),
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageTwo, RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                        2, true, -1));
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairReplayOutcome.SELECTED_REJECT,
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageTwo, RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                        1, false, 0).outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairReplayOutcome.SELECTED_REJECT,
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageTwo, RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        RtPathSpatialReuseReference.MappingKind.IDENTITY,
                        1, true, 0).outcome());
        assertEquals(new RtPathSpatialReuseReference.BranchDirectPairReplayAudit(
                        RtPathSpatialReuseReference.BranchDirectPairReplayOutcome
                                .RETAINED_ACCEPTED,
                        RtPathSpatialReuseReference.BranchDirectPairReplayComparator.EXACT,
                        ageThree, 1),
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageThree, RtPathSpatialReuseReference.BranchDirectPairOutcome.RETAINED_READY,
                        RtPathSpatialReuseReference.MappingKind.IDENTITY,
                        1, false, 0));
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairReplayOutcome.RETAINED_REJECT,
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageThree, RtPathSpatialReuseReference.BranchDirectPairOutcome.RETAINED_READY,
                        RtPathSpatialReuseReference.MappingKind.IDENTITY,
                        2, true, 4).outcome());
        assertEquals(RtPathSpatialReuseReference.BranchDirectPairReplayOutcome.RETAINED_REJECT,
                RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageThree, RtPathSpatialReuseReference.BranchDirectPairOutcome.RETAINED_READY,
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                        2, true, 0).outcome());
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        RtPathSpatialReuseReference.BranchCandidateRetentionOutcome.CURRENT,
                        RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                        1, true, 0));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathSpatialReuseReference.branchDirectPairReplayAudit(
                        ageOne, RtPathSpatialReuseReference.BranchDirectPairOutcome.SELECTED_READY,
                        RtPathSpatialReuseReference.MappingKind.DIFFUSE_RECONNECTION,
                        3, true, 0));
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
        assertEquals(128, RtPathReservoirHistory.GUIDE_SCRATCH_VALIDATE_PASS_FLAG);
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
        assertEquals(60, RtPathReservoirHistory.CROSS_FRAME_EDGE_BREAKDOWN_ELIGIBLE_INDEX);
        assertEquals(61, RtPathReservoirHistory.CROSS_FRAME_EDGE_MISSING_VALID_INDEX);
        assertEquals(62, RtPathReservoirHistory.CROSS_FRAME_EDGE_DEPTH_REJECT_INDEX);
        assertEquals(63, RtPathReservoirHistory.CROSS_FRAME_EDGE_EVENT_REJECT_INDEX);
        assertEquals(64, RtPathReservoirHistory.CROSS_FRAME_EDGE_MAPPING_REJECT_INDEX);
        assertEquals(65, RtPathReservoirHistory.CROSS_FRAME_EDGE_PDF_REJECT_INDEX);
        assertEquals(66, RtPathReservoirHistory.CROSS_FRAME_EDGE_FINITE_REJECT_INDEX);
        assertEquals(67, RtPathReservoirHistory.RECEIVER_GUIDE_ATTEMPTED_INDEX);
        assertEquals(68, RtPathReservoirHistory.RECEIVER_GUIDE_INVALID_INDEX);
        assertEquals(69, RtPathReservoirHistory.RECEIVER_GUIDE_NO_STORED_EDGE_INDEX);
        assertEquals(70, RtPathReservoirHistory.RECEIVER_GUIDE_STORED_ELIGIBLE_INDEX);
        assertEquals(71, RtPathReservoirHistory.RECEIVER_GUIDE_ACCEPTED_INDEX);
        assertEquals(72, RtPathReservoirHistory.RECEIVER_GUIDE_MASS_MISMATCH_INDEX);
        assertEquals(73, RtPathReservoirHistory.RECEIVER_GUIDE_THROUGHPUT_MISMATCH_INDEX);
        assertEquals(74, RtPathReservoirHistory.RECEIVER_GUIDE_PDF_MISMATCH_INDEX);
        assertEquals(75, RtPathReservoirHistory.GUIDE_REMAP_ELIGIBLE_INDEX);
        assertEquals(76, RtPathReservoirHistory.GUIDE_REMAP_READY_INDEX);
        assertEquals(77, RtPathReservoirHistory.GUIDE_REMAP_GEOMETRY_REJECT_INDEX);
        assertEquals(78, RtPathReservoirHistory.GUIDE_REMAP_PDF_REJECT_INDEX);
        assertEquals(79, RtPathReservoirHistory.GUIDE_REMAP_THROUGHPUT_REJECT_INDEX);
        assertEquals(80, RtPathReservoirHistory.GUIDE_VISIBILITY_ELIGIBLE_INDEX);
        assertEquals(81, RtPathReservoirHistory.GUIDE_VISIBILITY_CLEAR_INDEX);
        assertEquals(82, RtPathReservoirHistory.GUIDE_VISIBILITY_TINTED_INDEX);
        assertEquals(83, RtPathReservoirHistory.GUIDE_VISIBILITY_OCCLUDED_INDEX);
        assertEquals(84, RtPathReservoirHistory.GUIDE_VISIBILITY_INVALID_INDEX);
        assertEquals(85, RtPathReservoirHistory.GUIDE_TARGET_ELIGIBLE_INDEX);
        assertEquals(86, RtPathReservoirHistory.GUIDE_TARGET_POSITIVE_INDEX);
        assertEquals(87, RtPathReservoirHistory.GUIDE_TARGET_ZERO_INDEX);
        assertEquals(88, RtPathReservoirHistory.GUIDE_TARGET_INVALID_INDEX);
        assertEquals(89, RtPathReservoirHistory.GUIDE_WEIGHT_ELIGIBLE_INDEX);
        assertEquals(90, RtPathReservoirHistory.GUIDE_WEIGHT_POSITIVE_INDEX);
        assertEquals(91, RtPathReservoirHistory.GUIDE_WEIGHT_ZERO_INDEX);
        assertEquals(92, RtPathReservoirHistory.GUIDE_WEIGHT_INVALID_INDEX);
        assertEquals(93, RtPathReservoirHistory.GUIDE_SELECTION_ELIGIBLE_INDEX);
        assertEquals(94, RtPathReservoirHistory.GUIDE_SELECTION_POSITIVE_INDEX);
        assertEquals(95, RtPathReservoirHistory.GUIDE_SELECTION_ZERO_INDEX);
        assertEquals(96, RtPathReservoirHistory.GUIDE_SELECTION_CURRENT_REJECT_INDEX);
        assertEquals(97, RtPathReservoirHistory.GUIDE_SELECTION_PROBABILITY_HIGH_REJECT_INDEX);
        assertEquals(98, RtPathReservoirHistory.GUIDE_SELECTION_ARITHMETIC_REJECT_INDEX);
        assertEquals(99, RtPathReservoirHistory.GUIDE_STABLE_SELECTION_ELIGIBLE_INDEX);
        assertEquals(100, RtPathReservoirHistory.GUIDE_STABLE_SELECTION_POSITIVE_INDEX);
        assertEquals(101, RtPathReservoirHistory.GUIDE_STABLE_SELECTION_ZERO_INDEX);
        assertEquals(102, RtPathReservoirHistory.GUIDE_STABLE_SELECTION_INVALID_INDEX);
        assertEquals(103, RtPathReservoirHistory.GUIDE_STABLE_BERNOULLI_ELIGIBLE_INDEX);
        assertEquals(104, RtPathReservoirHistory.GUIDE_STABLE_BERNOULLI_SELECTED_INDEX);
        assertEquals(105, RtPathReservoirHistory.GUIDE_STABLE_BERNOULLI_RETAINED_INDEX);
        assertEquals(106, RtPathReservoirHistory.GUIDE_STABLE_BERNOULLI_INVALID_INDEX);
        assertEquals(107, RtPathReservoirHistory.GUIDE_POST_SELECTION_ELIGIBLE_INDEX);
        assertEquals(108, RtPathReservoirHistory.GUIDE_POST_SELECTION_SELECTED_READY_INDEX);
        assertEquals(109, RtPathReservoirHistory.GUIDE_POST_SELECTION_RETAINED_READY_INDEX);
        assertEquals(110, RtPathReservoirHistory.GUIDE_POST_SELECTION_EMPTY_INDEX);
        assertEquals(111, RtPathReservoirHistory.GUIDE_POST_SELECTION_SAMPLE_REJECT_INDEX);
        assertEquals(112, RtPathReservoirHistory.GUIDE_POST_SELECTION_ARITHMETIC_REJECT_INDEX);
        assertEquals(113, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_ELIGIBLE_INDEX);
        assertEquals(114, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_SELECTED_INDEX);
        assertEquals(115, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_RETAINED_INDEX);
        assertEquals(116, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_EMPTY_INDEX);
        assertEquals(117, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_METADATA_REJECT_INDEX);
        assertEquals(118, RtPathReservoirHistory.GUIDE_SAMPLE_COPY_ARITHMETIC_REJECT_INDEX);
        assertEquals(119, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_ELIGIBLE_INDEX);
        assertEquals(120, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_SELECTED_INDEX);
        assertEquals(121, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_RETAINED_INDEX);
        assertEquals(122, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_EMPTY_INDEX);
        assertEquals(123, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_METADATA_REJECT_INDEX);
        assertEquals(124, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_ARITHMETIC_REJECT_INDEX);
        assertEquals(125, RtPathReservoirHistory.GUIDE_SCRATCH_STORAGE_INVALID_INDEX);
        assertEquals(126, RtPathReservoirHistory.GUIDE_ROOT_STORAGE_ELIGIBLE_INDEX);
        assertEquals(127, RtPathReservoirHistory.GUIDE_ROOT_STORAGE_SELECTED_READY_INDEX);
        assertEquals(128, RtPathReservoirHistory.GUIDE_ROOT_STORAGE_RETAINED_READY_INDEX);
        assertEquals(129, RtPathReservoirHistory.GUIDE_ROOT_STORAGE_MISSING_INDEX);
        assertEquals(130, RtPathReservoirHistory.GUIDE_ROOT_STORAGE_METADATA_REJECT_INDEX);
        assertEquals(131, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_ATTEMPTED_INDEX);
        assertEquals(132, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_LIFECYCLE_REJECT_INDEX);
        assertEquals(133,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_RECEIVER_REPROJECTION_REJECT_INDEX);
        assertEquals(134, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_EMPTY_INDEX);
        assertEquals(135, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_METADATA_REJECT_INDEX);
        assertEquals(136,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_RECEIVER_SURFACE_REJECT_INDEX);
        assertEquals(137,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_SOURCE_REPROJECTION_REJECT_INDEX);
        assertEquals(138,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_SOURCE_SURFACE_REJECT_INDEX);
        assertEquals(139,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_SOURCE_REPLAY_REJECT_INDEX);
        assertEquals(140,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_SELECTED_ACCEPTED_INDEX);
        assertEquals(141,
                RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_RETAINED_ACCEPTED_INDEX);
        assertEquals(142, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_ELIGIBLE_INDEX);
        assertEquals(143, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_RECEIVER_GUIDE_REJECT_INDEX);
        assertEquals(144, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_READY_INDEX);
        assertEquals(145, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_GEOMETRY_REJECT_INDEX);
        assertEquals(146, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_PDF_REJECT_INDEX);
        assertEquals(147, RtPathReservoirHistory.GUIDE_PREVIOUS_REMAP_THROUGHPUT_REJECT_INDEX);
        assertEquals(148, RtPathReservoirHistory.GUIDE_PREVIOUS_VISIBILITY_ELIGIBLE_INDEX);
        assertEquals(149, RtPathReservoirHistory.GUIDE_PREVIOUS_VISIBILITY_CLEAR_INDEX);
        assertEquals(150, RtPathReservoirHistory.GUIDE_PREVIOUS_VISIBILITY_TINTED_INDEX);
        assertEquals(151, RtPathReservoirHistory.GUIDE_PREVIOUS_VISIBILITY_OCCLUDED_INDEX);
        assertEquals(152, RtPathReservoirHistory.GUIDE_PREVIOUS_VISIBILITY_INVALID_INDEX);
        assertEquals(153, RtPathReservoirHistory.GUIDE_PREVIOUS_TARGET_ELIGIBLE_INDEX);
        assertEquals(154, RtPathReservoirHistory.GUIDE_PREVIOUS_TARGET_POSITIVE_INDEX);
        assertEquals(155, RtPathReservoirHistory.GUIDE_PREVIOUS_TARGET_ZERO_INDEX);
        assertEquals(156, RtPathReservoirHistory.GUIDE_PREVIOUS_TARGET_INVALID_INDEX);
        assertEquals(157,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_METADATA_ELIGIBLE_INDEX);
        assertEquals(158, RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_TARGET_MATCH_INDEX);
        assertEquals(159,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_TARGET_MISMATCH_INDEX);
        assertEquals(160, RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_TARGET_INVALID_INDEX);
        assertEquals(161,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_THROUGHPUT_MATCH_INDEX);
        assertEquals(162,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_THROUGHPUT_MISMATCH_INDEX);
        assertEquals(163,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STORED_THROUGHPUT_INVALID_INDEX);
        assertEquals(164, RtPathReservoirHistory.GUIDE_PREVIOUS_WEIGHT_ELIGIBLE_INDEX);
        assertEquals(165, RtPathReservoirHistory.GUIDE_PREVIOUS_WEIGHT_ZERO_INDEX);
        assertEquals(166, RtPathReservoirHistory.GUIDE_PREVIOUS_WEIGHT_POSITIVE_INDEX);
        assertEquals(167, RtPathReservoirHistory.GUIDE_PREVIOUS_WEIGHT_INVALID_INDEX);
        assertEquals(168,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_ELIGIBLE_INDEX);
        assertEquals(169,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_NEXT_VALID_INDEX);
        assertEquals(170,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_NEXT_INVALID_INDEX);
        assertEquals(171,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_VALID_INDEX);
        assertEquals(172,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_CURRENT_INVALID_INDEX);
        assertEquals(173,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_VALID_INDEX);
        assertEquals(174,
                RtPathReservoirHistory.GUIDE_PREVIOUS_POST_WEIGHT_SOURCE_INVALID_INDEX);
        assertEquals(175,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_PROBABILITY_ELIGIBLE_INDEX);
        assertEquals(176,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_DENOMINATOR_VALID_INDEX);
        assertEquals(177,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_DENOMINATOR_INVALID_INDEX);
        assertEquals(178,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_PROBABILITY_VALID_INDEX);
        assertEquals(179,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_PROBABILITY_INVALID_INDEX);
        assertEquals(180,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_VALID_INDEX);
        assertEquals(181,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_CURRENT_DENOMINATOR_INVALID_INDEX);
        assertEquals(182,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_VALID_INDEX);
        assertEquals(183,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_SOURCE_DENOMINATOR_INVALID_INDEX);
        assertEquals(184,
                RtPathReservoirHistory.GUIDE_PREVIOUS_SELECTION_PROBABILITY_HIGH_REJECT_INDEX);
        assertEquals(185,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_SELECTION_ELIGIBLE_INDEX);
        assertEquals(186,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_SELECTION_POSITIVE_INDEX);
        assertEquals(187,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_SELECTION_ZERO_INDEX);
        assertEquals(188,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_SELECTION_INVALID_INDEX);
        assertEquals(189,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_BERNOULLI_ELIGIBLE_INDEX);
        assertEquals(190,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_BERNOULLI_SELECTED_INDEX);
        assertEquals(191,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_BERNOULLI_RETAINED_INDEX);
        assertEquals(192,
                RtPathReservoirHistory.GUIDE_PREVIOUS_STABLE_BERNOULLI_INVALID_INDEX);
        assertEquals(193, RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_ELIGIBLE_INDEX);
        assertEquals(194,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_VALID_INDEX);
        assertEquals(195,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_SELECTED_FINAL_INVALID_INDEX);
        assertEquals(196,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_VALID_INDEX);
        assertEquals(197,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_RETAINED_FINAL_INVALID_INDEX);
        assertEquals(198,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_VALID_INDEX);
        assertEquals(199,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_SELECTED_PAYLOAD_INVALID_INDEX);
        assertEquals(200,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_VALID_INDEX);
        assertEquals(201,
                RtPathReservoirHistory.GUIDE_PREVIOUS_BRANCH_RETAINED_PAYLOAD_INVALID_INDEX);
        assertEquals(202, RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_ELIGIBLE_INDEX);
        assertEquals(203,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_SELECTED_REWRITE_READY_INDEX);
        assertEquals(204,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_SELECTED_REWRITE_REJECT_INDEX);
        assertEquals(205,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_SELECTED_PRESERVE_READY_INDEX);
        assertEquals(206,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_SELECTED_PRESERVE_REJECT_INDEX);
        assertEquals(207,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_RETAINED_PRESERVE_READY_INDEX);
        assertEquals(208,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_RETAINED_PRESERVE_REJECT_INDEX);
        assertEquals(209,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_WEIGHTS_READY_INDEX);
        assertEquals(210,
                RtPathReservoirHistory.GUIDE_PREVIOUS_LANE_WEIGHTS_REJECT_INDEX);
        assertEquals(211, RtPathReservoirHistory.GUIDE_PREVIOUS_RECORD_ELIGIBLE_INDEX);
        assertEquals(212,
                RtPathReservoirHistory.GUIDE_PREVIOUS_RECORD_SELECTED_READY_INDEX);
        assertEquals(213,
                RtPathReservoirHistory.GUIDE_PREVIOUS_RECORD_SELECTED_REJECT_INDEX);
        assertEquals(214,
                RtPathReservoirHistory.GUIDE_PREVIOUS_RECORD_RETAINED_READY_INDEX);
        assertEquals(215,
                RtPathReservoirHistory.GUIDE_PREVIOUS_RECORD_RETAINED_REJECT_INDEX);
        assertEquals(216, RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_ELIGIBLE_INDEX);
        assertEquals(217,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_SELECTED_READY_INDEX);
        assertEquals(218,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_SELECTED_REJECT_INDEX);
        assertEquals(219,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_RETAINED_READY_INDEX);
        assertEquals(220,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_RETAINED_REJECT_INDEX);
        assertEquals(221,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_ROOT_CAPTURE_REJECT_INDEX);
        assertEquals(222,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_ROOT_CHAIN_REJECT_INDEX);
        assertEquals(223,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_ROOT_IDENTITY_REJECT_INDEX);
        assertEquals(224,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_REPLAY_ELIGIBLE_INDEX);
        assertEquals(225,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_REPLAY_SELECTED_ACCEPTED_INDEX);
        assertEquals(226,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_REPLAY_SELECTED_REJECT_INDEX);
        assertEquals(227,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_REPLAY_RETAINED_ACCEPTED_INDEX);
        assertEquals(228,
                RtPathReservoirHistory.GUIDE_PREVIOUS_PAIR_REPLAY_RETAINED_REJECT_INDEX);
        assertEquals(229,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_ATTEMPTED_INDEX);
        assertEquals(230,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_EMPTY_INDEX);
        assertEquals(231,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_METADATA_REJECT_INDEX);
        assertEquals(232,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_RECEIVER_REPROJECTION_REJECT_INDEX);
        assertEquals(233,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_RECEIVER_SURFACE_REJECT_INDEX);
        assertEquals(234,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_REPROJECTION_REJECT_INDEX);
        assertEquals(235,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_SURFACE_REJECT_INDEX);
        assertEquals(236,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_SOURCE_REPLAY_REJECT_INDEX);
        assertEquals(237,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_SELECTED_ACCEPTED_INDEX);
        assertEquals(238,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_RETAINED_ACCEPTED_INDEX);
        assertEquals(239, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_REPLAY_DELTA_INDEX);
        assertEquals(240, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_WRITE_ELIGIBLE_INDEX);
        assertEquals(241, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_WRITE_COMPLETED_INDEX);
        assertEquals(242, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_CAPTURE_CURSOR_INDEX);
        assertEquals(243, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_ATTEMPTED_INDEX);
        assertEquals(244,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_METADATA_REJECT_INDEX);
        assertEquals(245,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_RESERVOIR_MATCH_INDEX);
        assertEquals(246,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_RESERVOIR_MISMATCH_INDEX);
        assertEquals(247,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_ROOT_MATCH_INDEX);
        assertEquals(248,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_ROOT_MISMATCH_INDEX);
        assertEquals(249,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_SELECTED_ACCEPTED_INDEX);
        assertEquals(250,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_RETAINED_ACCEPTED_INDEX);
        assertEquals(251,
                RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_PAIR_REJECT_INDEX);
        assertEquals(252, RtPathReservoirHistory.GUIDE_BRANCH_SCRATCH_VALIDATE_DELTA_INDEX);
        assertEquals(253,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_WRITE_ELIGIBLE_INDEX);
        assertEquals(254,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_SELECTED_WRITTEN_INDEX);
        assertEquals(255,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_RETAINED_WRITTEN_INDEX);
        assertEquals(256,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_EMPTY_INDEX);
        assertEquals(257,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_METADATA_REJECT_INDEX);
        assertEquals(258,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_SELECTED_ADMITTED_INDEX);
        assertEquals(259,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_RETAINED_ADMITTED_INDEX);
        assertEquals(260,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_TAG_REPLAY_DELTA_INDEX);
        assertEquals(261,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_ATTEMPTED_INDEX);
        assertEquals(262,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_EMPTY_INDEX);
        assertEquals(263,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_FUTURE_REJECT_INDEX);
        assertEquals(264,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_GENERATION_REJECT_INDEX);
        assertEquals(265,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_MAPPING_REJECT_INDEX);
        assertEquals(266,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_CURRENT_INDEX);
        assertEquals(267,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_AGE_ONE_INDEX);
        assertEquals(268,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_AGE_TWO_INDEX);
        assertEquals(269,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_AGE_THREE_INDEX);
        assertEquals(270,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_EXPIRED_INDEX);
        assertEquals(271,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_IDENTITY_LIVE_INDEX);
        assertEquals(272,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_MAPPED_LIVE_INDEX);
        assertEquals(273,
                RtPathReservoirHistory.GUIDE_BRANCH_CANDIDATE_RETENTION_DELTA_INDEX);
        assertEquals(274, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_ATTEMPTED_INDEX);
        assertEquals(275, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_EMPTY_INDEX);
        assertEquals(276,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_METADATA_REJECT_INDEX);
        assertEquals(277,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_CURRENT_SKIP_INDEX);
        assertEquals(278, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_EXPIRED_INDEX);
        assertEquals(279,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_PROVENANCE_REJECT_INDEX);
        assertEquals(280,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_ONE_ELIGIBLE_INDEX);
        assertEquals(281,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_TWO_ELIGIBLE_INDEX);
        assertEquals(282,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_THREE_ELIGIBLE_INDEX);
        assertEquals(283,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_IDENTITY_ACCEPTED_INDEX);
        assertEquals(284,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_IDENTITY_REJECT_INDEX);
        assertEquals(285,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_MAPPED_ROOT_ACCEPTED_INDEX);
        assertEquals(286,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_MAPPED_ROOT_REJECT_INDEX);
        assertEquals(287, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_DELTA_INDEX);
        assertEquals(288,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_ONE_ACCEPTED_INDEX);
        assertEquals(289,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_TWO_ACCEPTED_INDEX);
        assertEquals(290,
                RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_THREE_ACCEPTED_INDEX);
        assertEquals(291, RtPathReservoirHistory.GUIDE_BRANCH_AGE_REPLAY_AGE_DELTA_INDEX);
        assertEquals(292, RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_ELIGIBLE_INDEX);
        assertEquals(293, RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_CLIP_REJECT_INDEX);
        assertEquals(294, RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_BOUNDS_REJECT_INDEX);
        assertEquals(295, RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_SURFACE_REJECT_INDEX);
        assertEquals(296,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_AGE_ONE_ADMITTED_INDEX);
        assertEquals(297,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_AGE_TWO_ADMITTED_INDEX);
        assertEquals(298,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_AGE_THREE_ADMITTED_INDEX);
        assertEquals(299,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_IDENTITY_ADMITTED_INDEX);
        assertEquals(300,
                RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_MAPPED_ADMITTED_INDEX);
        assertEquals(301, RtPathReservoirHistory.GUIDE_BRANCH_RECEIVER_DELTA_INDEX);
        assertEquals(302, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_ELIGIBLE_INDEX);
        assertEquals(303,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_GUIDE_REJECT_INDEX);
        assertEquals(304,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_EDGE_REJECT_INDEX);
        assertEquals(305,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_GEOMETRY_REJECT_INDEX);
        assertEquals(306,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_PDF_REJECT_INDEX);
        assertEquals(307,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_THROUGHPUT_REJECT_INDEX);
        assertEquals(308,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_AGE_ONE_READY_INDEX);
        assertEquals(309,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_AGE_TWO_READY_INDEX);
        assertEquals(310,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_AGE_THREE_READY_INDEX);
        assertEquals(311,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_IDENTITY_READY_INDEX);
        assertEquals(312,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_MAPPED_READY_INDEX);
        assertEquals(313, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_REMAP_DELTA_INDEX);
        assertEquals(314,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_ELIGIBLE_INDEX);
        assertEquals(315,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_CLEAR_INDEX);
        assertEquals(316,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_TINTED_INDEX);
        assertEquals(317,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_OCCLUDED_INDEX);
        assertEquals(318,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_INVALID_INDEX);
        assertEquals(319, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_ELIGIBLE_INDEX);
        assertEquals(320, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_POSITIVE_INDEX);
        assertEquals(321, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_ZERO_INDEX);
        assertEquals(322, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_INVALID_INDEX);
        assertEquals(323,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_AGE_ONE_READY_INDEX);
        assertEquals(324,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_AGE_TWO_READY_INDEX);
        assertEquals(325,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_AGE_THREE_READY_INDEX);
        assertEquals(326,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_IDENTITY_READY_INDEX);
        assertEquals(327,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_MAPPED_READY_INDEX);
        assertEquals(328,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_VISIBILITY_DELTA_INDEX);
        assertEquals(329, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_TARGET_DELTA_INDEX);
        assertEquals(330, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_ELIGIBLE_INDEX);
        assertEquals(331, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_POSITIVE_INDEX);
        assertEquals(332, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_ZERO_INDEX);
        assertEquals(333, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_INVALID_INDEX);
        assertEquals(334,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_AGE_ONE_READY_INDEX);
        assertEquals(335,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_AGE_TWO_READY_INDEX);
        assertEquals(336,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_AGE_THREE_READY_INDEX);
        assertEquals(337,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_IDENTITY_READY_INDEX);
        assertEquals(338,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_MAPPED_READY_INDEX);
        assertEquals(339,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_COUNT_UNCAPPED_INDEX);
        assertEquals(340,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_COUNT_CAPPED_INDEX);
        assertEquals(341, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_WEIGHT_DELTA_INDEX);
        assertEquals(342, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_ELIGIBLE_INDEX);
        assertEquals(343,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_CURRENT_REJECT_INDEX);
        assertEquals(344,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_PROBABILITY_ZERO_INDEX);
        assertEquals(345,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_PROBABILITY_OPEN_INDEX);
        assertEquals(346,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_PROBABILITY_ONE_INDEX);
        assertEquals(347,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_PROBABILITY_INVALID_INDEX);
        assertEquals(348,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_AGE_ONE_READY_INDEX);
        assertEquals(349,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_AGE_TWO_READY_INDEX);
        assertEquals(350,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_AGE_THREE_READY_INDEX);
        assertEquals(351,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_IDENTITY_READY_INDEX);
        assertEquals(352,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_MAPPED_READY_INDEX);
        assertEquals(353,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_CURRENT_ZERO_INDEX);
        assertEquals(354,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_CURRENT_POSITIVE_INDEX);
        assertEquals(355, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_SELECTION_DELTA_INDEX);
        assertEquals(356, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_ELIGIBLE_INDEX);
        assertEquals(357, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_SELECTED_INDEX);
        assertEquals(358, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_RETAINED_INDEX);
        assertEquals(359, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_INVALID_INDEX);
        assertEquals(360,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_PROBABILITY_ZERO_INDEX);
        assertEquals(361,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_PROBABILITY_OPEN_INDEX);
        assertEquals(362,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_PROBABILITY_ONE_INDEX);
        assertEquals(363,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_ZERO_VIOLATION_INDEX);
        assertEquals(364,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_ONE_VIOLATION_INDEX);
        assertEquals(365,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_AGE_ONE_READY_INDEX);
        assertEquals(366,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_AGE_TWO_READY_INDEX);
        assertEquals(367,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_AGE_THREE_READY_INDEX);
        assertEquals(368,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_IDENTITY_READY_INDEX);
        assertEquals(369,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_MAPPED_READY_INDEX);
        assertEquals(370, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_BERNOULLI_DELTA_INDEX);
        assertEquals(371,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_ELIGIBLE_INDEX);
        assertEquals(372,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_CURRENT_REJECT_INDEX);
        assertEquals(373,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_NEXT_INVALID_INDEX);
        assertEquals(374,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_SELECTED_READY_INDEX);
        assertEquals(375, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_POST_SELECTION_SELECTED_TARGET_REJECT_INDEX);
        assertEquals(376, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_POST_SELECTION_SELECTED_FINAL_REJECT_INDEX);
        assertEquals(377,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_RETAINED_READY_INDEX);
        assertEquals(378, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_POST_SELECTION_RETAINED_TARGET_REJECT_INDEX);
        assertEquals(379, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_POST_SELECTION_RETAINED_FINAL_REJECT_INDEX);
        assertEquals(380,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_EMPTY_READY_INDEX);
        assertEquals(381,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_WEIGHT_UNCAPPED_INDEX);
        assertEquals(382,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_WEIGHT_CAPPED_INDEX);
        assertEquals(383,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_COUNT_UNCAPPED_INDEX);
        assertEquals(384,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_COUNT_CAPPED_INDEX);
        assertEquals(385,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_AGE_ONE_READY_INDEX);
        assertEquals(386,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_AGE_TWO_READY_INDEX);
        assertEquals(387,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_AGE_THREE_READY_INDEX);
        assertEquals(388,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_IDENTITY_READY_INDEX);
        assertEquals(389,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_MAPPED_READY_INDEX);
        assertEquals(390,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_POST_SELECTION_DELTA_INDEX);
        assertEquals(391, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_ELIGIBLE_INDEX);
        assertEquals(392,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_SELECTED_READY_INDEX);
        assertEquals(393,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_SELECTED_REJECT_INDEX);
        assertEquals(394,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_RETAINED_READY_INDEX);
        assertEquals(395,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_RETAINED_REJECT_INDEX);
        assertEquals(396,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_EMPTY_READY_INDEX);
        assertEquals(397, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_SELECTED_REWRITE_READY_INDEX);
        assertEquals(398, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_SELECTED_REWRITE_REJECT_INDEX);
        assertEquals(399, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_SELECTED_PRESERVE_READY_INDEX);
        assertEquals(400, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_SELECTED_PRESERVE_REJECT_INDEX);
        assertEquals(401, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_RETAINED_PRESERVE_READY_INDEX);
        assertEquals(402, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_RETAINED_PRESERVE_REJECT_INDEX);
        assertEquals(403,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_WEIGHTS_READY_INDEX);
        assertEquals(404,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_WEIGHTS_REJECT_INDEX);
        assertEquals(405,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_AGE_ONE_READY_INDEX);
        assertEquals(406,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_AGE_TWO_READY_INDEX);
        assertEquals(407,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_AGE_THREE_READY_INDEX);
        assertEquals(408, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_RECORD_IDENTITY_SOURCE_READY_INDEX);
        assertEquals(409,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_MAPPED_SOURCE_READY_INDEX);
        assertEquals(410, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_RECORD_DELTA_INDEX);
        assertEquals(411, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_ELIGIBLE_INDEX);
        assertEquals(412,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_READY_INDEX);
        assertEquals(413,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_REJECT_INDEX);
        assertEquals(414,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_RETAINED_READY_INDEX);
        assertEquals(415,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_RETAINED_REJECT_INDEX);
        assertEquals(416,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_EMPTY_READY_INDEX);
        assertEquals(417,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SOURCE_READY_INDEX);
        assertEquals(418,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SOURCE_CLIP_REJECT_INDEX);
        assertEquals(419,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SOURCE_BOUNDS_REJECT_INDEX);
        assertEquals(420,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SOURCE_SURFACE_REJECT_INDEX);
        assertEquals(421,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_KEY_READY_INDEX);
        assertEquals(422,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_KEY_REJECT_INDEX);
        assertEquals(423,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_ROOT_READY_INDEX);
        assertEquals(424,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_SELECTED_ROOT_REJECT_INDEX);
        assertEquals(425,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_RETAINED_ROOT_READY_INDEX);
        assertEquals(426,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_RETAINED_ROOT_REJECT_INDEX);
        assertEquals(427,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_ROOT_CHAIN_REJECT_INDEX);
        assertEquals(428,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_ROOT_IDENTITY_REJECT_INDEX);
        assertEquals(429,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_AGE_ONE_READY_INDEX);
        assertEquals(430,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_AGE_TWO_READY_INDEX);
        assertEquals(431,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_AGE_THREE_READY_INDEX);
        assertEquals(432, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_IDENTITY_SOURCE_READY_INDEX);
        assertEquals(433,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_MAPPED_SOURCE_READY_INDEX);
        assertEquals(434, RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_DELTA_INDEX);
        assertEquals(435,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_REPLAY_ELIGIBLE_INDEX);
        assertEquals(436, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_SELECTED_ACCEPTED_INDEX);
        assertEquals(437,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_REPLAY_SELECTED_REJECT_INDEX);
        assertEquals(438, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_RETAINED_ACCEPTED_INDEX);
        assertEquals(439,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_REPLAY_RETAINED_REJECT_INDEX);
        assertEquals(440, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_AGE_ONE_ACCEPTED_INDEX);
        assertEquals(441, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_AGE_TWO_ACCEPTED_INDEX);
        assertEquals(442, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_AGE_THREE_ACCEPTED_INDEX);
        assertEquals(443, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_IDENTITY_SOURCE_ACCEPTED_INDEX);
        assertEquals(444, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_MAPPED_SOURCE_ACCEPTED_INDEX);
        assertEquals(445, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_ONE_SEGMENT_ACCEPTED_INDEX);
        assertEquals(446, RtPathReservoirHistory
                .GUIDE_BRANCH_DIRECT_PAIR_REPLAY_TWO_SEGMENT_ACCEPTED_INDEX);
        assertEquals(447,
                RtPathReservoirHistory.GUIDE_BRANCH_DIRECT_PAIR_REPLAY_DELTA_INDEX);
        assertEquals(448, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(448 * Integer.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_BYTES);
        assertEquals(32, RtPathReservoirHistory.SHIFTED_RECEIVER_GUIDE_STRIDE);
        assertEquals(27_566_080L,
                RtPathReservoirHistory.shiftedReceiverGuideBytes(1280, 673));
        assertEquals(160, PathSourceRootData.BYTE_SIZE);
        assertEquals(4096, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        assertEquals(4096, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_DENSITY_PAIR_OFFSET);
        assertEquals(8192, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_MERGE_PAIR_OFFSET);
        assertEquals(12288, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_SELECTION_PAIR_OFFSET);
        assertEquals(4096 * 8, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_PAIR_FLOAT_COUNT);
        assertEquals(4096, RtPathReservoirHistory.PATH_BRANCH_SCRATCH_CAPTURE_CAPACITY);
        assertEquals(4, RtPathReservoirHistory.PATH_BRANCH_CANDIDATE_HISTORY_SLOT_COUNT);
        assertEquals(384, RtPathReservoirHistory.PATH_BRANCH_SCRATCH_CAPTURE_STRIDE);
        assertEquals(4 * 4096 * 384,
                RtPathReservoirHistory.PATH_BRANCH_SCRATCH_CAPTURE_BYTES);
        assertEquals(4096 * 8 * Float.BYTES + 4 * 4096 * 384,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_BYTES);
    }
}
