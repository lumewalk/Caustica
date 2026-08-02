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
        assertEquals(164, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(164 * Integer.BYTES,
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
        assertEquals(4096 * 8 * Float.BYTES,
                RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_PAIR_BYTES);
    }
}
