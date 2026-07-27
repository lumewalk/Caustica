package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

final class RtReservoirReferenceTest {
    @Test
    void proposalScheduleMatchesShaderStratification() {
        assertEquals(2, RtReservoirReference.globalCandidateCount(8, true));
        int local = 0;
        for (int candidate = 0; candidate < 8; candidate++) {
            if (RtReservoirReference.usesLocalCandidate(candidate, 8, true)) {
                local++;
            }
        }
        assertEquals(6, local);
        assertFalse(RtReservoirReference.usesLocalCandidate(0, 1, true));
        assertEquals(8, RtReservoirReference.globalCandidateCount(8, false));
        assertThrows(IllegalArgumentException.class,
                () -> RtReservoirReference.globalCandidateCount(0, true));
    }

    @Test
    void streamingUpdateUsesTargetOverProposalAndDeterministicSelection() {
        RtReservoirReference.Reservoir reservoir = new RtReservoirReference.Reservoir();
        reservoir.update(new RtReservoirReference.Candidate(10L, 2.0, 1.0), 0.5);
        reservoir.update(new RtReservoirReference.Candidate(20L, 6.0, 1.0), 0.70);

        RtReservoirReference.Snapshot result = reservoir.snapshot();
        assertEquals(20L, result.selectedSampleId());
        assertEquals(8.0, result.weightSum());
        assertEquals(2.0, result.effectiveCount());
        assertEquals(8.0 / 12.0, result.finalWeight(), 1.0e-12);
    }

    @Test
    void zeroWeightCandidateCountsWithoutSelecting() {
        RtReservoirReference.Reservoir reservoir = new RtReservoirReference.Reservoir();
        reservoir.update(new RtReservoirReference.Candidate(1L, 0.0, 0.25), 0.0);

        RtReservoirReference.Snapshot result = reservoir.snapshot();
        assertFalse(result.selected());
        assertEquals(1.0, result.effectiveCount());
        assertEquals(0.0, result.weightSum());
        assertEquals(0.0, result.finalWeight());
    }

    @Test
    void mergeUsesSourceWeightAndEffectiveCount() {
        RtReservoirReference.Reservoir source = new RtReservoirReference.Reservoir();
        source.update(new RtReservoirReference.Candidate(7L, 4.0, 1.0), 0.0);
        source.update(new RtReservoirReference.Candidate(8L, 0.0, 1.0), 0.9);

        RtReservoirReference.Reservoir receiver = new RtReservoirReference.Reservoir();
        receiver.merge(source, 2.0, 0.0);

        RtReservoirReference.Snapshot result = receiver.snapshot();
        assertTrue(result.selected());
        assertEquals(7L, result.selectedSampleId());
        assertEquals(2.0, result.weightSum());
        assertEquals(2.0, result.effectiveCount());
        assertEquals(0.5, result.finalWeight());
        assertEquals(1, result.age());
    }

    @Test
    void temporalMergeClampsHistoricalEffectiveCount() {
        RtReservoirReference.Reservoir source = new RtReservoirReference.Reservoir();
        for (int candidate = 0; candidate < 100; candidate++) {
            source.update(new RtReservoirReference.Candidate(candidate, 2.0, 1.0), 0.99);
        }
        RtReservoirReference.Reservoir receiver = new RtReservoirReference.Reservoir();
        receiver.update(new RtReservoirReference.Candidate(1000L, 1.0, 1.0), 0.0);
        receiver.merge(source, 1.0, 0.0, 20.0);

        RtReservoirReference.Snapshot result = receiver.snapshot();
        assertEquals(21.0, result.effectiveCount());
        assertTrue(Double.isFinite(result.finalWeight()));
    }

    @Test
    void spatialMergeClampsNeighborAndPreservesSameFrameAge() {
        RtReservoirReference.Reservoir source = new RtReservoirReference.Reservoir();
        for (int candidate = 0; candidate < 32; candidate++) {
            source.update(new RtReservoirReference.Candidate(candidate, 2.0, 1.0), 0.99);
        }
        RtReservoirReference.Reservoir receiver = new RtReservoirReference.Reservoir();
        receiver.update(new RtReservoirReference.Candidate(1000L, 1.0, 1.0), 0.0);
        receiver.mergeSpatial(source, 1.0, 0.0, 1.0);

        RtReservoirReference.Snapshot result = receiver.snapshot();
        assertEquals(2.0, result.effectiveCount());
        assertEquals(0, result.age());
        assertTrue(Double.isFinite(result.finalWeight()));
    }

    @Test
    void extremeFiniteWeightsRemainFinite() {
        RtReservoirReference.Reservoir reservoir = new RtReservoirReference.Reservoir();
        reservoir.update(new RtReservoirReference.Candidate(1L, 1.0e250, 1.0), 0.0);
        reservoir.update(new RtReservoirReference.Candidate(2L, 5.0e249, 1.0), 0.9);

        RtReservoirReference.Snapshot result = reservoir.snapshot();
        assertTrue(Double.isFinite(result.weightSum()));
        assertTrue(Double.isFinite(result.finalWeight()));
        assertEquals(1.5e250, result.weightSum(), 1.0e236);
    }

    @Test
    void invalidInputsAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtReservoirReference.Candidate(1L, 1.0, 0.0));
        RtReservoirReference.Reservoir reservoir = new RtReservoirReference.Reservoir();
        assertThrows(IllegalArgumentException.class,
                () -> reservoir.update(new RtReservoirReference.Candidate(1L, 1.0, 1.0), 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> reservoir.merge(reservoir, 1.0, 0.5));
        assertThrows(IllegalArgumentException.class,
                () -> reservoir.merge(new RtReservoirReference.Reservoir(), 1.0, 0.5, 0.0));
    }
}
