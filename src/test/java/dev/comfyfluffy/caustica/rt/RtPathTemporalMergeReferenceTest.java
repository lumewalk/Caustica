package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

final class RtPathTemporalMergeReferenceTest {
    private static RtPathReservoirReference.PathSample sample(
            long id, long topology, double footprint) {
        return RtPathReservoirReference.PathSample.candidate(
                id, 2, topology, 3L, 7L, 2, id * 31L,
                3.0, 2.0, 1.0, footprint);
    }

    @Test
    void identityReplayMergeIsAcceptedAndUsesOneJacobian() {
        var sourceSample = sample(10L, 42L, 1.0);
        var source = new RtPathReservoirReference.Reservoir();
        source.update(sourceSample, 0.0);
        var receiver = new RtPathReservoirReference.Reservoir();
        var decision = RtPathTemporalMergeReference.merge(
                receiver, source, sourceSample, sourceSample,
                RtPathTemporalMergeReference.Mapping.identity(true), 0.0);

        assertEquals(RtPathTemporalMergeReference.Decision.ACCEPTED, decision);
        var snapshot = receiver.snapshot();
        assertEquals(1.0, snapshot.effectiveCount());
        assertEquals(1.0, snapshot.finalWeight());
        assertEquals(3.0, snapshot.estimatedContribution());
        assertEquals(1, snapshot.age());
    }

    @Test
    void replayMismatchDoesNotReuseHistory() {
        var sourceSample = sample(10L, 42L, 1.0);
        var source = new RtPathReservoirReference.Reservoir();
        source.update(sourceSample, 0.0);
        var receiver = new RtPathReservoirReference.Reservoir();
        var decision = RtPathTemporalMergeReference.merge(
                receiver, source, sourceSample, sourceSample,
                RtPathTemporalMergeReference.Mapping.identity(false), 0.0);

        assertEquals(RtPathTemporalMergeReference.Decision.REPLAY_MISMATCH, decision);
        assertFalse(receiver.snapshot().selected());
        assertEquals(0.0, receiver.snapshot().effectiveCount());
    }

    @Test
    void historicalEffectiveCountIsCappedWithoutChangingEnergy() {
        var sourceSample = sample(10L, 42L, 1.0);
        var source = new RtPathReservoirReference.Reservoir();
        for (int i = 0; i < 16; i++) {
            source.update(sourceSample, 0.0);
        }
        var receiver = new RtPathReservoirReference.Reservoir();
        var decision = RtPathTemporalMergeReference.merge(
                receiver, source, sourceSample, sourceSample,
                RtPathTemporalMergeReference.Mapping.identity(true), 0.0);

        assertEquals(RtPathTemporalMergeReference.Decision.ACCEPTED, decision);
        assertEquals(8.0, receiver.snapshot().effectiveCount());
        assertEquals(1.0, receiver.snapshot().finalWeight());
        assertEquals(3.0, receiver.snapshot().estimatedContribution());
    }

    @Test
    void footprintAndJacobianAreExplicitAdmissionRules() {
        var sourceSample = sample(10L, 42L, 1.0);
        var source = new RtPathReservoirReference.Reservoir();
        source.update(sourceSample, 0.0);
        var receiver = new RtPathReservoirReference.Reservoir();

        var wide = sample(11L, 42L, 4.1);
        assertEquals(RtPathTemporalMergeReference.Decision.FOOTPRINT_MISMATCH,
                RtPathTemporalMergeReference.merge(receiver, source, sourceSample, wide,
                        RtPathTemporalMergeReference.Mapping.identity(true), 0.0));

        var nonIdentity = new RtPathReservoirReference.PathSample(
                10L, 2, 42L, 3L, 7L, 2, 10L * 31L,
                3.0, 2.0, 1.0, 1.5, 1.0);
        assertEquals(RtPathTemporalMergeReference.Decision.INVALID_JACOBIAN,
                RtPathTemporalMergeReference.merge(receiver, source, sourceSample, nonIdentity,
                        RtPathTemporalMergeReference.Mapping.identity(true), 0.0));
        assertFalse(receiver.snapshot().selected());
    }
}
