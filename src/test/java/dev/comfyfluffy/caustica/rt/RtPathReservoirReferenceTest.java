package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RtPathReservoirReferenceTest {
    private static RtPathReservoirReference.PathSample sample(
            long id, long topology, long geometry, long material, double footprint) {
        return RtPathReservoirReference.PathSample.candidate(
                id, 1, topology, geometry, material, 0, id * 17L,
                2.0, 4.0, 1.0, footprint);
    }

    @Test
    void compatibilityRequiresPathTopologyAndBoundedFootprint() {
        var receiver = sample(1L, 10L, 3L, 7L, 1.0);
        assertTrue(RtPathReservoirReference.compatible(receiver,
                sample(2L, 10L, 3L, 7L, 3.9)));
        assertFalse(RtPathReservoirReference.compatible(receiver,
                sample(3L, 11L, 3L, 7L, 1.0)));
        assertFalse(RtPathReservoirReference.compatible(receiver,
                sample(4L, 10L, 4L, 7L, 1.0)));
        assertFalse(RtPathReservoirReference.compatible(receiver,
                sample(5L, 10L, 3L, 7L, 4.1)));
    }

    @Test
    void compatibleMergeIncludesShiftJacobianInReservoirWeight() {
        var source = new RtPathReservoirReference.Reservoir();
        source.update(sample(10L, 4L, 2L, 9L, 1.0), 0.0);

        var shifted = new RtPathReservoirReference.PathSample(
                10L, 1, 4L, 2L, 9L, 0, 170L,
                3.0, 2.0, 1.0, 2.0, 1.0);
        var receiver = new RtPathReservoirReference.Reservoir();
        receiver.merge(source, shifted, 0.0, 8.0);

        var result = receiver.snapshot();
        assertTrue(result.selected());
        assertEquals(10L, result.selectedSampleId());
        assertEquals(4.0, result.weightSum());
        assertEquals(1.0, result.effectiveCount());
        assertEquals(2.0, result.finalWeight());
        assertEquals(6.0, result.estimatedContribution());
        assertEquals(1, result.age());
    }

    @Test
    void incompatibleMergeDoesNotInflateEffectiveCount() {
        var source = new RtPathReservoirReference.Reservoir();
        source.update(sample(1L, 4L, 2L, 9L, 1.0), 0.0);
        var incompatible = sample(2L, 5L, 2L, 9L, 1.0);
        var receiver = new RtPathReservoirReference.Reservoir();
        receiver.merge(source, incompatible, 0.0, 8.0);

        var result = receiver.snapshot();
        assertFalse(result.selected());
        assertEquals(0.0, result.effectiveCount());
        assertEquals(0.0, result.finalWeight());
    }

    @Test
    void invalidPathContractValuesAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new RtPathReservoirReference.PathSample(
                        1L, 0, 1L, 1L, 1L, 0, 1L,
                        1.0, 1.0, 1.0, 0.0, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> RtPathReservoirReference.PathSample.candidate(
                        1L, 0, 1L, 1L, 1L, 0, 1L,
                        1.0, 1.0, 1.0, Double.NaN));
    }
}
