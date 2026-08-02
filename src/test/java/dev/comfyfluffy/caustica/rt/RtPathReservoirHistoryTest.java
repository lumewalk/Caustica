package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.comfyfluffy.caustica.rt.gen.PathReservoirData;
import dev.comfyfluffy.caustica.rt.gen.WorldPushData;
import org.junit.jupiter.api.Test;

final class RtPathReservoirHistoryTest {
    @Test
    void reflectedAbiIncludesReplayAndReconnectionGeometryLanes() {
        assertEquals(176, PathReservoirData.BYTE_SIZE);
        assertEquals(176, RtPathReservoirHistory.BYTES_PER_RESERVOIR);
        assertEquals(640, WorldPushData.BYTE_SIZE);
        assertEquals(168, RtPathReservoirHistory.SHIFTED_DIAGNOSTIC_COUNTER_COUNT);
        assertEquals(672, RtPathReservoirHistory.SPATIAL_DIAGNOSTIC_COUNTER_BYTES);
        assertEquals(1 << 8, RtPathReservoirHistory.GUIDE_PREVIOUS_REPLAY_PASS_FLAG);
        assertEquals(1 << 9, RtPathReservoirHistory.GUIDE_PREVIOUS_AVAILABLE_FLAG);
    }

    @Test
    void pingPongKeepsPreviousFrameSeparateFromWriteSlot() {
        RtHistoryState history = new RtHistoryState();
        RtPathReservoirHistory.State reservoirs = new RtPathReservoirHistory.State();

        RtPathReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        assertEquals(0, first.writeSlot());
        assertEquals(1, first.scratchSlot());
        assertFalse(first.previousAvailable());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        RtPathReservoirHistory.Frame second = reservoirs.begin(history.beginFrame());
        assertEquals(1, second.writeSlot());
        assertEquals(0, second.previousSlot());
        assertEquals(second.previousSlot(), second.scratchSlot());
        assertTrue(second.previousAvailable());
    }

    @Test
    void invalidationDropsPreviousPathReservoirs() {
        RtHistoryState history = new RtHistoryState();
        RtPathReservoirHistory.State reservoirs = new RtPathReservoirHistory.State();
        RtPathReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        history.invalidate(RtHistoryState.Reason.MATERIAL_EPOCH);
        RtPathReservoirHistory.Frame reset = reservoirs.begin(history.beginFrame());
        assertEquals(1, reset.writeSlot());
        assertEquals(-1, reset.previousSlot());
        assertFalse(reset.previousAvailable());
    }

    @Test
    void shiftedSnapshotRequiresOneFrameContinuityAndMatchingGeneration() {
        var snapshots = new RtPathReservoirHistory.ShiftedSnapshotState();
        var first = new RtPathReservoirHistory.Frame(7L, 0, -1, false);
        var second = new RtPathReservoirHistory.Frame(7L, 1, 0, true);
        var nextGeneration = new RtPathReservoirHistory.Frame(8L, 0, 1, true);

        assertFalse(snapshots.previousAvailable(first, 100L));
        snapshots.commit(first, 100L);
        assertTrue(snapshots.previousAvailable(second, 101L));
        assertFalse(snapshots.previousAvailable(second, 102L));
        assertFalse(snapshots.previousAvailable(nextGeneration, 101L));

        snapshots.reset();
        assertFalse(snapshots.previousAvailable(second, 101L));
    }

    @Test
    void memoryAccountingUsesTwoFullResolutionSlots() {
        long perSlot = RtPathReservoirHistory.bytesPerSlot(1280, 673);
        assertEquals(151_613_440L, perSlot);
        assertEquals(303_226_880L, Math.multiplyExact(perSlot, 2L));
        // View 20 lazily adds one separate full-record scratch. It is not one of the two committed
        // history slots and cannot alias the persistent mapped snapshot that cross-frame replay reads.
        assertEquals(151_613_440L, RtPathReservoirHistory.bytesPerSlot(1280, 673));
        assertEquals(137_830_400L, Math.multiplyExact(1280L * 673L,
                dev.comfyfluffy.caustica.rt.gen.PathSourceRootData.BYTE_SIZE));
    }

    @Test
    void sortedPercentileInterpolatesBoundedDiagnosticSamples() {
        double[] values = {1.0, 2.0, 4.0, 8.0};
        assertEquals(3.0, RtPathReservoirHistory.sortedPercentile(values, 4, 0.5), 1.0e-12);
        assertEquals(7.4, RtPathReservoirHistory.sortedPercentile(values, 4, 0.95), 1.0e-12);
        assertEquals(8.0, RtPathReservoirHistory.sortedPercentile(values, 4, 1.0), 1.0e-12);
        assertThrows(IllegalArgumentException.class,
                () -> RtPathReservoirHistory.sortedPercentile(values, 0, 0.5));
    }
}
