package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.comfyfluffy.caustica.rt.gen.DirectReservoirData;
import org.junit.jupiter.api.Test;

final class RtDirectReservoirHistoryTest {
    @Test
    void reflectedAbiIsClarityFirstFiveLaneRecord() {
        assertEquals(80, DirectReservoirData.BYTE_SIZE);
        assertEquals(80, RtDirectReservoirHistory.BYTES_PER_RESERVOIR);
    }

    @Test
    void alternatesSlotsWithinOneHistoryGeneration() {
        RtHistoryState history = new RtHistoryState();
        RtDirectReservoirHistory.State reservoirs = new RtDirectReservoirHistory.State();

        RtDirectReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        assertEquals(0, first.writeSlot());
        assertFalse(first.previousAvailable());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        RtDirectReservoirHistory.Frame second = reservoirs.begin(history.beginFrame());
        assertEquals(1, second.writeSlot());
        assertEquals(0, second.previousSlot());
        assertTrue(second.previousAvailable());
    }

    @Test
    void rejectsOldSlotAfterInvalidation() {
        RtHistoryState history = new RtHistoryState();
        RtDirectReservoirHistory.State reservoirs = new RtDirectReservoirHistory.State();
        RtDirectReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        history.invalidate(RtHistoryState.Reason.MATERIAL_EPOCH);
        RtDirectReservoirHistory.Frame reset = reservoirs.begin(history.beginFrame());
        assertFalse(reset.previousAvailable());
        assertEquals(-1, reset.previousSlot());
    }

    @Test
    void memoryAccountingIncludesBothSlots() {
        long perSlot = RtDirectReservoirHistory.bytesPerSlot(1280, 673);
        assertEquals(68_915_200L, perSlot);
        assertEquals(137_830_400L, Math.multiplyExact(perSlot, 2L));
    }
}
