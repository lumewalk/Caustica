package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.comfyfluffy.caustica.rt.gen.PathReservoirData;
import org.junit.jupiter.api.Test;

final class RtPathReservoirHistoryTest {
    @Test
    void reflectedAbiIncludesReplaySeedsAndControls() {
        assertEquals(112, PathReservoirData.BYTE_SIZE);
        assertEquals(112, RtPathReservoirHistory.BYTES_PER_RESERVOIR);
    }

    @Test
    void pingPongKeepsPreviousFrameSeparateFromWriteSlot() {
        RtHistoryState history = new RtHistoryState();
        RtPathReservoirHistory.State reservoirs = new RtPathReservoirHistory.State();

        RtPathReservoirHistory.Frame first = reservoirs.begin(history.beginFrame());
        assertEquals(0, first.writeSlot());
        assertFalse(first.previousAvailable());
        reservoirs.commit(first);
        history.markProduced(first.generation());

        RtPathReservoirHistory.Frame second = reservoirs.begin(history.beginFrame());
        assertEquals(1, second.writeSlot());
        assertEquals(0, second.previousSlot());
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
    void memoryAccountingUsesTwoFullResolutionSlots() {
        long perSlot = RtPathReservoirHistory.bytesPerSlot(1280, 673);
        assertEquals(96_481_280L, perSlot);
        assertEquals(192_962_560L, Math.multiplyExact(perSlot, 2L));
    }
}
