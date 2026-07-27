package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

final class RtSurfaceHistoryTest {
    @Test
    void alternatesWriteSlotsAndExposesOnlyCommittedPreviousData() {
        RtHistoryState history = new RtHistoryState();
        RtSurfaceHistory.State surfaces = new RtSurfaceHistory.State();

        RtHistoryState.Frame firstHistory = history.beginFrame();
        RtSurfaceHistory.Frame first = surfaces.begin(firstHistory);
        assertEquals(0, first.writeSlot());
        assertFalse(first.previousAvailable());
        surfaces.commit(first);
        assertTrue(history.markProduced(first.generation()));

        RtHistoryState.Frame secondHistory = history.beginFrame();
        RtSurfaceHistory.Frame second = surfaces.begin(secondHistory);
        assertEquals(1, second.writeSlot());
        assertEquals(0, second.previousSlot());
        assertTrue(second.previousAvailable());
        surfaces.commit(second);
        assertTrue(history.markProduced(second.generation()));

        RtSurfaceHistory.Frame third = surfaces.begin(history.beginFrame());
        assertEquals(0, third.writeSlot());
        assertEquals(1, third.previousSlot());
        assertTrue(third.previousAvailable());
    }

    @Test
    void rejectsPreviousSlotAcrossHistoryGenerationChange() {
        RtHistoryState history = new RtHistoryState();
        RtSurfaceHistory.State surfaces = new RtSurfaceHistory.State();

        RtHistoryState.Frame firstHistory = history.beginFrame();
        RtSurfaceHistory.Frame first = surfaces.begin(firstHistory);
        surfaces.commit(first);
        history.markProduced(first.generation());

        history.invalidate(RtHistoryState.Reason.CAMERA_CUT);
        RtSurfaceHistory.Frame reset = surfaces.begin(history.beginFrame());
        assertFalse(reset.previousAvailable());
        assertEquals(-1, reset.previousSlot());
    }

    @Test
    void resetForcesFreshSlotZero() {
        RtHistoryState history = new RtHistoryState();
        RtSurfaceHistory.State surfaces = new RtSurfaceHistory.State();
        RtSurfaceHistory.Frame first = surfaces.begin(history.beginFrame());
        surfaces.commit(first);

        surfaces.reset();
        RtSurfaceHistory.Frame fresh = surfaces.begin(history.beginFrame());
        assertEquals(0, fresh.writeSlot());
        assertFalse(fresh.previousAvailable());
    }
}
