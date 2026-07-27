package dev.comfyfluffy.caustica.rt;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Set;

import org.junit.jupiter.api.Test;

final class RtHistoryStateTest {
    @Test
    void startsInvalidAndBecomesReusableOnlyAfterProducingAFrame() {
        RtHistoryState state = new RtHistoryState();

        RtHistoryState.Frame first = state.beginFrame();
        assertEquals(1L, first.generation());
        assertFalse(first.reuseAllowed());
        assertEquals(Set.of(RtHistoryState.Reason.SESSION_START), first.resetReasons());

        state.markProduced(first.generation());
        RtHistoryState.Frame second = state.beginFrame();
        assertEquals(first.generation(), second.generation());
        assertTrue(second.reuseAllowed());
        assertTrue(second.resetReasons().isEmpty());
    }

    @Test
    void coalescesReasonsRaisedBeforeTheNextFrame() {
        RtHistoryState state = reusableState();

        state.invalidate(RtHistoryState.Reason.RESOURCE_RELOAD);
        state.invalidate(RtHistoryState.Reason.MATERIAL_EPOCH);
        state.invalidate(RtHistoryState.Reason.RESOURCE_RELOAD);

        RtHistoryState.Frame reset = state.beginFrame();
        assertEquals(2L, reset.generation());
        assertFalse(reset.reuseAllowed());
        assertEquals(Set.of(
                RtHistoryState.Reason.RESOURCE_RELOAD,
                RtHistoryState.Reason.MATERIAL_EPOCH), reset.resetReasons());
    }

    @Test
    void staleCompletionCannotRevalidateANewerGeneration() {
        RtHistoryState state = reusableState();
        state.invalidate(RtHistoryState.Reason.RENDER_TARGET_RECREATED);
        RtHistoryState.Frame oldFrame = state.beginFrame();

        state.invalidate(RtHistoryState.Reason.CAMERA_CUT);
        state.markProduced(oldFrame.generation());

        RtHistoryState.Frame newFrame = state.beginFrame();
        assertEquals(oldFrame.generation() + 1L, newFrame.generation());
        assertFalse(newFrame.reuseAllowed());
        assertEquals(Set.of(RtHistoryState.Reason.CAMERA_CUT), newFrame.resetReasons());
    }

    @Test
    void sessionResetReturnsToFreshGeneration() {
        RtHistoryState state = reusableState();
        state.invalidate(RtHistoryState.Reason.WORLD_RENDER_STATE);
        state.beginFrame();

        state.resetForSession();
        RtHistoryState.Frame fresh = state.beginFrame();
        assertEquals(1L, fresh.generation());
        assertFalse(fresh.reuseAllowed());
        assertEquals(Set.of(RtHistoryState.Reason.SESSION_START), fresh.resetReasons());
    }

    private static RtHistoryState reusableState() {
        RtHistoryState state = new RtHistoryState();
        RtHistoryState.Frame first = state.beginFrame();
        state.markProduced(first.generation());
        assertTrue(state.beginFrame().reuseAllowed());
        return state;
    }
}
