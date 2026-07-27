package dev.comfyfluffy.caustica.rt;

import java.util.EnumSet;
import java.util.Set;

/**
 * Renderer-wide temporal-history generation and reset contract.
 *
 * <p>Every temporal consumer (motion reprojection today; reservoirs later) observes the same generation.
 * Invalidations that happen before the next rendered frame are coalesced into one generation change, while
 * a completion from an older in-flight generation can never make history reusable again.</p>
 */
final class RtHistoryState {
    enum Reason {
        SESSION_START,
        RENDER_TARGET_RECREATED,
        WORLD_RENDER_STATE,
        RESOURCE_RELOAD,
        MATERIAL_EPOCH,
        CAMERA_CUT
    }

    record Frame(long generation, boolean reuseAllowed, Set<Reason> resetReasons) {
        boolean resetRequired() {
            return !resetReasons.isEmpty();
        }
    }

    private final EnumSet<Reason> pendingReasons = EnumSet.noneOf(Reason.class);
    private long generation;
    private boolean reusable;

    RtHistoryState() {
        resetForSession();
    }

    synchronized void resetForSession() {
        generation = 0L;
        reusable = false;
        pendingReasons.clear();
        invalidate(Reason.SESSION_START);
    }

    synchronized void invalidate(Reason reason) {
        if (pendingReasons.isEmpty()) {
            generation = Math.incrementExact(generation);
        }
        pendingReasons.add(reason);
        reusable = false;
    }

    synchronized Frame beginFrame() {
        Set<Reason> reasons = pendingReasons.isEmpty()
                ? Set.of()
                : Set.copyOf(pendingReasons);
        boolean allowReuse = reusable && reasons.isEmpty();
        pendingReasons.clear();
        return new Frame(generation, allowReuse, reasons);
    }

    synchronized boolean markProduced(long producedGeneration) {
        if (producedGeneration == generation && pendingReasons.isEmpty()) {
            reusable = true;
            return true;
        }
        return false;
    }
}
