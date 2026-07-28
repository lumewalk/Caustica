package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.pipeline.RtPathTemporalPipeline;
import java.nio.ByteBuffer;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Double-buffered GPU storage and temporal-admission diagnostics for ReSTIR PT path reservoirs.
 *
 * <p>Candidate generation remains in the wavefront ray pass. This owner validates historical
 * reprojection and strict path compatibility, while replay/reconnection and the eventual GRIS merge
 * remain explicit later stages.</p>
 */
final class RtPathReservoirHistory {
    static final int SLOT_COUNT = 2;
    static final int BYTES_PER_RESERVOIR = 64;

    record Frame(long generation, int writeSlot, int previousSlot, boolean previousAvailable) {
        int finalSlot() {
            return writeSlot;
        }
    }

    static final class State {
        private int latestSlot = -1;
        private long latestGeneration = Long.MIN_VALUE;

        Frame begin(RtHistoryState.Frame historyFrame) {
            boolean previousAvailable = historyFrame.reuseAllowed()
                    && latestSlot >= 0
                    && latestGeneration == historyFrame.generation();
            int writeSlot = latestSlot < 0 ? 0 : 1 - latestSlot;
            return new Frame(historyFrame.generation(), writeSlot,
                    previousAvailable ? latestSlot : -1, previousAvailable);
        }

        void commit(Frame frame) {
            latestSlot = frame.writeSlot();
            latestGeneration = frame.generation();
        }

        void reset() {
            latestSlot = -1;
            latestGeneration = Long.MIN_VALUE;
        }
    }

    private final State state = new State();
    private final RtBuffer[] slots = new RtBuffer[SLOT_COUNT];
    private RtPathTemporalPipeline temporalPipeline;
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight,
                RtImage receiverMotion, RtImage validationMetadata, RtImage debugColor) {
        if (ready() && width == requestedWidth && height == requestedHeight) {
            return;
        }
        destroy();
        width = requestedWidth;
        height = requestedHeight;
        long bytesPerSlot = bytesPerSlot(width, height);
        int usage = VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            slots[slot] = ctx.createBuffer(bytesPerSlot, usage, false,
                    "path reservoir history slot " + slot + " " + width + "x" + height);
        }
        temporalPipeline = RtPathTemporalPipeline.create(ctx, receiverMotion.view,
                validationMetadata.view, debugColor.view);
        state.reset();
    }

    Frame beginFrame(RtHistoryState.Frame historyFrame) {
        if (!ready()) {
            throw new IllegalStateException("Path reservoirs used before allocation");
        }
        return state.begin(historyFrame);
    }

    void commit(Frame frame) {
        state.commit(frame);
    }

    void recordTemporalAdmission(VkCommandBuffer cmd, ByteBuffer pushConstants) {
        if (temporalPipeline == null) {
            throw new IllegalStateException("Path temporal pipeline used before allocation");
        }
        temporalPipeline.dispatch(cmd, width, height, pushConstants);
    }

    void reset() {
        state.reset();
    }

    RtBuffer finalBuffer(Frame frame) {
        return slot(frame.finalSlot());
    }

    RtBuffer previousBuffer(Frame frame) {
        return frame.previousAvailable() ? slot(frame.previousSlot()) : null;
    }

    long allocatedBytes() {
        return ready() ? Math.multiplyExact(bytesPerSlot(width, height), SLOT_COUNT) : 0L;
    }

    static long bytesPerSlot(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Path reservoir extent must be positive");
        }
        return Math.multiplyExact(Math.multiplyExact((long) width, height),
                BYTES_PER_RESERVOIR);
    }

    boolean ready() {
        return slots[0] != null && slots[1] != null && temporalPipeline != null;
    }

    void destroy() {
        if (temporalPipeline != null) {
            temporalPipeline.destroy();
            temporalPipeline = null;
        }
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            if (slots[slot] != null) {
                slots[slot].destroy();
                slots[slot] = null;
            }
        }
        width = -1;
        height = -1;
        state.reset();
    }

    private RtBuffer slot(int slot) {
        if (!ready()) {
            throw new IllegalStateException("Path reservoirs used before allocation");
        }
        if (slot < 0 || slot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Path reservoir slot out of range: " + slot);
        }
        return slots[slot];
    }
}
