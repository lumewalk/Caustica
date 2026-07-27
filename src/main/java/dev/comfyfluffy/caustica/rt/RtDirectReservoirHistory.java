package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.DirectReservoirData;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;

/**
 * Double-buffered per-pixel direct-light reservoirs.
 *
 * <p>The current write slot is cleared at an explicit render-graph boundary before future candidate
 * generation. The other slot remains immutable history for the frame. Generation matching prevents an
 * old world/resource epoch from becoming visible after a reset.</p>
 */
final class RtDirectReservoirHistory {
    static final int SLOT_COUNT = 2;
    static final int BYTES_PER_RESERVOIR = DirectReservoirData.BYTE_SIZE;

    record Frame(long generation, int writeSlot, int previousSlot, boolean previousAvailable) {
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
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight) {
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
                    "direct reservoir history slot " + slot + " " + width + "x" + height);
        }
        state.reset();
    }

    Frame beginFrame(RtHistoryState.Frame historyFrame) {
        if (!ready()) {
            throw new IllegalStateException("Direct reservoirs used before allocation");
        }
        return state.begin(historyFrame);
    }

    void recordInitialize(VkCommandBuffer cmd, Frame frame) {
        RtBuffer write = slot(frame.writeSlot());
        VK10.vkCmdFillBuffer(cmd, write.handle, 0L, write.size, 0);
    }

    void commit(Frame frame) {
        state.commit(frame);
    }

    RtBuffer writeBuffer(Frame frame) {
        return slot(frame.writeSlot());
    }

    RtBuffer previousBuffer(Frame frame) {
        return frame.previousAvailable() ? slot(frame.previousSlot()) : null;
    }

    long allocatedBytes() {
        return ready() ? Math.multiplyExact(bytesPerSlot(width, height), SLOT_COUNT) : 0L;
    }

    static long bytesPerSlot(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Reservoir extent must be positive");
        }
        return Math.multiplyExact(
                Math.multiplyExact((long) width, (long) height),
                BYTES_PER_RESERVOIR);
    }

    boolean ready() {
        return slots[0] != null && slots[1] != null;
    }

    void destroy() {
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
            throw new IllegalStateException("Direct reservoirs used before allocation");
        }
        if (slot < 0 || slot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Direct reservoir slot out of range: " + slot);
        }
        return slots[slot];
    }
}
