package dev.comfyfluffy.caustica.rt;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.PathReservoirData;
import dev.comfyfluffy.caustica.rt.pipeline.RtPathTemporalPipeline;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Locale;
import org.lwjgl.system.MemoryUtil;
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
    static final int BYTES_PER_RESERVOIR = PathReservoirData.BYTE_SIZE;
    static final int SPATIAL_DIAGNOSTIC_CATEGORY_COUNT = 9;
    static final int SPATIAL_DIAGNOSTIC_PAIR_CURSOR_INDEX =
            SPATIAL_DIAGNOSTIC_CATEGORY_COUNT;
    static final int SPATIAL_DIAGNOSTIC_COUNTER_COUNT =
            SPATIAL_DIAGNOSTIC_CATEGORY_COUNT + 1;
    static final int SPATIAL_DIAGNOSTIC_COUNTER_BYTES =
            SPATIAL_DIAGNOSTIC_COUNTER_COUNT * Integer.BYTES;
    static final int SPATIAL_DIAGNOSTIC_PAIR_CAPACITY = 4096;
    static final int SPATIAL_DIAGNOSTIC_PAIR_BYTES =
            SPATIAL_DIAGNOSTIC_PAIR_CAPACITY * 2 * Float.BYTES;

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
    private RtBuffer spatialDiagnosticCounters;
    private RtBuffer spatialDiagnosticPairs;
    private RtPathTemporalPipeline temporalPipeline;
    private boolean spatialDiagnosticsPending;
    private int width = -1;
    private int height = -1;

    void ensure(RtContext ctx, int requestedWidth, int requestedHeight,
                RtImage receiverMotion, RtImage validationMetadata, RtImage debugColor,
                RtImage receiverPositionMaterial, RtImage receiverNormalRoughness) {
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
        spatialDiagnosticCounters = ctx.createBuffer(SPATIAL_DIAGNOSTIC_COUNTER_BYTES,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                true, "path spatial diagnostic counters");
        spatialDiagnosticPairs = ctx.createBuffer(SPATIAL_DIAGNOSTIC_PAIR_BYTES,
                VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true, "path spatial diagnostic target pairs");
        temporalPipeline = RtPathTemporalPipeline.create(ctx, receiverMotion.view,
                validationMetadata.view, debugColor.view,
                receiverPositionMaterial.view, receiverNormalRoughness.view,
                spatialDiagnosticCounters.handle, spatialDiagnosticPairs.handle);
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

    void recordTemporalAdmission(VkCommandBuffer cmd, ByteBuffer pushConstants,
                                 boolean spatialDiagnostics) {
        if (temporalPipeline == null) {
            throw new IllegalStateException("Path temporal pipeline used before allocation");
        }
        if (spatialDiagnostics) {
            VK10.vkCmdFillBuffer(cmd, spatialDiagnosticCounters.handle, 0L,
                    spatialDiagnosticCounters.size, 0);
            try (var stack = org.lwjgl.system.MemoryStack.stackPush()) {
                VulkanCommandEncoder.memoryBarrier(cmd, stack);
            }
        }
        temporalPipeline.dispatch(cmd, width, height, pushConstants);
        spatialDiagnosticsPending = spatialDiagnostics;
    }

    void pollSpatialDiagnosticCounters(RtContext ctx, long frameIndex) {
        if (!spatialDiagnosticsPending || frameIndex == 0L || frameIndex % 60L != 0L) {
            return;
        }
        ctx.waitIdle();
        spatialDiagnosticCounters.invalidate();
        spatialDiagnosticPairs.invalidate();
        IntBuffer counters = MemoryUtil.memIntBuffer(spatialDiagnosticCounters.mapped,
                SPATIAL_DIAGNOSTIC_COUNTER_COUNT);
        long[] values = new long[SPATIAL_DIAGNOSTIC_CATEGORY_COUNT];
        long total = 0L;
        for (int index = 0; index < values.length; index++) {
            values[index] = Integer.toUnsignedLong(counters.get(index));
            total += values[index];
        }
        long pairAttempts = Integer.toUnsignedLong(
                counters.get(SPATIAL_DIAGNOSTIC_PAIR_CURSOR_INDEX));
        int capturedPairs = (int) Math.min(pairAttempts, SPATIAL_DIAGNOSTIC_PAIR_CAPACITY);
        FloatBuffer pairValues = MemoryUtil.memFloatBuffer(
                spatialDiagnosticPairs.mapped, capturedPairs * 2);
        var rawMoments = new RtPathSpatialReuseReference.PairMoments();
        var logMoments = new RtPathSpatialReuseReference.PairMoments();
        for (int pair = 0; pair < capturedPairs; pair++) {
            float receiverTarget = pairValues.get(pair * 2);
            float sourceTarget = pairValues.get(pair * 2 + 1);
            if (!Float.isFinite(receiverTarget) || !Float.isFinite(sourceTarget)
                    || receiverTarget < 0.0f || sourceTarget < 0.0f) {
                continue;
            }
            rawMoments.add(receiverTarget, sourceTarget);
            logMoments.add(Math.log1p(receiverTarget), Math.log1p(sourceTarget));
        }
        if (total > 0L) {
            CausticaMod.LOGGER.info(
                    "RT path spatial categories: total={}, empty={} ({}%), admitted={} ({}%), "
                            + "footprintReject={} ({}%), depthReject={} ({}%), "
                            + "topologyReject={} ({}%), transportReject={} ({}%), "
                            + "noReconnection={} ({}%), neighborEmpty={} ({}%), "
                            + "surfaceReject={} ({}%), pairs={}/{}, validPairs={}, "
                            + "targetCorr={}, logTargetCorr={}",
                    total,
                    values[0], percent(values[0], total),
                    values[1], percent(values[1], total),
                    values[2], percent(values[2], total),
                    values[3], percent(values[3], total),
                    values[4], percent(values[4], total),
                    values[5], percent(values[5], total),
                    values[6], percent(values[6], total),
                    values[7], percent(values[7], total),
                    values[8], percent(values[8], total),
                    capturedPairs, pairAttempts, rawMoments.count(),
                    metric(rawMoments.correlation()), metric(logMoments.correlation()));
        }
        spatialDiagnosticsPending = false;
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
        return slots[0] != null && slots[1] != null
                && spatialDiagnosticCounters != null && spatialDiagnosticPairs != null
                && temporalPipeline != null;
    }

    void destroy() {
        if (temporalPipeline != null) {
            temporalPipeline.destroy();
            temporalPipeline = null;
        }
        if (spatialDiagnosticCounters != null) {
            spatialDiagnosticCounters.destroy();
            spatialDiagnosticCounters = null;
        }
        if (spatialDiagnosticPairs != null) {
            spatialDiagnosticPairs.destroy();
            spatialDiagnosticPairs = null;
        }
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            if (slots[slot] != null) {
                slots[slot].destroy();
                slots[slot] = null;
            }
        }
        width = -1;
        height = -1;
        spatialDiagnosticsPending = false;
        state.reset();
    }

    private static String percent(long value, long total) {
        return String.format(Locale.ROOT, "%.2f", value * 100.0 / total);
    }

    private static String metric(double value) {
        return Double.isFinite(value)
                ? String.format(Locale.ROOT, "%.4f", value)
                : "n/a";
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
