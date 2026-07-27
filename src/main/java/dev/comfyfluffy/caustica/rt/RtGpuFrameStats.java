package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaMod;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import net.fabricmc.loader.api.FabricLoader;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;

/**
 * Opt-in, non-blocking GPU timestamps for the composite command buffer. Query results are only read after
 * the graphics timeline already proves their submission complete, so diagnostics never wait for the GPU.
 */
final class RtGpuFrameStats {
    private static final int SLOT_COUNT = 8;
    private static final int QUERY_COUNT = 9;
    private static final int ENTITY_BLAS = 1;
    private static final int TLAS = 2;
    private static final int TRACE_PRIMARY = 3;
    private static final int TRACE_INDIRECT = 4;
    private static final int UPSCALE = 5;
    private static final int EXPOSURE = 6;
    private static final int DISPLAY_MAP = 7;
    private static final int COPY_OUTPUT = 8;

    private final Slot[] slots = new Slot[SLOT_COUNT];
    private long queryPool;
    private int nextSlot;
    private PrintWriter csv;
    private boolean csvOpenAttempted;
    private boolean unsupportedLogged;

    RtGpuFrameStats() {
        for (int i = 0; i < slots.length; i++) {
            slots[i] = new Slot(i * QUERY_COUNT);
        }
    }

    Slot begin(RtContext ctx, VkCommandBuffer cmd, RtGpuExecutor.GraphicsUseWaiter waiter,
               long frameIndex, int renderWidth, int renderHeight, int displayWidth, int displayHeight) {
        if (!RtFrameStats.enabled()) {
            return null;
        }
        int validBits = ctx.graphicsTimestampValidBits();
        if (validBits == 0) {
            if (!unsupportedLogged) {
                unsupportedLogged = true;
                CausticaMod.LOGGER.warn("GPU frame timestamps unavailable: graphics queue reports zero valid bits");
            }
            return null;
        }
        ensurePool(ctx);
        reapCompleted(ctx, waiter);

        for (int i = 0; i < slots.length; i++) {
            int index = (nextSlot + i) % slots.length;
            Slot slot = slots[index];
            if (slot.pending || slot.recording) {
                continue;
            }
            nextSlot = (index + 1) % slots.length;
            slot.recording = true;
            slot.frameIndex = frameIndex;
            slot.renderWidth = renderWidth;
            slot.renderHeight = renderHeight;
            slot.displayWidth = displayWidth;
            slot.displayHeight = displayHeight;
            slot.hasEntityBlas = false;
            slot.rrDone = false;
            VK10.vkCmdResetQueryPool(cmd, queryPool, slot.firstQuery, QUERY_COUNT);
            writeTimestamp(cmd, slot, 0);
            return slot;
        }
        return null;
    }

    void markEntityBlas(Slot slot, VkCommandBuffer cmd, boolean hasEntityBlas) {
        if (slot != null) {
            slot.hasEntityBlas = hasEntityBlas;
            writeTimestamp(cmd, slot, ENTITY_BLAS);
        }
    }

    void markTlas(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, TLAS);
    }

    void markTracePrimary(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, TRACE_PRIMARY);
    }

    void markTraceIndirect(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, TRACE_INDIRECT);
    }

    void markUpscale(Slot slot, VkCommandBuffer cmd, boolean rrDone) {
        if (slot != null) {
            slot.rrDone = rrDone;
            writeTimestamp(cmd, slot, UPSCALE);
        }
    }

    void markExposure(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, EXPOSURE);
    }

    void markDisplayMap(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, DISPLAY_MAP);
    }

    void markCopyOutput(Slot slot, VkCommandBuffer cmd) {
        mark(slot, cmd, COPY_OUTPUT);
    }

    void commit(Slot slot, RtGpuExecutor.GraphicsUse graphicsUse) {
        if (slot == null) {
            return;
        }
        slot.recording = false;
        slot.pending = true;
        slot.graphicsUse = graphicsUse;
    }

    void cancel(Slot slot) {
        if (slot != null && slot.recording) {
            slot.recording = false;
            slot.graphicsUse = null;
        }
    }

    void destroy(RtContext ctx) {
        if (queryPool != 0L && ctx != null) {
            for (Slot slot : slots) {
                if (slot.pending) {
                    readCompleted(ctx, slot);
                }
                slot.reset();
            }
            VK10.vkDestroyQueryPool(ctx.vk(), queryPool, null);
            queryPool = 0L;
        }
        if (csv != null) {
            csv.close();
            csv = null;
        }
        csvOpenAttempted = false;
        unsupportedLogged = false;
        nextSlot = 0;
    }

    private void ensurePool(RtContext ctx) {
        if (queryPool != 0L) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkQueryPoolCreateInfo createInfo = VkQueryPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .queryType(VK10.VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(SLOT_COUNT * QUERY_COUNT);
            LongBuffer result = stack.mallocLong(1);
            RtContext.check(VK10.vkCreateQueryPool(ctx.vk(), createInfo, null, result),
                    "vkCreateQueryPool(frame timestamps)");
            queryPool = result.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_QUERY_POOL, queryPool, "RT frame GPU timestamps");
        }
    }

    private void reapCompleted(RtContext ctx, RtGpuExecutor.GraphicsUseWaiter waiter) {
        for (Slot slot : slots) {
            if (slot.pending && waiter.isComplete(slot.graphicsUse) && readCompleted(ctx, slot)) {
                slot.reset();
            }
        }
    }

    private boolean readCompleted(RtContext ctx, Slot slot) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer timestamps = stack.mallocLong(QUERY_COUNT);
            int result = VK10.vkGetQueryPoolResults(ctx.vk(), queryPool, slot.firstQuery, QUERY_COUNT,
                    timestamps, Long.BYTES, VK10.VK_QUERY_RESULT_64_BIT);
            if (result == VK10.VK_NOT_READY) {
                return false;
            }
            RtContext.check(result, "vkGetQueryPoolResults(frame timestamps)");
            writeRow(ctx, slot, timestamps);
            return true;
        }
    }

    private void writeRow(RtContext ctx, Slot slot, LongBuffer timestamps) {
        PrintWriter writer = csv();
        if (writer == null) {
            return;
        }
        int bits = ctx.graphicsTimestampValidBits();
        double period = ctx.timestampPeriodNanos();
        double entityBlasMs = slot.hasEntityBlas
                ? millis(timestampDelta(timestamps.get(0), timestamps.get(ENTITY_BLAS), bits), period) : 0.0;
        double tlasMs = millis(timestampDelta(timestamps.get(ENTITY_BLAS), timestamps.get(TLAS), bits), period);
        double tracePrimaryMs = millis(
                timestampDelta(timestamps.get(TLAS), timestamps.get(TRACE_PRIMARY), bits), period);
        // Includes the primary-to-indirect visibility barrier recorded between the two timestamps.
        double traceIndirectMs = millis(
                timestampDelta(timestamps.get(TRACE_PRIMARY), timestamps.get(TRACE_INDIRECT), bits), period);
        double traceMs = millis(timestampDelta(timestamps.get(TLAS), timestamps.get(TRACE_INDIRECT), bits), period);
        double upscaleMs = millis(timestampDelta(timestamps.get(TRACE_INDIRECT), timestamps.get(UPSCALE), bits), period);
        double exposureMs = millis(timestampDelta(timestamps.get(UPSCALE), timestamps.get(EXPOSURE), bits), period);
        double displayMapMs = millis(timestampDelta(timestamps.get(EXPOSURE), timestamps.get(DISPLAY_MAP), bits), period);
        double copyOutputMs = millis(timestampDelta(timestamps.get(DISPLAY_MAP), timestamps.get(COPY_OUTPUT), bits), period);
        double totalMs = millis(timestampDelta(timestamps.get(0), timestamps.get(COPY_OUTPUT), bits), period);
        writer.printf(Locale.ROOT,
                "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                slot.frameIndex, slot.renderWidth, slot.renderHeight, slot.displayWidth, slot.displayHeight,
                totalMs, entityBlasMs, tlasMs, traceMs, tracePrimaryMs, traceIndirectMs,
                slot.rrDone ? upscaleMs : 0.0,
                slot.rrDone ? 0.0 : upscaleMs, exposureMs, displayMapMs, copyOutputMs);
        writer.flush();
    }

    private PrintWriter csv() {
        if (csv != null || csvOpenAttempted) {
            return csv;
        }
        csvOpenAttempted = true;
        Path directory = FabricLoader.getInstance().getGameDir().resolve("rt-frame-stats");
        Path file = directory.resolve("gpu.csv");
        try {
            Files.createDirectories(directory);
            csv = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8));
            csv.println("frame,renderWidth,renderHeight,displayWidth,displayHeight,totalMs,entityBlasMs,"
                    + "tlasMs,traceMs,tracePrimaryMs,traceIndirectMs,dlssRrMs,upscaleMs,exposureMs,"
                    + "displayMapMs,copyOutputMs");
            csv.flush();
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("RtGpuFrameStats: failed to open CSV {}: {}", file, e.toString());
        }
        return csv;
    }

    private void mark(Slot slot, VkCommandBuffer cmd, int queryOffset) {
        if (slot != null) {
            writeTimestamp(cmd, slot, queryOffset);
        }
    }

    private void writeTimestamp(VkCommandBuffer cmd, Slot slot, int queryOffset) {
        KHRSynchronization2.vkCmdWriteTimestamp2KHR(cmd,
                KHRSynchronization2.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
                queryPool, slot.firstQuery + queryOffset);
    }

    static long timestampDelta(long start, long end, int validBits) {
        if (validBits < 1 || validBits > Long.SIZE) {
            throw new IllegalArgumentException("timestamp valid bits must be in [1, 64]");
        }
        long delta = end - start;
        if (validBits == Long.SIZE) {
            return delta;
        }
        return delta & ((1L << validBits) - 1L);
    }

    static double millis(long ticks, double periodNanos) {
        return ticks * periodNanos / 1_000_000.0;
    }

    static final class Slot {
        private final int firstQuery;
        private boolean recording;
        private boolean pending;
        private boolean hasEntityBlas;
        private boolean rrDone;
        private long frameIndex;
        private int renderWidth;
        private int renderHeight;
        private int displayWidth;
        private int displayHeight;
        private RtGpuExecutor.GraphicsUse graphicsUse;

        private Slot(int firstQuery) {
            this.firstQuery = firstQuery;
        }

        private void reset() {
            recording = false;
            pending = false;
            graphicsUse = null;
        }
    }
}
