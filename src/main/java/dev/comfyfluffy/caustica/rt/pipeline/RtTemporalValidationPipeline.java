package dev.comfyfluffy.caustica.rt.pipeline;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

/**
 * Immutable two-slot descriptor seam for temporal surface-history validation.
 *
 * <p>Each descriptor set names one history slot. The current guides and output metadata are shared,
 * while dispatch selects the set containing the previous slot. Descriptors therefore never change
 * while frames are in flight.</p>
 */
public final class RtTemporalValidationPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";
    private static final int SLOT_COUNT = 2;
    private static final int IMAGE_BINDING_COUNT = 7;
    private static final int PUSH_BYTES = 108;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long pipeline;
    private boolean destroyed;

    private RtTemporalValidationPipeline(RtContext ctx, long descriptorSetLayout, long descriptorPool,
                                         long[] descriptorSets, long pipelineLayout, long pipeline) {
        this.ctx = ctx;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSets = descriptorSets;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static RtTemporalValidationPipeline create(
            RtContext ctx,
            long currentNormalView,
            long currentDepthView,
            long currentMotionView,
            long[] previousNormalViews,
            long[] previousDepthViews,
            long metadataView,
            long debugColorView) {
        if (previousNormalViews.length != SLOT_COUNT || previousDepthViews.length != SLOT_COUNT) {
            throw new IllegalArgumentException("Temporal validation requires exactly two history slots");
        }
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(IMAGE_BINDING_COUNT, stack);
            for (int binding = 0; binding < IMAGE_BINDING_COUNT; binding++) {
                bindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo dslInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslInfo, null, handle),
                    "vkCreateDescriptorSetLayout(temporal validation)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, descriptorSetLayout,
                    "temporal validation descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(SLOT_COUNT * IMAGE_BINDING_COUNT);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(SLOT_COUNT).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(temporal validation)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptorPool,
                    "temporal validation descriptor pool");

            LongBuffer layouts = stack.mallocLong(SLOT_COUNT);
            for (int slot = 0; slot < SLOT_COUNT; slot++) {
                layouts.put(slot, descriptorSetLayout);
            }
            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool).pSetLayouts(layouts);
            LongBuffer setsBuffer = stack.mallocLong(SLOT_COUNT);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setsBuffer),
                    "vkAllocateDescriptorSets(temporal validation)");
            long[] descriptorSets = new long[SLOT_COUNT];
            setsBuffer.get(descriptorSets);
            for (int slot = 0; slot < SLOT_COUNT; slot++) {
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, descriptorSets[slot],
                        "temporal validation descriptor set " + slot);
            }

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(PUSH_BYTES);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, layoutInfo, null, handle),
                    "vkCreatePipelineLayout(temporal validation)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout,
                    "temporal validation pipeline layout");

            long module = loadModule(vk, stack, "temporal_validate.comp.spv");
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module,
                    "temporal validation shader module");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, pipelineInfo, null, handle),
                    "vkCreateComputePipelines(temporal validation)");
            long pipeline = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline,
                    "temporal validation compute pipeline");
            VK10.vkDestroyShaderModule(vk, module, null);

            bindImages(vk, stack, descriptorSets,
                    currentNormalView, currentDepthView, currentMotionView,
                    previousNormalViews, previousDepthViews, metadataView, debugColorView);
            return new RtTemporalValidationPipeline(ctx, descriptorSetLayout, descriptorPool,
                    descriptorSets, pipelineLayout, pipeline);
        }
    }

    public void dispatch(VkCommandBuffer cmd, int previousSlot, int width, int height,
                         Matrix4fc currentFromPreviousClip, boolean previousAvailable, int debugView,
                         float normalCosMin, float roughnessTolerance,
                         float depthBaseTolerance, float depthRelativeTolerance,
                         float depthMaxTolerance,
                         float projectionM22, float projectionM23,
                         float projectionM32, float projectionM33) {
        if (previousSlot < 0 || previousSlot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Previous history slot out of range: " + previousSlot);
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "temporal validation")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(descriptorSets[previousSlot]), null);
            ByteBuffer push = stack.malloc(PUSH_BYTES);
            currentFromPreviousClip.get(0, push);
            push.putInt(64, previousAvailable ? 1 : 0);
            push.putInt(68, debugView);
            push.putFloat(72, normalCosMin);
            push.putFloat(76, roughnessTolerance);
            push.putFloat(80, depthBaseTolerance);
            push.putFloat(84, depthRelativeTolerance);
            push.putFloat(88, depthMaxTolerance);
            push.putFloat(92, projectionM22);
            push.putFloat(96, projectionM23);
            push.putFloat(100, projectionM32);
            push.putFloat(104, projectionM33);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static void bindImages(
            VkDevice vk, MemoryStack stack, long[] sets,
            long currentNormalView, long currentDepthView, long currentMotionView,
            long[] previousNormalViews, long[] previousDepthViews,
            long metadataView, long debugColorView) {
        VkDescriptorImageInfo.Buffer imageInfos =
                VkDescriptorImageInfo.calloc(SLOT_COUNT * IMAGE_BINDING_COUNT, stack);
        VkWriteDescriptorSet.Buffer writes =
                VkWriteDescriptorSet.calloc(SLOT_COUNT * IMAGE_BINDING_COUNT, stack);
        for (int slot = 0; slot < SLOT_COUNT; slot++) {
            long[] views = {
                    currentNormalView,
                    currentDepthView,
                    currentMotionView,
                    previousNormalViews[slot],
                    previousDepthViews[slot],
                    metadataView,
                    debugColorView
            };
            for (int binding = 0; binding < IMAGE_BINDING_COUNT; binding++) {
                int index = slot * IMAGE_BINDING_COUNT + binding;
                imageInfos.get(index).imageView(views[binding])
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(index).sType$Default().dstSet(sets[slot]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(
                                imageInfos.address(index), 1));
            }
        }
        VK10.vkUpdateDescriptorSets(vk, writes, null);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtTemporalValidationPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module),
                    "vkCreateShaderModule(" + name + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
