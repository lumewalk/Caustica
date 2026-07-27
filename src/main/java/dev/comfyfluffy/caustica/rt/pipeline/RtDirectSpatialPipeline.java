package dev.comfyfluffy.caustica.rt.pipeline;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.gen.WorldPushConstantsData;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
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

/** Immutable input/output descriptor pairs for race-free direct-reservoir spatial reuse. */
public final class RtDirectSpatialPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";
    private static final int SLOT_COUNT = 2;
    private static final int IMAGE_COUNT = 4;
    private static final int BUFFER_COUNT = 2;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long pipeline;
    private boolean destroyed;

    private RtDirectSpatialPipeline(RtContext ctx, long descriptorSetLayout, long descriptorPool,
                                    long[] descriptorSets, long pipelineLayout, long pipeline) {
        this.ctx = ctx;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSets = descriptorSets;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static RtDirectSpatialPipeline create(
            RtContext ctx,
            long receiverPositionMaterialView,
            long receiverNormalRoughnessView,
            long receiverAlbedoSssView,
            long debugColorView,
            RtBuffer[] reservoirSlots) {
        if (reservoirSlots.length != SLOT_COUNT) {
            throw new IllegalArgumentException("Spatial reuse requires exactly two reservoir slots");
        }
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(IMAGE_COUNT + BUFFER_COUNT, stack);
            for (int binding = 0; binding < IMAGE_COUNT; binding++) {
                bindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            for (int buffer = 0; buffer < BUFFER_COUNT; buffer++) {
                bindings.get(IMAGE_COUNT + buffer).binding(IMAGE_COUNT + buffer)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo dslInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateDescriptorSetLayout(vk, dslInfo, null, handle),
                    "vkCreateDescriptorSetLayout(direct spatial)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, descriptorSetLayout,
                    "direct spatial descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(SLOT_COUNT * IMAGE_COUNT);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(SLOT_COUNT * BUFFER_COUNT);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(SLOT_COUNT).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(direct spatial)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptorPool,
                    "direct spatial descriptor pool");

            LongBuffer layouts = stack.mallocLong(SLOT_COUNT);
            for (int slot = 0; slot < SLOT_COUNT; slot++) layouts.put(slot, descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool).pSetLayouts(layouts);
            LongBuffer setsBuffer = stack.mallocLong(SLOT_COUNT);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setsBuffer),
                    "vkAllocateDescriptorSets(direct spatial)");
            long[] descriptorSets = new long[SLOT_COUNT];
            setsBuffer.get(descriptorSets);

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(WorldPushConstantsData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, layoutInfo, null, handle),
                    "vkCreatePipelineLayout(direct spatial)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout,
                    "direct spatial pipeline layout");

            long module = loadModule(vk, stack, "direct_spatial.comp.spv");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, pipelineInfo, null, handle),
                    "vkCreateComputePipelines(direct spatial)");
            long pipeline = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline,
                    "direct spatial compute pipeline");
            VK10.vkDestroyShaderModule(vk, module, null);

            bindResources(vk, stack, descriptorSets,
                    receiverPositionMaterialView, receiverNormalRoughnessView,
                    receiverAlbedoSssView, debugColorView, reservoirSlots);
            return new RtDirectSpatialPipeline(ctx, descriptorSetLayout, descriptorPool,
                    descriptorSets, pipelineLayout, pipeline);
        }
    }

    public void dispatch(VkCommandBuffer cmd, int inputSlot, int width, int height,
                         ByteBuffer pushConstants) {
        if (inputSlot < 0 || inputSlot >= SLOT_COUNT) {
            throw new IllegalArgumentException("Reservoir input slot out of range: " + inputSlot);
        }
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "direct spatial reuse")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                    stack.longs(descriptorSets[inputSlot]), null);
            VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    pushConstants);
            VK10.vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        }
    }

    public void destroy() {
        if (destroyed) return;
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static void bindResources(
            VkDevice vk, MemoryStack stack, long[] sets,
            long receiverPositionMaterialView, long receiverNormalRoughnessView,
            long receiverAlbedoSssView, long debugColorView, RtBuffer[] reservoirSlots) {
        long[] imageViews = {
                receiverPositionMaterialView,
                receiverNormalRoughnessView,
                receiverAlbedoSssView,
                debugColorView
        };
        int writesPerSet = IMAGE_COUNT + BUFFER_COUNT;
        VkDescriptorImageInfo.Buffer imageInfos =
                VkDescriptorImageInfo.calloc(SLOT_COUNT * IMAGE_COUNT, stack);
        VkDescriptorBufferInfo.Buffer bufferInfos =
                VkDescriptorBufferInfo.calloc(SLOT_COUNT * BUFFER_COUNT, stack);
        VkWriteDescriptorSet.Buffer writes =
                VkWriteDescriptorSet.calloc(SLOT_COUNT * writesPerSet, stack);
        for (int inputSlot = 0; inputSlot < SLOT_COUNT; inputSlot++) {
            for (int binding = 0; binding < IMAGE_COUNT; binding++) {
                int imageIndex = inputSlot * IMAGE_COUNT + binding;
                int writeIndex = inputSlot * writesPerSet + binding;
                imageInfos.get(imageIndex).imageView(imageViews[binding])
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(writeIndex).sType$Default().dstSet(sets[inputSlot]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(imageInfos.address(imageIndex), 1));
            }
            int outputSlot = 1 - inputSlot;
            int inputBufferIndex = inputSlot * BUFFER_COUNT;
            int outputBufferIndex = inputBufferIndex + 1;
            bufferInfos.get(inputBufferIndex).buffer(reservoirSlots[inputSlot].handle)
                    .offset(0L).range(reservoirSlots[inputSlot].size);
            bufferInfos.get(outputBufferIndex).buffer(reservoirSlots[outputSlot].handle)
                    .offset(0L).range(reservoirSlots[outputSlot].size);
            for (int buffer = 0; buffer < BUFFER_COUNT; buffer++) {
                int bufferIndex = inputSlot * BUFFER_COUNT + buffer;
                int binding = IMAGE_COUNT + buffer;
                int writeIndex = inputSlot * writesPerSet + binding;
                writes.get(writeIndex).sType$Default().dstSet(sets[inputSlot]).dstBinding(binding)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.create(bufferInfos.address(bufferIndex), 1));
            }
        }
        VK10.vkUpdateDescriptorSets(vk, writes, null);
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtDirectSpatialPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
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
