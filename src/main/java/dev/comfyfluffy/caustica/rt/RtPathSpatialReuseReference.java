package dev.comfyfluffy.caustica.rt;

/**
 * Admission and Jacobian contract for the first path spatial-reuse experiment.
 *
 * <p>This is intentionally reference-only. A GPU shift must store a reconnection vertex (and the
 * associated geometric/PDF terms) before it can consume this contract.</p>
 */
final class RtPathSpatialReuseReference {
    static final double NORMAL_COSINE_THRESHOLD = 0.85;
    static final double RELATIVE_DEPTH_THRESHOLD = 0.10;
    static final double MAX_FOOTPRINT_RATIO = 4.0;
    static final int RECONNECTION_HIT_DEPTH = 1;
    static final int RECONNECTION_VALID = 1;
    static final int RECONNECTION_EVENT_SHIFT = 5;
    static final int RECONNECTION_EVENT_MASK = 0x7 << RECONNECTION_EVENT_SHIFT;
    static final int MATERIAL_OPAQUE = 0;
    static final int MATERIAL_PARTICLE = 2;
    static final int RECEIVER_MATERIAL_KEY_MASK = 0x003F_FFFF;
    static final int MAPPING_KIND_MASK = 0xF;
    static final int MAPPING_RESERVED_MASK = 0xFFFF_FFF0;

    enum ReconnectionEvent {
        NONE(0),
        DIFFUSE(1),
        GLOSSY(2),
        DELTA(3),
        TRANSMISSION(4);

        private final int code;

        ReconnectionEvent(int code) {
            this.code = code;
        }

        int code() {
            return code;
        }

        static ReconnectionEvent fromCode(int code) {
            for (ReconnectionEvent event : values()) {
                if (event.code == code) {
                    return event;
                }
            }
            throw new IllegalArgumentException("unknown reconnection event: " + code);
        }

        boolean continuous() {
            return this == DIFFUSE || this == GLOSSY;
        }
    }

    /**
     * Compact control stored in {@code PathReservoir.reconnectionThroughput.w}. The first
     * persistent spatial contract is deliberately one-hop: a diffuse reconnection can be replayed
     * against its original identity source, but cannot itself be used as another spatial source
     * until mapping/Jacobian composition is defined.
     */
    enum MappingKind {
        IDENTITY(0),
        DIFFUSE_RECONNECTION(1);

        private final int code;

        MappingKind(int code) {
            this.code = code;
        }

        int code() {
            return code;
        }

        static MappingKind fromCode(int code) {
            for (MappingKind kind : values()) {
                if (kind.code == code) {
                    return kind;
                }
            }
            throw new IllegalArgumentException("unknown spatial mapping kind: " + code);
        }
    }

    enum Decision {
        ACCEPTED,
        MATERIAL_MISMATCH,
        SURFACE_NORMAL_MISMATCH,
        RELATIVE_DEPTH_MISMATCH,
        DEPTH_MISMATCH,
        TOPOLOGY_MISMATCH,
        TRANSPORT_MISMATCH,
        FOOTPRINT_MISMATCH
    }

    static int packMappingControl(MappingKind kind) {
        if (kind == null) {
            throw new IllegalArgumentException("spatial mapping kind must not be null");
        }
        return kind.code();
    }

    static boolean mappingControlValid(int control) {
        int kind = control & MAPPING_KIND_MASK;
        return (control & MAPPING_RESERVED_MASK) == 0
                && (kind == MappingKind.IDENTITY.code()
                        || kind == MappingKind.DIFFUSE_RECONNECTION.code());
    }

    static MappingKind mappingKind(int control) {
        if (!mappingControlValid(control)) {
            throw new IllegalArgumentException("invalid spatial mapping control");
        }
        return MappingKind.fromCode(control & MAPPING_KIND_MASK);
    }

    static boolean genericIdentityReplayEligible(int control) {
        return mappingControlValid(control) && mappingKind(control) == MappingKind.IDENTITY;
    }

    static boolean oneHopSpatialSourceEligible(int control) {
        return genericIdentityReplayEligible(control);
    }

    /**
     * Reference-only compatibility variants. LIMITED_TOPOLOGY keeps every existing safety
     * condition and relaxes only exact path-topology identity; it is intentionally not wired into
     * the GPU estimator.
     */
    enum AdmissionPolicy {
        STRICT,
        LIMITED_TOPOLOGY
    }

    record PolicyComparison(Decision strictDecision, Decision limitedDecision) {
        boolean topologyRescued() {
            return strictDecision == Decision.TOPOLOGY_MISMATCH
                    && limitedDecision == Decision.ACCEPTED;
        }
    }

    /** Deterministic CPU-only totals for comparing policy candidates before any GPU experiment. */
    static final class PolicyComparisonCounters {
        private long samples;
        private long strictAccepted;
        private long limitedAccepted;
        private long topologyRescued;

        void add(Surface receiver, Surface source) {
            add(comparePolicies(receiver, source));
        }

        void add(PolicyComparison comparison) {
            samples++;
            if (comparison.strictDecision() == Decision.ACCEPTED) {
                strictAccepted++;
            }
            if (comparison.limitedDecision() == Decision.ACCEPTED) {
                limitedAccepted++;
            }
            if (comparison.topologyRescued()) {
                topologyRescued++;
            }
        }

        long samples() {
            return samples;
        }

        long strictAccepted() {
            return strictAccepted;
        }

        long limitedAccepted() {
            return limitedAccepted;
        }

        long topologyRescued() {
            return topologyRescued;
        }

        double strictAcceptanceRatio() {
            return samples == 0L ? Double.NaN : (double) strictAccepted / samples;
        }

        double limitedAcceptanceRatio() {
            return samples == 0L ? Double.NaN : (double) limitedAccepted / samples;
        }

        double topologyRescueRatio() {
            return samples == 0L ? Double.NaN : (double) topologyRescued / samples;
        }
    }

    enum DiagnosticCategory {
        RECEIVER_EMPTY,
        ACCEPTED_RECONNECTION,
        FOOTPRINT_REJECT,
        DEPTH_REJECT,
        TOPOLOGY_REJECT,
        TRANSPORT_REJECT,
        COMPATIBLE_NO_RECONNECTION,
        COMPATIBLE_NEIGHBOR_EMPTY,
        SURFACE_REJECT
    }

    /**
     * Streaming category totals for the view-17 mask. Counts are intentionally independent of
     * resolution and frame size so a later GPU readback can compare ratios directly.
     */
    static final class DiagnosticCounters {
        private final long[] counts = new long[DiagnosticCategory.values().length];

        void add(DiagnosticCategory category) {
            counts[category.ordinal()]++;
        }

        long count(DiagnosticCategory category) {
            return counts[category.ordinal()];
        }

        long total() {
            long total = 0L;
            for (long count : counts) {
                total = Math.addExact(total, count);
            }
            return total;
        }

        double ratio(DiagnosticCategory category) {
            long total = total();
            return total == 0L ? Double.NaN : (double) count(category) / total;
        }
    }

    /**
     * Numerically stable paired moments for receiver/source luminance measurements. The sample
     * covariance and Pearson correlation are undefined until two finite pairs are observed; this
     * explicit contract prevents a zero-variance batch from looking like perfect reuse.
     */
    static final class PairMoments {
        private long count;
        private double meanX;
        private double meanY;
        private double m2X;
        private double m2Y;
        private double coMoment;

        void add(double x, double y) {
            if (!Double.isFinite(x) || !Double.isFinite(y)) {
                throw new IllegalArgumentException("paired moments require finite samples");
            }
            long nextCount = Math.addExact(count, 1L);
            double deltaX = x - meanX;
            double deltaY = y - meanY;
            meanX += deltaX / nextCount;
            meanY += deltaY / nextCount;
            m2X += deltaX * (x - meanX);
            m2Y += deltaY * (y - meanY);
            coMoment += deltaX * (y - meanY);
            count = nextCount;
        }

        long count() {
            return count;
        }

        double meanX() {
            return meanX;
        }

        double meanY() {
            return meanY;
        }

        double varianceX() {
            return count > 1L ? m2X / (count - 1L) : Double.NaN;
        }

        double varianceY() {
            return count > 1L ? m2Y / (count - 1L) : Double.NaN;
        }

        double covariance() {
            return count > 1L ? coMoment / (count - 1L) : Double.NaN;
        }

        double correlation() {
            if (count < 2L || m2X <= 0.0 || m2Y <= 0.0) {
                return Double.NaN;
            }
            return coMoment / Math.sqrt(m2X * m2Y);
        }
    }

    record Surface(long materialKey, double normalCosine, double relativeDepth,
                   int depth, long topologyKey, int transportClass, double footprint) {
        Surface {
            if (!Double.isFinite(normalCosine) || normalCosine < -1.0 || normalCosine > 1.0
                    || !Double.isFinite(relativeDepth) || relativeDepth < 0.0
                    || depth < 0 || topologyKey < 0L || transportClass < 0
                    || !Double.isFinite(footprint) || footprint <= 0.0) {
                throw new IllegalArgumentException("invalid spatial compatibility surface");
            }
        }
    }

    /**
     * Distances are |x_(i+1)-x_i| and |x_(i+1)-y_i|. Cosines are measured at the shared next
     * vertex against the source and receiver directions respectively.
     */
    record ReconnectionGeometry(double sourceDistance, double receiverDistance,
                                double sourceCosine, double receiverCosine,
                                double sourceDirectionalPdf, double receiverDirectionalPdf) {
        ReconnectionGeometry {
            if (!positiveFinite(sourceDistance) || !positiveFinite(receiverDistance)
                    || !positiveFinite(sourceCosine) || !positiveFinite(receiverCosine)
                    || !positiveFinite(sourceDirectionalPdf)
                    || !positiveFinite(receiverDirectionalPdf)) {
                throw new IllegalArgumentException("invalid reconnection geometry/PDF");
            }
        }

        double solidAngleJacobian() {
            return (receiverCosine * sourceDistance * sourceDistance)
                    / (sourceCosine * receiverDistance * receiverDistance);
        }

        /**
         * Primary-sample-space form: the solid-angle reconnection Jacobian multiplied by the
         * receiver/source directional-PDF ratio. Random replay itself contributes unit Jacobian.
         */
        double primarySampleJacobian() {
            return solidAngleJacobian() * receiverDirectionalPdf / sourceDirectionalPdf;
        }
    }

    /** Receiver/source density terms used by the diffuse reconnection contract. */
    record DiffuseShiftDensity(double currentDirectionalDensity,
                               double shiftedDirectionalDensity,
                               double sourceDirectionalDensity,
                               double currentSourcePdf, double sourcePdf) {
        DiffuseShiftDensity {
            if (!positiveFinite(currentDirectionalDensity)
                    || !positiveFinite(shiftedDirectionalDensity)
                    || !positiveFinite(sourceDirectionalDensity)
                    || !positiveFinite(currentSourcePdf)
                    || !positiveFinite(sourcePdf)) {
                throw new IllegalArgumentException("invalid diffuse shifted density terms");
            }
        }

        double currentTechniqueMass() {
            return finitePositiveRatio(currentSourcePdf, currentDirectionalDensity,
                    "current technique mass");
        }

        double sourceTechniqueMass() {
            return finitePositiveRatio(sourcePdf, sourceDirectionalDensity,
                    "source technique mass");
        }

        double receiverDirectionalPdf() {
            double receiverPdf = currentTechniqueMass() * shiftedDirectionalDensity;
            if (!positiveFinite(receiverPdf)) {
                throw new IllegalArgumentException("receiver directional PDF must be finite and positive");
            }
            return receiverPdf;
        }

        double primarySampleJacobian(ReconnectionGeometry geometry) {
            if (geometry == null) {
                throw new IllegalArgumentException("reconnection geometry is required");
            }
            double pssJacobian = geometry.solidAngleJacobian()
                    * receiverDirectionalPdf() / sourcePdf;
            if (!positiveFinite(pssJacobian)) {
                throw new IllegalArgumentException("PSS Jacobian must be finite and positive");
            }
            return pssJacobian;
        }
    }

    /**
     * The first spatial GRIS merge weight, kept as a small CPU-only contract until the GPU can
     * evaluate the receiver-side shifted target density and the complete reconnection mapping.
     *
     * <p>The PSS Jacobian belongs in this weight. It must not be applied again to the shifted
     * radiance or to the final reservoir resolve:</p>
     *
     * <pre>
     * w_spatial = shiftedTargetDensity * sourceFinalWeight
     *              * min(sourceEffectiveCount, maxSourceCount)
     *              * primarySampleJacobian
     * </pre>
     */
    record SpatialGrisWeight(double shiftedTargetDensity, double sourceFinalWeight,
                             double sourceEffectiveCount, double maxSourceCount,
                             double primarySampleJacobian) {
        SpatialGrisWeight {
            if (!nonNegativeFinite(shiftedTargetDensity)
                    || !nonNegativeFinite(sourceFinalWeight)
                    || !nonNegativeFinite(sourceEffectiveCount)
                    || !positiveFinite(maxSourceCount)
                    || !positiveFinite(primarySampleJacobian)) {
                throw new IllegalArgumentException("invalid spatial GRIS weight terms");
            }
        }

        double clampedSourceCount() {
            return Math.min(sourceEffectiveCount, maxSourceCount);
        }

        double mergeWeight() {
            double sourceCount = clampedSourceCount();
            if (sourceCount == 0.0 || shiftedTargetDensity == 0.0 || sourceFinalWeight == 0.0) {
                return 0.0;
            }
            double weight = shiftedTargetDensity * sourceFinalWeight;
            weight *= sourceCount;
            weight *= primarySampleJacobian;
            if (!Double.isFinite(weight) || weight < 0.0) {
                throw new IllegalArgumentException("spatial GRIS weight overflow");
            }
            return weight;
        }

        double selectionProbability(double currentWeightSum) {
            if (!nonNegativeFinite(currentWeightSum)) {
                throw new IllegalArgumentException("invalid current reservoir weight sum");
            }
            double weight = mergeWeight();
            if (weight == 0.0) {
                return 0.0;
            }
            double nextWeightSum = Math.min(currentWeightSum + weight, 1.0e30);
            double probability = weight / nextWeightSum;
            if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException("invalid spatial GRIS selection probability");
            }
            return probability;
        }

        boolean selectsSource(double currentWeightSum, double randomUnit) {
            if (!Double.isFinite(randomUnit) || randomUnit < 0.0 || randomUnit >= 1.0) {
                throw new IllegalArgumentException("invalid spatial GRIS selection random value");
            }
            return randomUnit < selectionProbability(currentWeightSum);
        }

        SpatialScratchMerge scratchMerge(double currentWeightSum, double currentEffectiveCount,
                                         double currentTarget, double randomUnit) {
            if (!nonNegativeFinite(currentEffectiveCount) || !positiveFinite(currentTarget)) {
                throw new IllegalArgumentException("invalid current scratch reservoir");
            }
            boolean sourceSelected = selectsSource(currentWeightSum, randomUnit);
            double nextWeightSum = Math.min(currentWeightSum + mergeWeight(), 1.0e30);
            double nextEffectiveCount = Math.min(
                    currentEffectiveCount + clampedSourceCount(), 16777216.0);
            double selectedTarget = sourceSelected ? shiftedTargetDensity : currentTarget;
            double finalWeight = nextWeightSum / (nextEffectiveCount * selectedTarget);
            return new SpatialScratchMerge(
                    nextWeightSum, nextEffectiveCount, finalWeight, sourceSelected);
        }
    }

    record SpatialScratchMerge(double weightSum, double effectiveCount,
                               double finalWeight, boolean sourceSelected) {
        SpatialScratchMerge {
            if (!positiveFinite(weightSum) || !positiveFinite(effectiveCount)
                    || !positiveFinite(finalWeight)) {
                throw new IllegalArgumentException("invalid spatial scratch merge result");
            }
        }
    }

    record Rgb(double r, double g, double b) {
        Rgb {
            if (!nonNegativeFinite(r) || !nonNegativeFinite(g) || !nonNegativeFinite(b)) {
                throw new IllegalArgumentException("RGB terms must be finite and non-negative");
            }
        }

        double luminance() {
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    /**
     * Camera-relative storage for an exact retained source queue root. The capture frame's terrain
     * rebase is deliberately not part of persistent identity: reconstructing in a later frame adds
     * the current camera offset and subtracts the camera translation since capture.
     */
    record PersistentSourceRootOrigin(double cameraRelativeX, double cameraRelativeY,
                                      double cameraRelativeZ) {
        PersistentSourceRootOrigin {
            if (!Double.isFinite(cameraRelativeX) || !Double.isFinite(cameraRelativeY)
                    || !Double.isFinite(cameraRelativeZ)) {
                throw new IllegalArgumentException("persistent source-root origin must be finite");
            }
        }

        static PersistentSourceRootOrigin capture(
                double rootX, double rootY, double rootZ,
                double cameraOffsetX, double cameraOffsetY, double cameraOffsetZ) {
            return new PersistentSourceRootOrigin(
                    rootX - cameraOffsetX,
                    rootY - cameraOffsetY,
                    rootZ - cameraOffsetZ);
        }

        PersistentSourceRootOrigin reconstruct(
                double currentCameraOffsetX, double currentCameraOffsetY,
                double currentCameraOffsetZ, double cameraDeltaX,
                double cameraDeltaY, double cameraDeltaZ) {
            return new PersistentSourceRootOrigin(
                    cameraRelativeX + currentCameraOffsetX - cameraDeltaX,
                    cameraRelativeY + currentCameraOffsetY - cameraDeltaY,
                    cameraRelativeZ + currentCameraOffsetZ - cameraDeltaZ);
        }
    }

    /**
     * Shader-independent receiver-aware replay boundary for an ABI-10 one-hop diffuse mapping.
     * The source path is replayed from its original seeds; this record then applies the stored
     * receiver factor and freshly traced transmittance. Generic identity replay must never consume
     * this descriptor.
     */
    record DiffuseMappingReplay(int replayVersion, int mappingControl,
                                ReconnectionEvent replayEvent,
                                double receiverDirectionalPdf,
                                double primarySampleJacobian,
                                Rgb sourceRadiance,
                                Rgb sourceThroughput,
                                Rgb receiverThroughput,
                                Rgb transmittance) {
        DiffuseMappingReplay {
            if (replayVersion != RtPathReplayReference.REPLAY_VERSION) {
                throw new IllegalArgumentException("spatial mapping replay ABI mismatch");
            }
            if (!mappingControlValid(mappingControl)
                    || mappingKind(mappingControl) != MappingKind.DIFFUSE_RECONNECTION) {
                throw new IllegalArgumentException("diffuse mapping descriptor is required");
            }
            if (replayEvent != ReconnectionEvent.DIFFUSE) {
                throw new IllegalArgumentException("diffuse mapping replay event mismatch");
            }
            if (!positiveFinite(receiverDirectionalPdf)
                    || !positiveFinite(primarySampleJacobian)
                    || sourceRadiance == null || sourceThroughput == null
                    || receiverThroughput == null || transmittance == null
                    || transmittance.r() > 1.0 || transmittance.g() > 1.0
                    || transmittance.b() > 1.0) {
                throw new IllegalArgumentException("invalid receiver-aware replay terms");
            }
            // Validate the spectral support before the record can be used.
            shiftedDiffuseRadiance(sourceRadiance.r(), sourceThroughput.r(),
                    receiverThroughput.r(), transmittance.r());
            shiftedDiffuseRadiance(sourceRadiance.g(), sourceThroughput.g(),
                    receiverThroughput.g(), transmittance.g());
            shiftedDiffuseRadiance(sourceRadiance.b(), sourceThroughput.b(),
                    receiverThroughput.b(), transmittance.b());
        }

        Rgb shiftedRadiance() {
            return new Rgb(
                    shiftedDiffuseRadiance(sourceRadiance.r(), sourceThroughput.r(),
                            receiverThroughput.r(), transmittance.r()),
                    shiftedDiffuseRadiance(sourceRadiance.g(), sourceThroughput.g(),
                            receiverThroughput.g(), transmittance.g()),
                    shiftedDiffuseRadiance(sourceRadiance.b(), sourceThroughput.b(),
                            receiverThroughput.b(), transmittance.b()));
        }

        double shiftedTarget() {
            return shiftedRadiance().luminance();
        }

        boolean matchesStoredShift(Rgb storedShift) {
            if (storedShift == null) {
                return false;
            }
            Rgb shifted = shiftedRadiance();
            return replayFloatMatches(shifted.r(), storedShift.r())
                    && replayFloatMatches(shifted.g(), storedShift.g())
                    && replayFloatMatches(shifted.b(), storedShift.b());
        }
    }

    /**
     * Reference boundary for carrying a one-hop diffuse mapping across a frame boundary.
     *
     * <p>The original source path remains the canonical replay root. A later receiver remap must
     * therefore prove both the receiver reprojection and the stability of that source queue root,
     * then replay the source exactly. The old receiver's Jacobian is validated as stored metadata
     * but is never composed with the new mapping: the source-to-current-receiver Jacobian is
     * recomputed directly.</p>
     */
    record PersistentDiffuseRemap(int replayVersion, int mappingControl,
                                  boolean receiverReprojectionValid,
                                  boolean sourceQueueRootStable,
                                  boolean sourceReplayExact,
                                  double previousPrimarySampleJacobian,
                                  ReconnectionGeometry currentGeometry,
                                  DiffuseShiftDensity currentDensity,
                                  Rgb sourceRadiance,
                                  Rgb sourceThroughput,
                                  Rgb currentReceiverThroughput,
                                  Rgb currentTransmittance) {
        PersistentDiffuseRemap {
            if (replayVersion != RtPathReplayReference.REPLAY_VERSION
                    || !mappingControlValid(mappingControl)
                    || mappingKind(mappingControl) != MappingKind.DIFFUSE_RECONNECTION) {
                throw new IllegalArgumentException("persistent diffuse mapping descriptor mismatch");
            }
            if (!receiverReprojectionValid || !sourceQueueRootStable || !sourceReplayExact) {
                throw new IllegalArgumentException("persistent diffuse mapping replay is not stable");
            }
            if (!positiveFinite(previousPrimarySampleJacobian)
                    || currentGeometry == null || currentDensity == null
                    || sourceRadiance == null || sourceThroughput == null
                    || currentReceiverThroughput == null || currentTransmittance == null) {
                throw new IllegalArgumentException("invalid persistent diffuse remap terms");
            }
        }

        double currentPrimarySampleJacobian() {
            return currentDensity.primarySampleJacobian(currentGeometry);
        }

        DiffuseMappingReplay currentReplay() {
            return new DiffuseMappingReplay(
                    replayVersion, mappingControl, ReconnectionEvent.DIFFUSE,
                    currentDensity.receiverDirectionalPdf(),
                    currentPrimarySampleJacobian(),
                    sourceRadiance, sourceThroughput,
                    currentReceiverThroughput, currentTransmittance);
        }
    }

    /**
     * Re-evaluates one color channel of a diffuse reconnection. The stored source endpoint already
     * contains the source first-edge throughput, so the shift removes that exact stored factor,
     * applies the receiver's exact stored diffuse factor, and finally applies newly traced
     * transmittance.
     */
    static double shiftedDiffuseRadiance(
            double sourceRadiance, double sourceThroughput, double receiverThroughput,
            double transmittance) {
        if (!nonNegativeFinite(sourceRadiance) || !nonNegativeFinite(sourceThroughput)
                || !nonNegativeFinite(receiverThroughput)
                || !nonNegativeFinite(transmittance) || transmittance > 1.0
                || (sourceRadiance > 0.0 && sourceThroughput <= 0.0)) {
            throw new IllegalArgumentException("invalid diffuse reconnection radiance terms");
        }
        if (sourceRadiance == 0.0 || receiverThroughput == 0.0 || transmittance == 0.0) {
            return 0.0;
        }
        return sourceRadiance * receiverThroughput / sourceThroughput * transmittance;
    }

    static boolean replayFloatMatches(double expected, double actual) {
        double tolerance = Math.max(1.0e-5,
                Math.max(Math.abs(expected), Math.abs(actual)) * 1.0e-4);
        return Double.isFinite(expected) && Double.isFinite(actual)
                && Math.abs(expected - actual) <= tolerance;
    }

    /** Matches the shader's absolute-cosine diffuse density used for a shifted edge. */
    static double diffuseShiftDirectionalDensity(double cosine) {
        if (!Double.isFinite(cosine) || cosine < -1.0 || cosine > 1.0) {
            throw new IllegalArgumentException("diffuse shift cosine must be finite and in [-1, 1]");
        }
        return Math.abs(cosine) / Math.PI;
    }

    /**
     * The primary pass consumes water and dielectric interfaces before the canonical path starts.
     * Only opaque and particle guides therefore describe the same first vertex as a stored diffuse
     * reconnection event.
     */
    static boolean supportsDiffuseShiftReceiver(int material) {
        return material == MATERIAL_OPAQUE || material == MATERIAL_PARTICLE;
    }

    /** Packs a 2-bit model and 22-bit stable registry key into an exactly representable R32F integer. */
    static int packReceiverMaterialIdentity(int material, int materialKey) {
        if ((material & ~3) != 0 || (materialKey & ~RECEIVER_MATERIAL_KEY_MASK) != 0) {
            throw new IllegalArgumentException("receiver material identity exceeds 24 bits");
        }
        return (materialKey << 2) | material;
    }

    static int receiverMaterialModel(int identity) {
        return identity & 3;
    }

    static int receiverMaterialKey(int identity) {
        return identity >>> 2;
    }

    static Decision admit(Surface receiver, Surface source) {
        return admit(receiver, source, AdmissionPolicy.STRICT);
    }

    static Decision admit(Surface receiver, Surface source, AdmissionPolicy policy) {
        if (policy == null) {
            throw new IllegalArgumentException("spatial admission policy is required");
        }
        if (receiver.materialKey() != source.materialKey()) {
            return Decision.MATERIAL_MISMATCH;
        }
        if (receiver.normalCosine() < NORMAL_COSINE_THRESHOLD) {
            return Decision.SURFACE_NORMAL_MISMATCH;
        }
        if (receiver.relativeDepth() > RELATIVE_DEPTH_THRESHOLD) {
            return Decision.RELATIVE_DEPTH_MISMATCH;
        }
        if (receiver.depth() != source.depth()) {
            return Decision.DEPTH_MISMATCH;
        }
        if (policy == AdmissionPolicy.STRICT
                && receiver.topologyKey() != source.topologyKey()) {
            return Decision.TOPOLOGY_MISMATCH;
        }
        if (receiver.transportClass() != source.transportClass()) {
            return Decision.TRANSPORT_MISMATCH;
        }
        double footprintRatio = Math.max(receiver.footprint(), source.footprint())
                / Math.min(receiver.footprint(), source.footprint());
        return footprintRatio <= MAX_FOOTPRINT_RATIO
                ? Decision.ACCEPTED : Decision.FOOTPRINT_MISMATCH;
    }

    static PolicyComparison comparePolicies(Surface receiver, Surface source) {
        return new PolicyComparison(
                admit(receiver, source, AdmissionPolicy.STRICT),
                admit(receiver, source, AdmissionPolicy.LIMITED_TOPOLOGY));
    }

    static boolean captureReconnectionAtDepth(int hitDepth) {
        return hitDepth == RECONNECTION_HIT_DEPTH;
    }

    /** A second-hit shift can consume only an endpoint reached after the first selected edge. */
    static boolean reconnectionEndpointEligible(int hitDepth) {
        return hitDepth >= RECONNECTION_HIT_DEPTH;
    }

    static int packReconnectionMetadata(int depth, ReconnectionEvent event, boolean valid) {
        if (depth < 0) {
            throw new IllegalArgumentException("reconnection depth must be non-negative");
        }
        if (event == null) {
            throw new IllegalArgumentException("reconnection event must be non-null");
        }
        return (valid ? RECONNECTION_VALID : 0)
                | ((Math.min(depth, 15) & 0xF) << 1)
                | ((event.code() & 0x7) << RECONNECTION_EVENT_SHIFT);
    }

    static int packReconnectionMetadata(int depth, boolean valid) {
        return packReconnectionMetadata(depth, ReconnectionEvent.NONE, valid);
    }

    static boolean reconnectionValid(int packedMetadata) {
        return (packedMetadata & RECONNECTION_VALID) != 0;
    }

    static int reconnectionDepth(int packedMetadata) {
        return (packedMetadata >>> 1) & 0xF;
    }

    static ReconnectionEvent reconnectionEvent(int packedMetadata) {
        return ReconnectionEvent.fromCode(
                (packedMetadata & RECONNECTION_EVENT_MASK) >>> RECONNECTION_EVENT_SHIFT);
    }

    private static boolean positiveFinite(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static boolean nonNegativeFinite(double value) {
        return Double.isFinite(value) && value >= 0.0;
    }

    private static double finitePositiveRatio(double numerator, double denominator, String label) {
        double ratio = numerator / denominator;
        if (!positiveFinite(ratio)) {
            throw new IllegalArgumentException(label + " must be finite and positive");
        }
        return ratio;
    }

    private RtPathSpatialReuseReference() {
    }
}
