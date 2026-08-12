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
    static final double MAX_SPATIAL_SOURCE_COUNT = 8.0;
    static final double MAX_STORED_WEIGHT_SUM = 1.0e30;
    static final double MAX_STORED_EFFECTIVE_COUNT = 16_777_216.0;
    static final int RECONNECTION_HIT_DEPTH = 1;
    static final int RECONNECTION_VALID = 1;
    static final int RECONNECTION_EVENT_SHIFT = 5;
    static final int RECONNECTION_EVENT_MASK = 0x7 << RECONNECTION_EVENT_SHIFT;
    static final int MATERIAL_OPAQUE = 0;
    static final int MATERIAL_PARTICLE = 2;
    static final int RECEIVER_MATERIAL_KEY_MASK = 0x003F_FFFF;
    static final int MAPPING_KIND_MASK = 0xF;
    static final int MAPPING_RESERVED_MASK = 0xFFFF_FFF0;
    static final int BRANCH_CANDIDATE_TAG_VALID_BIT = 0x8000_0000;
    static final int BRANCH_CANDIDATE_TAG_MAPPING_SHIFT = 24;
    static final int BRANCH_CANDIDATE_TAG_GENERATION_MASK = 0x00FF_FFFF;
    static final int BRANCH_BERNOULLI_ENTRY_MULTIPLIER = 747_796_405;
    static final int BRANCH_BERNOULLI_FRAME_MULTIPLIER = (int) 2_891_336_453L;
    static final int BRANCH_BERNOULLI_SALT = (int) 0x94D0_49BBL;
    static final int BRANCH_WINNER_BERNOULLI_TAG_FRAME_MULTIPLIER = 277_803_737;
    static final int BRANCH_WINNER_BERNOULLI_PRIORITY_MULTIPLIER = 1_597_334_677;
    static final int BRANCH_WINNER_BERNOULLI_SALT = (int) 0xD1B5_4A35L;
    static final int BRANCH_RECEIVER_OWNER_INDEX_BITS = 14;
    static final int BRANCH_RECEIVER_OWNER_INDEX_MASK =
            (1 << BRANCH_RECEIVER_OWNER_INDEX_BITS) - 1;

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

        /**
         * Uncapped relative-weight selection ratio for the counter-only overflow A/B. This method
         * intentionally does not change the stored weight-sum cap or the existing scratch merge.
         */
        double stableSelectionProbability(double currentWeightSum) {
            if (!nonNegativeFinite(currentWeightSum)) {
                throw new IllegalArgumentException("invalid current reservoir weight sum");
            }
            double weight = mergeWeight();
            if (weight == 0.0) {
                return 0.0;
            }
            double probability;
            if (currentWeightSum <= weight) {
                probability = 1.0 / (1.0 + currentWeightSum / weight);
            } else {
                double ratio = weight / currentWeightSum;
                probability = ratio / (1.0 + ratio);
            }
            if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException("invalid stable spatial selection probability");
            }
            return probability;
        }

        boolean stableSelectsSource(double currentWeightSum, double randomUnit) {
            if (!Double.isFinite(randomUnit) || randomUnit < 0.0 || randomUnit >= 1.0) {
                throw new IllegalArgumentException(
                        "invalid stable spatial selection random value");
            }
            return randomUnit < stableSelectionProbability(currentWeightSum);
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

        /** Counter-only authority for the post-selection arithmetic gate. */
        SpatialShadowMerge stableShadowMerge(double currentWeightSum,
                                             double currentEffectiveCount,
                                             double currentTarget,
                                             double randomUnit) {
            if (!nonNegativeFinite(currentEffectiveCount)
                    || !nonNegativeFinite(currentTarget)) {
                throw new IllegalArgumentException("invalid current shadow reservoir");
            }
            boolean sourceSelected = stableSelectsSource(currentWeightSum, randomUnit);
            double nextWeightSum = Math.min(currentWeightSum + mergeWeight(), 1.0e30);
            double nextEffectiveCount = Math.min(
                    currentEffectiveCount + clampedSourceCount(), 16777216.0);
            double selectedTarget = sourceSelected ? shiftedTargetDensity : currentTarget;
            if (!nonNegativeFinite(nextWeightSum)
                    || !nonNegativeFinite(nextEffectiveCount)
                    || !nonNegativeFinite(selectedTarget)) {
                throw new IllegalArgumentException("invalid spatial shadow merge arithmetic");
            }
            if (nextWeightSum == 0.0) {
                return new SpatialShadowMerge(nextWeightSum, nextEffectiveCount,
                        selectedTarget, 0.0, sourceSelected, true);
            }
            if (nextEffectiveCount == 0.0 || selectedTarget == 0.0) {
                throw new IllegalArgumentException("invalid selected shadow sample");
            }
            double denominator = nextEffectiveCount * selectedTarget;
            double finalWeight = nextWeightSum / denominator;
            if (!positiveFinite(denominator) || !positiveFinite(finalWeight)) {
                throw new IllegalArgumentException("invalid spatial shadow final weight");
            }
            return new SpatialShadowMerge(nextWeightSum, nextEffectiveCount,
                    selectedTarget, finalWeight, sourceSelected, false);
        }

        /**
         * Register-only authority for selecting an opaque complete sample payload after the
         * arithmetic gate. Returning the original immutable payload object makes source-copy versus
         * current-retention observable without introducing a history or estimator mutation.
         */
        SpatialSampleScratch stableSampleScratch(double currentWeightSum,
                                                 double currentEffectiveCount,
                                                 SpatialSamplePayload currentSample,
                                                 SpatialSamplePayload shiftedSourceSample,
                                                 double randomUnit) {
            double currentTarget = currentSample == null ? 0.0 : currentSample.target();
            SpatialShadowMerge arithmetic = stableShadowMerge(currentWeightSum,
                    currentEffectiveCount, currentTarget, randomUnit);
            if (arithmetic.empty()) {
                return new SpatialSampleScratch(null, arithmetic);
            }
            SpatialSamplePayload selected = arithmetic.sourceSelected()
                    ? shiftedSourceSample : currentSample;
            if (selected == null || selected.target() != arithmetic.selectedTarget()) {
                throw new IllegalArgumentException("missing or mismatched spatial scratch sample");
            }
            return new SpatialSampleScratch(selected, arithmetic);
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

    record SpatialShadowMerge(double weightSum, double effectiveCount,
                              double selectedTarget, double finalWeight,
                              boolean sourceSelected, boolean empty) {
        SpatialShadowMerge {
            if (!nonNegativeFinite(weightSum) || !nonNegativeFinite(effectiveCount)
                    || !nonNegativeFinite(selectedTarget) || !nonNegativeFinite(finalWeight)) {
                throw new IllegalArgumentException("invalid spatial shadow merge result");
            }
            if (empty) {
                if (weightSum != 0.0 || finalWeight != 0.0) {
                    throw new IllegalArgumentException("invalid empty spatial shadow merge");
                }
            } else if (weightSum == 0.0 || effectiveCount == 0.0
                    || selectedTarget == 0.0 || finalWeight == 0.0) {
                throw new IllegalArgumentException("invalid ready spatial shadow merge");
            }
        }
    }

    /** Opaque stand-in for every copied PathReservoir sample lane in the CPU authority. */
    record SpatialSamplePayload(long metadataToken, double target) {
        SpatialSamplePayload {
            if (!positiveFinite(target)) {
                throw new IllegalArgumentException("spatial scratch sample target must be positive");
            }
        }
    }

    record SpatialSampleScratch(SpatialSamplePayload sample, SpatialShadowMerge arithmetic) {
        SpatialSampleScratch {
            if (arithmetic == null || (arithmetic.empty() != (sample == null))) {
                throw new IllegalArgumentException("invalid spatial sample scratch result");
            }
            if (sample != null && sample.target() != arithmetic.selectedTarget()) {
                throw new IllegalArgumentException("scratch sample target does not match arithmetic");
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

    /** Opaque material terms available at the deterministic primary receiver, before endpoint selection. */
    record DiffuseReceiverMaterial(Rgb diffuseAlbedo, Rgb f0, boolean exactSpecular) {
        DiffuseReceiverMaterial {
            if (diffuseAlbedo == null || f0 == null) {
                throw new IllegalArgumentException("diffuse receiver material terms are required");
            }
        }

        double specularTechniqueProbability() {
            if (exactSpecular && diffuseAlbedo.luminance() <= 1.0e-6) return 1.0;
            double raw = f0.luminance()
                    / (f0.luminance() + diffuseAlbedo.luminance() + 1.0e-4);
            return Math.max(0.1, Math.min(0.9, raw));
        }

        DiffuseReceiverGuide exactGuide() {
            return new DiffuseReceiverGuide(
                    diffuseAlbedo, 1.0 - specularTechniqueProbability());
        }
    }

    /**
     * Exact Pass-B outgoing edge geometry. The origin is captured after the production surface bias;
     * it must not be reconstructed from a separately traced Pass-A guide hit.
     */
    record DiffuseReceiverEdge(double originX, double originY, double originZ,
                               double vertexX, double vertexY, double vertexZ,
                               double normalX, double normalY, double normalZ) {
        DiffuseReceiverEdge {
            if (!Double.isFinite(originX) || !Double.isFinite(originY)
                    || !Double.isFinite(originZ) || !Double.isFinite(vertexX)
                    || !Double.isFinite(vertexY) || !Double.isFinite(vertexZ)
                    || !Double.isFinite(normalX) || !Double.isFinite(normalY)
                    || !Double.isFinite(normalZ)) {
                throw new IllegalArgumentException("receiver edge terms must be finite");
            }
        }

        double cosine() {
            double edgeX = vertexX - originX;
            double edgeY = vertexY - originY;
            double edgeZ = vertexZ - originZ;
            double edgeLength = Math.sqrt(edgeX * edgeX + edgeY * edgeY + edgeZ * edgeZ);
            double normalLength = Math.sqrt(
                    normalX * normalX + normalY * normalY + normalZ * normalZ);
            if (!(edgeLength > 0.0) || !(normalLength > 0.0)) return 0.0;
            return Math.abs((normalX * edgeX + normalY * edgeY + normalZ * edgeZ)
                    / (normalLength * edgeLength));
        }
    }

    /** First lane of the exact receiver sidecar: RGB diffuse albedo plus technique mass. */
    record DiffuseReceiverGuide(Rgb diffuseAlbedo, double techniqueMass) {
        DiffuseReceiverGuide {
            if (diffuseAlbedo == null || !positiveFinite(techniqueMass)
                    || techniqueMass > 1.0) {
                throw new IllegalArgumentException("invalid exact diffuse receiver guide");
            }
        }

        Rgb eventThroughput() {
            return new Rgb(
                    diffuseAlbedo.r() / techniqueMass,
                    diffuseAlbedo.g() / techniqueMass,
                    diffuseAlbedo.b() / techniqueMass);
        }

        double directionalPdf(double cosine) {
            return techniqueMass * diffuseShiftDirectionalDensity(cosine);
        }

        static DiffuseReceiverGuide particle(Rgb diffuseAlbedo) {
            return new DiffuseReceiverGuide(diffuseAlbedo, 1.0);
        }

        ReceiverGuideComparison compareStored(
                Rgb storedThroughput, double storedDirectionalPdf, double cosine) {
            if (storedThroughput == null || !positiveFinite(storedDirectionalPdf)) {
                throw new IllegalArgumentException("invalid stored diffuse edge");
            }
            double dominantAlbedo = diffuseAlbedo.r();
            double dominantThroughput = storedThroughput.r();
            if (diffuseAlbedo.g() > dominantAlbedo) {
                dominantAlbedo = diffuseAlbedo.g();
                dominantThroughput = storedThroughput.g();
            }
            if (diffuseAlbedo.b() > dominantAlbedo) {
                dominantAlbedo = diffuseAlbedo.b();
                dominantThroughput = storedThroughput.b();
            }
            double storedTechniqueMass = dominantAlbedo > 0.0 && dominantThroughput > 0.0
                    ? dominantAlbedo / dominantThroughput : 0.0;
            if (!replayFloatMatches(techniqueMass, storedTechniqueMass)) {
                return ReceiverGuideComparison.MASS_MISMATCH;
            }
            Rgb expectedThroughput = eventThroughput();
            if (!replayFloatMatches(expectedThroughput.r(), storedThroughput.r())
                    || !replayFloatMatches(expectedThroughput.g(), storedThroughput.g())
                    || !replayFloatMatches(expectedThroughput.b(), storedThroughput.b())) {
                return ReceiverGuideComparison.THROUGHPUT_MISMATCH;
            }
            return replayFloatMatches(directionalPdf(cosine), storedDirectionalPdf)
                    ? ReceiverGuideComparison.ACCEPTED
                    : ReceiverGuideComparison.PDF_MISMATCH;
        }
    }

    enum ReceiverGuideComparison {
        ACCEPTED,
        MASS_MISMATCH,
        THROUGHPUT_MISMATCH,
        PDF_MISMATCH
    }

    /**
     * Endpoint-independent diffuse remap terms available before any visibility query. This is a
     * mathematical readiness contract only; it does not authorize history admission or weighting.
     */
    record GuideOnlyDiffuseRemap(DiffuseReceiverGuide receiverGuide,
                                 DiffuseReceiverEdge receiverEdge,
                                 double sourceDistance, double receiverDistance,
                                 double sourceCosine, double receiverCosine,
                                 double sourceDirectionalPdf,
                                 Rgb sourceRadiance, Rgb sourceThroughput) {
        GuideOnlyDiffuseRemap {
            if (receiverGuide == null || receiverEdge == null
                    || !positiveFinite(sourceDistance) || !positiveFinite(receiverDistance)
                    || !positiveFinite(sourceCosine) || !positiveFinite(receiverCosine)
                    || !positiveFinite(sourceDirectionalPdf)
                    || sourceRadiance == null || sourceThroughput == null) {
                throw new IllegalArgumentException("invalid guide-only diffuse remap terms");
            }
            double receiverPdf = receiverGuide.directionalPdf(receiverEdge.cosine());
            new ReconnectionGeometry(sourceDistance, receiverDistance,
                    sourceCosine, receiverCosine, sourceDirectionalPdf, receiverPdf);
            Rgb receiverThroughput = receiverGuide.eventThroughput();
            shiftedDiffuseRadiance(sourceRadiance.r(), sourceThroughput.r(),
                    receiverThroughput.r(), 1.0);
            shiftedDiffuseRadiance(sourceRadiance.g(), sourceThroughput.g(),
                    receiverThroughput.g(), 1.0);
            shiftedDiffuseRadiance(sourceRadiance.b(), sourceThroughput.b(),
                    receiverThroughput.b(), 1.0);
        }

        double receiverDirectionalPdf() {
            return receiverGuide.directionalPdf(receiverEdge.cosine());
        }

        double primarySampleJacobian() {
            return new ReconnectionGeometry(sourceDistance, receiverDistance,
                    sourceCosine, receiverCosine, sourceDirectionalPdf,
                    receiverDirectionalPdf()).primarySampleJacobian();
        }

        Rgb receiverThroughput() {
            return receiverGuide.eventThroughput();
        }

        Rgb shiftedRadiance(Rgb transmittance) {
            if (transmittance == null) {
                throw new IllegalArgumentException("guide-only visibility is required");
            }
            Rgb receiver = receiverThroughput();
            return new Rgb(
                    shiftedDiffuseRadiance(sourceRadiance.r(), sourceThroughput.r(),
                            receiver.r(), transmittance.r()),
                    shiftedDiffuseRadiance(sourceRadiance.g(), sourceThroughput.g(),
                            receiver.g(), transmittance.g()),
                    shiftedDiffuseRadiance(sourceRadiance.b(), sourceThroughput.b(),
                            receiver.b(), transmittance.b()));
        }

        Rgb unoccludedShiftedRadiance() {
            return shiftedRadiance(new Rgb(1.0, 1.0, 1.0));
        }

        double shiftedTarget(Rgb transmittance) {
            return shiftedRadiance(transmittance).luminance();
        }

        double mergeWeight(Rgb transmittance, double sourceFinalWeight,
                           double sourceEffectiveCount, double maxSourceCount) {
            return new SpatialGrisWeight(shiftedTarget(transmittance), sourceFinalWeight,
                    sourceEffectiveCount, maxSourceCount, primarySampleJacobian()).mergeWeight();
        }

        double selectionProbability(Rgb transmittance, double sourceFinalWeight,
                                    double sourceEffectiveCount, double maxSourceCount,
                                    double currentWeightSum) {
            return new SpatialGrisWeight(shiftedTarget(transmittance), sourceFinalWeight,
                    sourceEffectiveCount, maxSourceCount, primarySampleJacobian())
                    .selectionProbability(currentWeightSum);
        }

        double stableSelectionProbability(Rgb transmittance, double sourceFinalWeight,
                                          double sourceEffectiveCount, double maxSourceCount,
                                          double currentWeightSum) {
            return new SpatialGrisWeight(shiftedTarget(transmittance), sourceFinalWeight,
                    sourceEffectiveCount, maxSourceCount, primarySampleJacobian())
                    .stableSelectionProbability(currentWeightSum);
        }

        boolean stableSelectsSource(Rgb transmittance, double sourceFinalWeight,
                                    double sourceEffectiveCount, double maxSourceCount,
                                    double currentWeightSum, double randomUnit) {
            return new SpatialGrisWeight(shiftedTarget(transmittance), sourceFinalWeight,
                    sourceEffectiveCount, maxSourceCount, primarySampleJacobian())
                    .stableSelectsSource(currentWeightSum, randomUnit);
        }

        SpatialShadowMerge stableShadowMerge(Rgb transmittance, double sourceFinalWeight,
                                             double sourceEffectiveCount, double maxSourceCount,
                                             double currentWeightSum,
                                             double currentEffectiveCount,
                                             double currentTarget, double randomUnit) {
            return new SpatialGrisWeight(shiftedTarget(transmittance), sourceFinalWeight,
                    sourceEffectiveCount, maxSourceCount, primarySampleJacobian())
                    .stableShadowMerge(currentWeightSum, currentEffectiveCount,
                            currentTarget, randomUnit);
        }
    }

    enum GuideOnlyRemapDecision {
        READY,
        GEOMETRY_REJECT,
        PDF_REJECT,
        THROUGHPUT_REJECT
    }

    /** Ordered shadow-counter policy; strict cross-frame admission remains a separate decision. */
    record GuideOnlyRemapPolicy(boolean geometryValid, boolean pdfValid,
                                boolean throughputValid) {
        GuideOnlyRemapDecision firstReject() {
            if (!geometryValid) return GuideOnlyRemapDecision.GEOMETRY_REJECT;
            if (!pdfValid) return GuideOnlyRemapDecision.PDF_REJECT;
            if (!throughputValid) return GuideOnlyRemapDecision.THROUGHPUT_REJECT;
            return GuideOnlyRemapDecision.READY;
        }
    }

    enum GuideOnlyVisibilityDecision {
        CLEAR,
        TINTED,
        OCCLUDED,
        ARITHMETIC_REJECT
    }

    /** Ordered production-shadow result used only by the guide visibility counter audit. */
    record GuideOnlyVisibilityPolicy(boolean arithmeticValid, boolean anyTransmission,
                                     boolean fullyClear) {
        GuideOnlyVisibilityDecision result() {
            if (!arithmeticValid) return GuideOnlyVisibilityDecision.ARITHMETIC_REJECT;
            if (!anyTransmission) return GuideOnlyVisibilityDecision.OCCLUDED;
            if (fullyClear) return GuideOnlyVisibilityDecision.CLEAR;
            return GuideOnlyVisibilityDecision.TINTED;
        }
    }

    enum GuideOnlyShiftedTargetDecision {
        POSITIVE,
        ZERO,
        ARITHMETIC_REJECT
    }

    /** Ordered post-visibility target result; it does not authorize sampling or weighting. */
    record GuideOnlyShiftedTargetPolicy(boolean arithmeticValid, boolean positiveTarget) {
        GuideOnlyShiftedTargetDecision result() {
            if (!arithmeticValid) return GuideOnlyShiftedTargetDecision.ARITHMETIC_REJECT;
            return positiveTarget
                    ? GuideOnlyShiftedTargetDecision.POSITIVE
                    : GuideOnlyShiftedTargetDecision.ZERO;
        }
    }

    enum GuideOnlyMergeWeightDecision {
        POSITIVE,
        ZERO,
        ARITHMETIC_REJECT
    }

    /** Ordered counter-only result; it does not authorize selection or reservoir mutation. */
    record GuideOnlyMergeWeightPolicy(boolean arithmeticValid, boolean positiveWeight) {
        GuideOnlyMergeWeightDecision result() {
            if (!arithmeticValid) return GuideOnlyMergeWeightDecision.ARITHMETIC_REJECT;
            return positiveWeight
                    ? GuideOnlyMergeWeightDecision.POSITIVE
                    : GuideOnlyMergeWeightDecision.ZERO;
        }
    }

    enum GuideOnlySelectionDecision {
        POSITIVE,
        ZERO,
        CURRENT_WEIGHT_REJECT,
        PROBABILITY_HIGH_REJECT,
        ARITHMETIC_REJECT
    }

    /** Ordered probability-only result; it does not draw RNG or authorize reservoir mutation. */
    record GuideOnlySelectionPolicy(boolean currentWeightValid, boolean arithmeticValid,
                                    boolean probabilityAboveOne, boolean positiveProbability) {
        GuideOnlySelectionDecision firstReject() {
            if (!currentWeightValid) return GuideOnlySelectionDecision.CURRENT_WEIGHT_REJECT;
            if (!arithmeticValid) return GuideOnlySelectionDecision.ARITHMETIC_REJECT;
            if (probabilityAboveOne) return GuideOnlySelectionDecision.PROBABILITY_HIGH_REJECT;
            return positiveProbability
                    ? GuideOnlySelectionDecision.POSITIVE
                    : GuideOnlySelectionDecision.ZERO;
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

        PersistentSourceRootOrigin advance(double cameraDeltaX, double cameraDeltaY,
                                           double cameraDeltaZ) {
            return new PersistentSourceRootOrigin(
                    cameraRelativeX - cameraDeltaX,
                    cameraRelativeY - cameraDeltaY,
                    cameraRelativeZ - cameraDeltaZ);
        }
    }

    /**
     * Same-frame provenance paired with a guide scratch record. A selected historical path keeps
     * its original replay transcript but advances both its camera-relative root and screen key to
     * the independently reprojected source. A retained current path captures both values fresh.
     */
    record GuideSourceRootProvenance(int sourcePixelIndex, PersistentSourceRootOrigin origin) {
        GuideSourceRootProvenance {
            if (sourcePixelIndex < 0 || origin == null) {
                throw new IllegalArgumentException("guide source-root provenance is incomplete");
            }
        }

        static GuideSourceRootProvenance selected(
                int reprojectedSourcePixelIndex, PersistentSourceRootOrigin previousOrigin,
                double cameraDeltaX, double cameraDeltaY, double cameraDeltaZ) {
            if (previousOrigin == null) {
                throw new IllegalArgumentException("selected guide source root is missing");
            }
            return new GuideSourceRootProvenance(reprojectedSourcePixelIndex,
                    previousOrigin.advance(cameraDeltaX, cameraDeltaY, cameraDeltaZ));
        }

        static GuideSourceRootProvenance retained(
                int currentSourcePixelIndex,
                double rootX, double rootY, double rootZ,
                double cameraOffsetX, double cameraOffsetY, double cameraOffsetZ) {
            return new GuideSourceRootProvenance(currentSourcePixelIndex,
                    PersistentSourceRootOrigin.capture(rootX, rootY, rootZ,
                            cameraOffsetX, cameraOffsetY, cameraOffsetZ));
        }
    }

    enum GuidePreviousReplayOutcome {
        LIFECYCLE_REJECT,
        RECEIVER_REPROJECTION_REJECT,
        PREVIOUS_EMPTY,
        METADATA_REJECT,
        RECEIVER_SURFACE_REJECT,
        SOURCE_REPROJECTION_REJECT,
        SOURCE_SURFACE_REJECT,
        SOURCE_REPLAY_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    /**
     * Ordered, exclusive CPU authority for the one-frame guide-pair replay diagnostic. The result
     * is classification only: acceptance does not authorize a merge, history write, or use as a
     * spatial source.
     */
    static GuidePreviousReplayOutcome guidePreviousReplayOutcome(
            boolean lifecycleAvailable,
            boolean receiverReprojected,
            boolean previousPresent,
            boolean metadataValid,
            boolean receiverSurfaceValid,
            boolean sourceReprojected,
            boolean sourceSurfaceValid,
            boolean sourceReplayValid,
            MappingKind mappingKind) {
        if (!lifecycleAvailable) {
            return GuidePreviousReplayOutcome.LIFECYCLE_REJECT;
        }
        if (!receiverReprojected) {
            return GuidePreviousReplayOutcome.RECEIVER_REPROJECTION_REJECT;
        }
        if (!previousPresent) {
            return GuidePreviousReplayOutcome.PREVIOUS_EMPTY;
        }
        if (!metadataValid) {
            return GuidePreviousReplayOutcome.METADATA_REJECT;
        }
        if (!receiverSurfaceValid) {
            return GuidePreviousReplayOutcome.RECEIVER_SURFACE_REJECT;
        }
        if (!sourceReprojected) {
            return GuidePreviousReplayOutcome.SOURCE_REPROJECTION_REJECT;
        }
        if (!sourceSurfaceValid) {
            return GuidePreviousReplayOutcome.SOURCE_SURFACE_REJECT;
        }
        if (!sourceReplayValid) {
            return GuidePreviousReplayOutcome.SOURCE_REPLAY_REJECT;
        }
        if (mappingKind == null) {
            throw new IllegalArgumentException("accepted guide replay requires a mapping kind");
        }
        return mappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? GuidePreviousReplayOutcome.SELECTED_ACCEPTED
                : GuidePreviousReplayOutcome.RETAINED_ACCEPTED;
    }

    enum BranchScratchStorageOutcome {
        METADATA_REJECT,
        PAIR_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    enum BranchAgedStorageOutcome {
        METADATA_REJECT,
        PAIR_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    enum BranchWinnerStorageOutcome {
        EMPTY,
        EMPTY_DIRTY_REJECT,
        METADATA_REJECT,
        PAIR_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    enum BranchWinnerPreviousReplayOutcome {
        RECEIVER_REPROJECTION_REJECT,
        PREVIOUS_EMPTY,
        METADATA_REJECT,
        RECEIVER_SURFACE_REJECT,
        SOURCE_REPROJECTION_REJECT,
        SOURCE_SURFACE_REJECT,
        SOURCE_REPLAY_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    enum BranchWinnerCandidateOwnerOutcome {
        EMPTY,
        METADATA_REJECT,
        SELECTED_ACCEPTED,
        RETAINED_ACCEPTED
    }

    enum BranchCandidateRetentionOutcome {
        EMPTY,
        FUTURE_REJECT,
        GENERATION_REJECT,
        MAPPING_REJECT,
        CURRENT,
        AGE_ONE,
        AGE_TWO,
        AGE_THREE,
        EXPIRED;

        boolean live() {
            return this == CURRENT || this == AGE_ONE || this == AGE_TWO || this == AGE_THREE;
        }
    }

    /** Ordered CPU mirror for the bounded four-frame promotion-ring metadata audit. */
    static BranchCandidateRetentionOutcome branchCandidateRetentionOutcome(
            boolean promoted, long validationFrame, long currentFrame,
            int storedGeneration, int currentGeneration,
            MappingKind promotionMapping, MappingKind metadataMapping,
            MappingKind reservoirMapping) {
        if (!promoted) return BranchCandidateRetentionOutcome.EMPTY;
        if (validationFrame > currentFrame) {
            return BranchCandidateRetentionOutcome.FUTURE_REJECT;
        }
        if (storedGeneration != currentGeneration) {
            return BranchCandidateRetentionOutcome.GENERATION_REJECT;
        }
        if (promotionMapping == null || promotionMapping != metadataMapping
                || promotionMapping != reservoirMapping) {
            return BranchCandidateRetentionOutcome.MAPPING_REJECT;
        }
        long age = currentFrame - validationFrame;
        if (age == 0L) return BranchCandidateRetentionOutcome.CURRENT;
        if (age == 1L) return BranchCandidateRetentionOutcome.AGE_ONE;
        if (age == 2L) return BranchCandidateRetentionOutcome.AGE_TWO;
        if (age == 3L) return BranchCandidateRetentionOutcome.AGE_THREE;
        return BranchCandidateRetentionOutcome.EXPIRED;
    }

    record BranchReplayCameraProvenance(double deltaX, double deltaY, double deltaZ,
                                        long advancedFrame) {
        BranchReplayCameraProvenance {
            if (!Double.isFinite(deltaX) || !Double.isFinite(deltaY)
                    || !Double.isFinite(deltaZ)) {
                throw new IllegalArgumentException("branch replay camera provenance must be finite");
            }
        }

        static BranchReplayCameraProvenance capture(long frame) {
            return new BranchReplayCameraProvenance(0.0, 0.0, 0.0, frame);
        }

        BranchReplayCameraProvenance advance(
                long currentFrame, double frameDeltaX, double frameDeltaY, double frameDeltaZ) {
            if (currentFrame != advancedFrame + 1L) {
                throw new IllegalArgumentException(
                        "branch replay provenance must advance exactly one frame");
            }
            return new BranchReplayCameraProvenance(
                    deltaX + frameDeltaX, deltaY + frameDeltaY, deltaZ + frameDeltaZ,
                    currentFrame);
        }
    }

    enum BranchCandidateAgeReplayOutcome {
        EMPTY,
        METADATA_REJECT,
        CURRENT_SKIP,
        EXPIRED,
        PROVENANCE_REJECT,
        IDENTITY_ACCEPTED,
        IDENTITY_REJECT,
        MAPPED_ORIGINAL_ROOT_ACCEPTED,
        MAPPED_ORIGINAL_ROOT_REJECT
    }

    /** CPU mirror for age-aware original-root replay; it never authorizes another spatial shift. */
    static BranchCandidateAgeReplayOutcome branchCandidateAgeReplayOutcome(
            BranchCandidateRetentionOutcome retention, boolean provenanceCurrent,
            MappingKind mappingKind, boolean originalRootReplayMatches) {
        if (retention == null) {
            throw new IllegalArgumentException("branch retention outcome is required");
        }
        if (retention == BranchCandidateRetentionOutcome.EMPTY) {
            return BranchCandidateAgeReplayOutcome.EMPTY;
        }
        if (retention == BranchCandidateRetentionOutcome.FUTURE_REJECT
                || retention == BranchCandidateRetentionOutcome.GENERATION_REJECT
                || retention == BranchCandidateRetentionOutcome.MAPPING_REJECT) {
            return BranchCandidateAgeReplayOutcome.METADATA_REJECT;
        }
        if (retention == BranchCandidateRetentionOutcome.CURRENT) {
            return BranchCandidateAgeReplayOutcome.CURRENT_SKIP;
        }
        if (retention == BranchCandidateRetentionOutcome.EXPIRED) {
            return BranchCandidateAgeReplayOutcome.EXPIRED;
        }
        if (!provenanceCurrent) {
            return BranchCandidateAgeReplayOutcome.PROVENANCE_REJECT;
        }
        if (mappingKind == null) {
            throw new IllegalArgumentException("age replay requires a mapping kind");
        }
        if (mappingKind == MappingKind.DIFFUSE_RECONNECTION) {
            return originalRootReplayMatches
                    ? BranchCandidateAgeReplayOutcome.MAPPED_ORIGINAL_ROOT_ACCEPTED
                    : BranchCandidateAgeReplayOutcome.MAPPED_ORIGINAL_ROOT_REJECT;
        }
        return originalRootReplayMatches
                ? BranchCandidateAgeReplayOutcome.IDENTITY_ACCEPTED
                : BranchCandidateAgeReplayOutcome.IDENTITY_REJECT;
    }

    record BranchReceiverPosition(double x, double y, double z) {
        BranchReceiverPosition {
            if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(z)) {
                throw new IllegalArgumentException("branch receiver position must be finite");
            }
        }
    }

    /** Reconstructs a capture-time camera-relative receiver in the current guide-buffer frame. */
    static BranchReceiverPosition branchReceiverCurrentGuidePosition(
            BranchReceiverPosition capturedCameraRelative,
            BranchReceiverPosition currentCameraOffset,
            BranchReplayCameraProvenance replayCamera) {
        if (capturedCameraRelative == null || currentCameraOffset == null
                || replayCamera == null) {
            throw new IllegalArgumentException(
                    "branch receiver reprojection requires position and camera provenance");
        }
        return new BranchReceiverPosition(
                capturedCameraRelative.x() + currentCameraOffset.x() - replayCamera.deltaX(),
                capturedCameraRelative.y() + currentCameraOffset.y() - replayCamera.deltaY(),
                capturedCameraRelative.z() + currentCameraOffset.z() - replayCamera.deltaZ());
    }

    enum BranchReceiverAdmissionOutcome {
        SOURCE_REPLAY_REJECT,
        CLIP_REJECT,
        BOUNDS_REJECT,
        SURFACE_REJECT,
        AGE_ONE_ADMITTED,
        AGE_TWO_ADMITTED,
        AGE_THREE_ADMITTED
    }

    /** Ordered CPU mirror for direct world-space receiver-root admission. */
    static BranchReceiverAdmissionOutcome branchReceiverAdmissionOutcome(
            BranchCandidateRetentionOutcome retention, boolean originalRootReplayMatches,
            boolean clipVisible, boolean candidateInBounds, boolean strictSurfaceMatch) {
        if (!originalRootReplayMatches) {
            return BranchReceiverAdmissionOutcome.SOURCE_REPLAY_REJECT;
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "receiver admission requires a live aged branch candidate");
        }
        if (!clipVisible) {
            return BranchReceiverAdmissionOutcome.CLIP_REJECT;
        }
        if (!candidateInBounds) {
            return BranchReceiverAdmissionOutcome.BOUNDS_REJECT;
        }
        if (!strictSurfaceMatch) {
            return BranchReceiverAdmissionOutcome.SURFACE_REJECT;
        }
        return switch (retention) {
            case AGE_ONE -> BranchReceiverAdmissionOutcome.AGE_ONE_ADMITTED;
            case AGE_TWO -> BranchReceiverAdmissionOutcome.AGE_TWO_ADMITTED;
            case AGE_THREE -> BranchReceiverAdmissionOutcome.AGE_THREE_ADMITTED;
            default -> throw new IllegalStateException("unreachable branch receiver age");
        };
    }

    enum BranchDirectRemapOutcome {
        RECEIVER_ADMISSION_REJECT,
        GUIDE_REJECT,
        EDGE_REJECT,
        GEOMETRY_REJECT,
        PDF_REJECT,
        THROUGHPUT_REJECT,
        AGE_ONE_READY,
        AGE_TWO_READY,
        AGE_THREE_READY
    }

    /** Ordered CPU mirror for the pre-visibility direct aged remap gate. */
    static BranchDirectRemapOutcome branchDirectRemapOutcome(
            BranchCandidateRetentionOutcome retention, boolean receiverAdmitted,
            boolean receiverGuideValid, boolean sourceEdgeValid, boolean geometryValid,
            boolean pdfValid, boolean throughputValid) {
        if (!receiverAdmitted) {
            return BranchDirectRemapOutcome.RECEIVER_ADMISSION_REJECT;
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException("direct remap requires a live aged branch candidate");
        }
        if (!receiverGuideValid) {
            return BranchDirectRemapOutcome.GUIDE_REJECT;
        }
        if (!sourceEdgeValid) {
            return BranchDirectRemapOutcome.EDGE_REJECT;
        }
        if (!geometryValid) {
            return BranchDirectRemapOutcome.GEOMETRY_REJECT;
        }
        if (!pdfValid) {
            return BranchDirectRemapOutcome.PDF_REJECT;
        }
        if (!throughputValid) {
            return BranchDirectRemapOutcome.THROUGHPUT_REJECT;
        }
        return switch (retention) {
            case AGE_ONE -> BranchDirectRemapOutcome.AGE_ONE_READY;
            case AGE_TWO -> BranchDirectRemapOutcome.AGE_TWO_READY;
            case AGE_THREE -> BranchDirectRemapOutcome.AGE_THREE_READY;
            default -> throw new IllegalStateException("unreachable branch direct-remap age");
        };
    }

    enum BranchDirectVisibilityOutcome {
        DIRECT_REMAP_REJECT,
        CLEAR,
        TINTED,
        OCCLUDED,
        INVALID
    }

    enum BranchDirectTargetOutcome {
        NOT_ELIGIBLE,
        POSITIVE,
        ZERO,
        INVALID
    }

    record BranchDirectTargetAudit(
            BranchDirectVisibilityOutcome visibility,
            BranchDirectTargetOutcome target,
            BranchCandidateRetentionOutcome retention) {
    }

    /** Ordered CPU mirror for the production-shadow and shifted-target aged branch audit. */
    static BranchDirectTargetAudit branchDirectTargetAudit(
            BranchCandidateRetentionOutcome retention, boolean directRemapReady,
            boolean visibilityArithmeticValid, boolean anyTransmission, boolean fullyClear,
            boolean targetArithmeticValid, boolean positiveTarget) {
        if (!directRemapReady) {
            return new BranchDirectTargetAudit(
                    BranchDirectVisibilityOutcome.DIRECT_REMAP_REJECT,
                    BranchDirectTargetOutcome.NOT_ELIGIBLE, retention);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct target audit requires a live aged branch candidate");
        }
        if (!visibilityArithmeticValid) {
            return new BranchDirectTargetAudit(BranchDirectVisibilityOutcome.INVALID,
                    BranchDirectTargetOutcome.NOT_ELIGIBLE, retention);
        }
        BranchDirectVisibilityOutcome visibility = !anyTransmission
                ? BranchDirectVisibilityOutcome.OCCLUDED
                : fullyClear
                        ? BranchDirectVisibilityOutcome.CLEAR
                        : BranchDirectVisibilityOutcome.TINTED;
        BranchDirectTargetOutcome target = !targetArithmeticValid
                ? BranchDirectTargetOutcome.INVALID
                : positiveTarget
                        ? BranchDirectTargetOutcome.POSITIVE
                        : BranchDirectTargetOutcome.ZERO;
        return new BranchDirectTargetAudit(visibility, target, retention);
    }

    enum BranchDirectWeightOutcome {
        TARGET_REJECT,
        POSITIVE,
        ZERO,
        INVALID
    }

    enum BranchDirectSourceCountOutcome {
        NOT_ELIGIBLE,
        UNCAPPED,
        CAPPED
    }

    record BranchDirectWeightAudit(
            BranchDirectWeightOutcome weight,
            BranchDirectSourceCountOutcome sourceCount,
            BranchCandidateRetentionOutcome retention,
            double mergeWeight) {
    }

    /** CPU mirror for the aged counter-only GRIS weight and source-M cap partition. */
    static BranchDirectWeightAudit branchDirectWeightAudit(
            BranchCandidateRetentionOutcome retention, boolean targetReady,
            double shiftedTarget, double sourceFinalWeight, double sourceEffectiveCount,
            double directPssJacobian) {
        if (!targetReady) {
            return new BranchDirectWeightAudit(BranchDirectWeightOutcome.TARGET_REJECT,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, retention, 0.0);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct weight audit requires a live aged branch candidate");
        }
        boolean termsValid = nonNegativeFinite(shiftedTarget)
                && nonNegativeFinite(sourceFinalWeight)
                && nonNegativeFinite(sourceEffectiveCount)
                && positiveFinite(directPssJacobian);
        if (!termsValid) {
            return new BranchDirectWeightAudit(BranchDirectWeightOutcome.INVALID,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, retention, 0.0);
        }
        double sourceCount = Math.min(sourceEffectiveCount, MAX_SPATIAL_SOURCE_COUNT);
        double mergeWeight = shiftedTarget * sourceFinalWeight;
        mergeWeight *= sourceCount;
        mergeWeight *= directPssJacobian;
        if (!Double.isFinite(mergeWeight) || mergeWeight < 0.0) {
            return new BranchDirectWeightAudit(BranchDirectWeightOutcome.INVALID,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, retention, 0.0);
        }
        BranchDirectWeightOutcome weight = mergeWeight > 0.0
                ? BranchDirectWeightOutcome.POSITIVE
                : BranchDirectWeightOutcome.ZERO;
        BranchDirectSourceCountOutcome count =
                sourceEffectiveCount > MAX_SPATIAL_SOURCE_COUNT
                        ? BranchDirectSourceCountOutcome.CAPPED
                        : BranchDirectSourceCountOutcome.UNCAPPED;
        return new BranchDirectWeightAudit(weight, count, retention, mergeWeight);
    }

    enum BranchDirectSelectionOutcome {
        WEIGHT_REJECT,
        CURRENT_REJECT,
        ZERO,
        OPEN,
        ONE,
        INVALID
    }

    enum BranchDirectCurrentWeightOutcome {
        NOT_ELIGIBLE,
        ZERO,
        POSITIVE
    }

    record BranchDirectSelectionAudit(
            BranchDirectSelectionOutcome selection,
            BranchDirectCurrentWeightOutcome currentWeight,
            BranchCandidateRetentionOutcome retention,
            double probability) {
    }

    /** CPU mirror for the aged counter-only overflow-stable relative selection ratio. */
    static BranchDirectSelectionAudit branchDirectSelectionAudit(
            BranchCandidateRetentionOutcome retention, boolean weightReady,
            double currentWeightSum, double mergeWeight) {
        if (!weightReady) {
            return new BranchDirectSelectionAudit(BranchDirectSelectionOutcome.WEIGHT_REJECT,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, retention, 0.0);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct selection audit requires a live aged branch candidate");
        }
        if (!nonNegativeFinite(currentWeightSum)) {
            return new BranchDirectSelectionAudit(BranchDirectSelectionOutcome.CURRENT_REJECT,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, retention, 0.0);
        }
        if (!nonNegativeFinite(mergeWeight)) {
            return new BranchDirectSelectionAudit(BranchDirectSelectionOutcome.INVALID,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, retention, 0.0);
        }

        double probability;
        if (mergeWeight == 0.0) {
            probability = 0.0;
        } else if (currentWeightSum <= mergeWeight) {
            probability = 1.0 / (1.0 + currentWeightSum / mergeWeight);
        } else {
            double ratio = mergeWeight / currentWeightSum;
            probability = ratio / (1.0 + ratio);
        }
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            return new BranchDirectSelectionAudit(BranchDirectSelectionOutcome.INVALID,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, retention, 0.0);
        }

        BranchDirectSelectionOutcome selection = probability == 0.0
                ? BranchDirectSelectionOutcome.ZERO
                : probability == 1.0
                        ? BranchDirectSelectionOutcome.ONE
                        : BranchDirectSelectionOutcome.OPEN;
        BranchDirectCurrentWeightOutcome currentWeight = currentWeightSum == 0.0
                ? BranchDirectCurrentWeightOutcome.ZERO
                : BranchDirectCurrentWeightOutcome.POSITIVE;
        return new BranchDirectSelectionAudit(selection, currentWeight, retention, probability);
    }

    enum BranchDirectBernoulliOutcome {
        SELECTION_REJECT,
        SELECTED,
        RETAINED,
        INVALID
    }

    enum BranchDirectBernoulliBoundary {
        NOT_ELIGIBLE,
        ZERO,
        OPEN,
        ONE
    }

    record BranchDirectBernoulliAudit(
            BranchDirectBernoulliOutcome outcome,
            BranchDirectBernoulliBoundary boundary,
            BranchCandidateRetentionOutcome retention,
            double random,
            boolean zeroViolation,
            boolean oneViolation) {
    }

    /** CPU mirror for the aged local-hash Bernoulli partition without payload mutation. */
    static BranchDirectBernoulliAudit branchDirectBernoulliAudit(
            BranchCandidateRetentionOutcome retention, boolean selectionReady,
            double probability, int receiverPixelIndex, int candidateLinearIndex,
            int frameIndex) {
        if (!selectionReady) {
            return new BranchDirectBernoulliAudit(
                    BranchDirectBernoulliOutcome.SELECTION_REJECT,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, retention, 0.0, false, false);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct Bernoulli audit requires a live aged branch candidate");
        }
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            return new BranchDirectBernoulliAudit(BranchDirectBernoulliOutcome.INVALID,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, retention, 0.0, false, false);
        }

        int mixedSeed = receiverPixelIndex
                ^ candidateLinearIndex * BRANCH_BERNOULLI_ENTRY_MULTIPLIER
                ^ frameIndex * BRANCH_BERNOULLI_FRAME_MULTIPLIER
                ^ BRANCH_BERNOULLI_SALT;
        int hashedSeed = RtPathReplayReference.pathHash(mixedSeed);
        double random = (hashedSeed >>> 8) * (1.0 / 16_777_216.0);
        if (!Double.isFinite(random) || random < 0.0 || random >= 1.0) {
            return new BranchDirectBernoulliAudit(BranchDirectBernoulliOutcome.INVALID,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, retention, 0.0, false, false);
        }

        boolean selected = random < probability;
        BranchDirectBernoulliBoundary boundary = probability == 0.0
                ? BranchDirectBernoulliBoundary.ZERO
                : probability == 1.0
                        ? BranchDirectBernoulliBoundary.ONE
                        : BranchDirectBernoulliBoundary.OPEN;
        return new BranchDirectBernoulliAudit(
                selected ? BranchDirectBernoulliOutcome.SELECTED
                        : BranchDirectBernoulliOutcome.RETAINED,
                boundary, retention, random,
                probability == 0.0 && selected,
                probability == 1.0 && !selected);
    }

    enum BranchDirectPostSelectionOutcome {
        BERNOULLI_REJECT,
        CURRENT_REJECT,
        NEXT_REJECT,
        SELECTED_READY,
        SELECTED_TARGET_REJECT,
        SELECTED_FINAL_REJECT,
        RETAINED_READY,
        RETAINED_TARGET_REJECT,
        RETAINED_FINAL_REJECT
    }

    enum BranchDirectStoredCapOutcome {
        NOT_ELIGIBLE,
        UNCAPPED,
        CAPPED
    }

    record BranchDirectPostSelectionAudit(
            BranchDirectPostSelectionOutcome outcome,
            BranchDirectStoredCapOutcome weightCap,
            BranchDirectStoredCapOutcome countCap,
            BranchCandidateRetentionOutcome retention,
            double nextWeightSum,
            double nextEffectiveCount,
            double selectedTarget,
            double finalWeight,
            boolean sourceSelected,
            boolean empty) {
    }

    /** CPU mirror for the aged register-only selected/retained final-weight arithmetic. */
    static BranchDirectPostSelectionAudit branchDirectPostSelectionAudit(
            BranchCandidateRetentionOutcome retention, boolean bernoulliReady,
            boolean sourceSelected, double currentWeightSum, double currentEffectiveCount,
            double currentTarget, double mergeWeight, double sourceEffectiveCount,
            double shiftedTarget) {
        if (!bernoulliReady) {
            return branchDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.BERNOULLI_REJECT,
                    retention, sourceSelected);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct post-selection audit requires a live aged branch candidate");
        }
        if (!nonNegativeFinite(currentWeightSum)
                || !nonNegativeFinite(currentEffectiveCount)
                || !nonNegativeFinite(currentTarget)) {
            return branchDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.CURRENT_REJECT,
                    retention, sourceSelected);
        }
        if (!nonNegativeFinite(mergeWeight)
                || !nonNegativeFinite(sourceEffectiveCount)
                || !nonNegativeFinite(shiftedTarget)) {
            return branchDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.NEXT_REJECT,
                    retention, sourceSelected);
        }

        double sourceCount = Math.min(sourceEffectiveCount, MAX_SPATIAL_SOURCE_COUNT);
        boolean weightCapped = currentWeightSum > MAX_STORED_WEIGHT_SUM - mergeWeight;
        boolean countCapped = currentEffectiveCount
                > MAX_STORED_EFFECTIVE_COUNT - sourceCount;
        double nextWeightSum = weightCapped
                ? MAX_STORED_WEIGHT_SUM : currentWeightSum + mergeWeight;
        double nextEffectiveCount = countCapped
                ? MAX_STORED_EFFECTIVE_COUNT : currentEffectiveCount + sourceCount;
        if (!nonNegativeFinite(nextWeightSum) || !nonNegativeFinite(nextEffectiveCount)) {
            return branchDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.NEXT_REJECT,
                    retention, sourceSelected);
        }

        BranchDirectStoredCapOutcome weightCap = weightCapped
                ? BranchDirectStoredCapOutcome.CAPPED
                : BranchDirectStoredCapOutcome.UNCAPPED;
        BranchDirectStoredCapOutcome countCap = countCapped
                ? BranchDirectStoredCapOutcome.CAPPED
                : BranchDirectStoredCapOutcome.UNCAPPED;
        double selectedTarget = sourceSelected ? shiftedTarget : currentTarget;
        boolean empty = nextWeightSum == 0.0;
        if (!nonNegativeFinite(selectedTarget) || (!empty && selectedTarget == 0.0)) {
            return new BranchDirectPostSelectionAudit(
                    sourceSelected
                            ? BranchDirectPostSelectionOutcome.SELECTED_TARGET_REJECT
                            : BranchDirectPostSelectionOutcome.RETAINED_TARGET_REJECT,
                    weightCap, countCap, retention, nextWeightSum, nextEffectiveCount,
                    selectedTarget, 0.0, sourceSelected, false);
        }

        double finalWeight = 0.0;
        if (!empty) {
            if (nextEffectiveCount == 0.0) {
                return new BranchDirectPostSelectionAudit(
                        sourceSelected
                                ? BranchDirectPostSelectionOutcome.SELECTED_FINAL_REJECT
                                : BranchDirectPostSelectionOutcome.RETAINED_FINAL_REJECT,
                        weightCap, countCap, retention, nextWeightSum, nextEffectiveCount,
                        selectedTarget, 0.0, sourceSelected, false);
            }
            double denominator = nextEffectiveCount * selectedTarget;
            finalWeight = nextWeightSum / denominator;
            if (!positiveFinite(denominator) || !positiveFinite(finalWeight)) {
                return new BranchDirectPostSelectionAudit(
                        sourceSelected
                                ? BranchDirectPostSelectionOutcome.SELECTED_FINAL_REJECT
                                : BranchDirectPostSelectionOutcome.RETAINED_FINAL_REJECT,
                        weightCap, countCap, retention, nextWeightSum, nextEffectiveCount,
                        selectedTarget, 0.0, sourceSelected, false);
            }
        }
        return new BranchDirectPostSelectionAudit(
                sourceSelected
                        ? BranchDirectPostSelectionOutcome.SELECTED_READY
                        : BranchDirectPostSelectionOutcome.RETAINED_READY,
                weightCap, countCap, retention, nextWeightSum, nextEffectiveCount,
                selectedTarget, finalWeight, sourceSelected, empty);
    }

    private static BranchDirectPostSelectionAudit branchDirectPostSelectionReject(
            BranchDirectPostSelectionOutcome outcome,
            BranchCandidateRetentionOutcome retention,
            boolean sourceSelected) {
        return new BranchDirectPostSelectionAudit(outcome,
                BranchDirectStoredCapOutcome.NOT_ELIGIBLE,
                BranchDirectStoredCapOutcome.NOT_ELIGIBLE,
                retention, 0.0, 0.0, 0.0, 0.0, sourceSelected, false);
    }

    enum BranchDirectRecordOutcome {
        POST_SELECTION_REJECT,
        SELECTED_READY,
        SELECTED_REJECT,
        RETAINED_READY,
        RETAINED_REJECT,
        EMPTY_READY
    }

    enum BranchDirectLaneOutcome {
        NOT_ELIGIBLE,
        READY,
        REJECT
    }

    record BranchDirectRecordAudit(
            BranchDirectRecordOutcome outcome,
            BranchDirectLaneOutcome selectedRewrite,
            BranchDirectLaneOutcome selectedPreserve,
            BranchDirectLaneOutcome retainedPreserve,
            BranchDirectLaneOutcome weights,
            BranchCandidateRetentionOutcome retention,
            boolean sourceSelected) {
    }

    /** CPU mirror for complete aged branch-record construction without a device write. */
    static BranchDirectRecordAudit branchDirectRecordAudit(
            BranchCandidateRetentionOutcome retention,
            boolean postSelectionReady,
            boolean sourceSelected,
            boolean empty,
            boolean selectedRewriteReady,
            boolean selectedPreserveReady,
            boolean retainedPreserveReady,
            boolean weightsReady,
            boolean metadataReady) {
        if (!postSelectionReady) {
            return branchDirectRecordAuditResult(
                    BranchDirectRecordOutcome.POST_SELECTION_REJECT,
                    retention, sourceSelected,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct record audit requires a live aged branch candidate");
        }
        if (empty) {
            return branchDirectRecordAuditResult(
                    sourceSelected
                            ? BranchDirectRecordOutcome.SELECTED_REJECT
                            : BranchDirectRecordOutcome.EMPTY_READY,
                    retention, sourceSelected,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }

        BranchDirectLaneOutcome weightLane = weightsReady
                ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
        if (sourceSelected) {
            BranchDirectLaneOutcome rewriteLane = selectedRewriteReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            BranchDirectLaneOutcome preserveLane = selectedPreserveReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            boolean ready = selectedRewriteReady && selectedPreserveReady
                    && weightsReady && metadataReady;
            return branchDirectRecordAuditResult(
                    ready ? BranchDirectRecordOutcome.SELECTED_READY
                            : BranchDirectRecordOutcome.SELECTED_REJECT,
                    retention, true, rewriteLane, preserveLane,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE, weightLane);
        }

        BranchDirectLaneOutcome retainedLane = retainedPreserveReady
                ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
        boolean ready = retainedPreserveReady && weightsReady && metadataReady;
        return branchDirectRecordAuditResult(
                ready ? BranchDirectRecordOutcome.RETAINED_READY
                        : BranchDirectRecordOutcome.RETAINED_REJECT,
                retention, false,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                retainedLane, weightLane);
    }

    private static BranchDirectRecordAudit branchDirectRecordAuditResult(
            BranchDirectRecordOutcome outcome,
            BranchCandidateRetentionOutcome retention,
            boolean sourceSelected,
            BranchDirectLaneOutcome selectedRewrite,
            BranchDirectLaneOutcome selectedPreserve,
            BranchDirectLaneOutcome retainedPreserve,
            BranchDirectLaneOutcome weights) {
        return new BranchDirectRecordAudit(outcome, selectedRewrite, selectedPreserve,
                retainedPreserve, weights, retention, sourceSelected);
    }

    enum BranchDirectPairOutcome {
        RECORD_REJECT,
        EMPTY_READY,
        SELECTED_SOURCE_REJECT,
        SELECTED_KEY_REJECT,
        SELECTED_ROOT_REJECT,
        SELECTED_READY,
        RETAINED_ROOT_REJECT,
        RETAINED_READY
    }

    record BranchDirectPairAudit(
            BranchDirectPairOutcome outcome,
            BranchDirectLaneOutcome sourceProjection,
            BranchDirectLaneOutcome selectedKey,
            BranchDirectLaneOutcome root,
            BranchCandidateRetentionOutcome retention) {
    }

    /**
     * Ordered CPU mirror for pairing an aged register record with directly reprojected original
     * source-root provenance. Empty records intentionally carry no root.
     */
    static BranchDirectPairAudit branchDirectPairAudit(
            BranchCandidateRetentionOutcome retention,
            BranchDirectRecordOutcome recordOutcome,
            boolean selectedSourceReady,
            boolean selectedKeyReady,
            boolean rootReady,
            boolean rootChainValid,
            boolean rootIdentityValid) {
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct pair audit requires a live aged branch candidate");
        }
        if (recordOutcome == null) {
            throw new IllegalArgumentException("direct pair audit requires a record outcome");
        }
        if (recordOutcome == BranchDirectRecordOutcome.EMPTY_READY) {
            return branchDirectPairAuditResult(BranchDirectPairOutcome.EMPTY_READY,
                    retention, BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }
        if (recordOutcome != BranchDirectRecordOutcome.SELECTED_READY
                && recordOutcome != BranchDirectRecordOutcome.RETAINED_READY) {
            return branchDirectPairAuditResult(BranchDirectPairOutcome.RECORD_REJECT,
                    retention, BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }

        if (recordOutcome == BranchDirectRecordOutcome.SELECTED_READY) {
            if (!selectedSourceReady) {
                return branchDirectPairAuditResult(
                        BranchDirectPairOutcome.SELECTED_SOURCE_REJECT, retention,
                        BranchDirectLaneOutcome.REJECT,
                        BranchDirectLaneOutcome.NOT_ELIGIBLE,
                        BranchDirectLaneOutcome.NOT_ELIGIBLE);
            }
            if (!selectedKeyReady) {
                return branchDirectPairAuditResult(
                        BranchDirectPairOutcome.SELECTED_KEY_REJECT, retention,
                        BranchDirectLaneOutcome.READY,
                        BranchDirectLaneOutcome.REJECT,
                        BranchDirectLaneOutcome.NOT_ELIGIBLE);
            }
            boolean completeRoot = rootReady && rootChainValid && rootIdentityValid;
            return branchDirectPairAuditResult(
                    completeRoot ? BranchDirectPairOutcome.SELECTED_READY
                            : BranchDirectPairOutcome.SELECTED_ROOT_REJECT,
                    retention, BranchDirectLaneOutcome.READY,
                    BranchDirectLaneOutcome.READY,
                    completeRoot ? BranchDirectLaneOutcome.READY
                            : BranchDirectLaneOutcome.REJECT);
        }

        boolean completeRoot = rootReady && rootChainValid && rootIdentityValid;
        return branchDirectPairAuditResult(
                completeRoot ? BranchDirectPairOutcome.RETAINED_READY
                        : BranchDirectPairOutcome.RETAINED_ROOT_REJECT,
                retention, BranchDirectLaneOutcome.NOT_ELIGIBLE,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                completeRoot ? BranchDirectLaneOutcome.READY
                        : BranchDirectLaneOutcome.REJECT);
    }

    private static BranchDirectPairAudit branchDirectPairAuditResult(
            BranchDirectPairOutcome outcome,
            BranchCandidateRetentionOutcome retention,
            BranchDirectLaneOutcome sourceProjection,
            BranchDirectLaneOutcome selectedKey,
            BranchDirectLaneOutcome root) {
        return new BranchDirectPairAudit(
                outcome, sourceProjection, selectedKey, root, retention);
    }

    enum BranchDirectPairReplayOutcome {
        NOT_ELIGIBLE,
        SELECTED_ACCEPTED,
        SELECTED_REJECT,
        RETAINED_ACCEPTED,
        RETAINED_REJECT
    }

    enum BranchDirectPairReplayComparator {
        NONE,
        MAPPING_SOURCE,
        EXACT
    }

    record BranchDirectPairReplayAudit(
            BranchDirectPairReplayOutcome outcome,
            BranchDirectPairReplayComparator comparator,
            BranchCandidateRetentionOutcome retention,
            int segmentCount) {
    }

    /** CPU mirror for seeded replay of an aged register-only reservoir/root pair. */
    static BranchDirectPairReplayAudit branchDirectPairReplayAudit(
            BranchCandidateRetentionOutcome retention,
            BranchDirectPairOutcome pairOutcome,
            MappingKind mappingKind,
            int segmentCount,
            boolean mappingSourceReplayMatches,
            int exactReplayMask) {
        if (retention != BranchCandidateRetentionOutcome.AGE_ONE
                && retention != BranchCandidateRetentionOutcome.AGE_TWO
                && retention != BranchCandidateRetentionOutcome.AGE_THREE) {
            throw new IllegalArgumentException(
                    "direct pair replay requires a live aged branch candidate");
        }
        if (pairOutcome == null) {
            throw new IllegalArgumentException("direct pair replay requires a pair outcome");
        }
        if (pairOutcome != BranchDirectPairOutcome.SELECTED_READY
                && pairOutcome != BranchDirectPairOutcome.RETAINED_READY) {
            return new BranchDirectPairReplayAudit(
                    BranchDirectPairReplayOutcome.NOT_ELIGIBLE,
                    BranchDirectPairReplayComparator.NONE,
                    retention, 0);
        }
        if (mappingKind == null || (segmentCount != 1 && segmentCount != 2)) {
            throw new IllegalArgumentException(
                    "eligible direct pair replay requires mapping kind and one or two segments");
        }

        if (pairOutcome == BranchDirectPairOutcome.SELECTED_READY) {
            boolean accepted = mappingKind == MappingKind.DIFFUSE_RECONNECTION
                    && mappingSourceReplayMatches;
            return new BranchDirectPairReplayAudit(
                    accepted ? BranchDirectPairReplayOutcome.SELECTED_ACCEPTED
                            : BranchDirectPairReplayOutcome.SELECTED_REJECT,
                    BranchDirectPairReplayComparator.MAPPING_SOURCE,
                    retention, segmentCount);
        }

        boolean accepted = mappingKind == MappingKind.IDENTITY && exactReplayMask == 0;
        return new BranchDirectPairReplayAudit(
                accepted ? BranchDirectPairReplayOutcome.RETAINED_ACCEPTED
                        : BranchDirectPairReplayOutcome.RETAINED_REJECT,
                BranchDirectPairReplayComparator.EXACT,
                retention, segmentCount);
    }

    /** CPU mirror for bounded post-barrier storage of a replay-accepted aged pair. */
    static BranchAgedStorageOutcome branchAgedStorageOutcome(
            boolean metadataMatches, int age, MappingKind sourceMappingKind,
            MappingKind outputMappingKind, int segmentCount,
            boolean reservoirBitsMatch, boolean rootBitsMatch) {
        boolean metadataValid = metadataMatches
                && age >= 1 && age < 4
                && sourceMappingKind != null && outputMappingKind != null
                && (segmentCount == 1 || segmentCount == 2);
        if (!metadataValid) {
            return BranchAgedStorageOutcome.METADATA_REJECT;
        }
        if (!reservoirBitsMatch || !rootBitsMatch) {
            return BranchAgedStorageOutcome.PAIR_REJECT;
        }
        return outputMappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchAgedStorageOutcome.SELECTED_ACCEPTED
                : BranchAgedStorageOutcome.RETAINED_ACCEPTED;
    }

    record BranchReceiverOwnerClaim(int age, int linearIndex) {
        BranchReceiverOwnerClaim {
            if (age < 1 || age > 3) {
                throw new IllegalArgumentException("branch receiver owner age must be 1..3");
            }
            if (linearIndex < 0 || linearIndex > BRANCH_RECEIVER_OWNER_INDEX_MASK) {
                throw new IllegalArgumentException(
                        "branch receiver owner index must fit the four-slot ring");
            }
        }

        int priority() {
            return ((4 - age) << BRANCH_RECEIVER_OWNER_INDEX_BITS)
                    | (BRANCH_RECEIVER_OWNER_INDEX_MASK - linearIndex);
        }

        static BranchReceiverOwnerClaim fromPriority(int priority) {
            int ageBand = priority >>> BRANCH_RECEIVER_OWNER_INDEX_BITS;
            int age = 4 - ageBand;
            int linearIndex = BRANCH_RECEIVER_OWNER_INDEX_MASK
                    - (priority & BRANCH_RECEIVER_OWNER_INDEX_MASK);
            return new BranchReceiverOwnerClaim(age, linearIndex);
        }
    }

    /** Commutative CPU mirror of the shader's atomic-max receiver ownership policy. */
    static BranchReceiverOwnerClaim branchReceiverOwnerWinner(
            BranchReceiverOwnerClaim... claims) {
        if (claims == null || claims.length == 0) {
            throw new IllegalArgumentException("receiver ownership requires at least one claim");
        }
        BranchReceiverOwnerClaim winner = null;
        int winnerPriority = 0;
        for (BranchReceiverOwnerClaim claim : claims) {
            if (claim == null) {
                throw new IllegalArgumentException("receiver ownership claim must be present");
            }
            if (claim.priority() > winnerPriority) {
                winner = claim;
                winnerPriority = claim.priority();
            }
        }
        return BranchReceiverOwnerClaim.fromPriority(winnerPriority);
    }

    /** CPU mirror for the cleared full-resolution winner-pair scratch validator. */
    static BranchWinnerStorageOutcome branchWinnerStorageOutcome(
            int fanIn, boolean emptyTagCleared, boolean metadataValid,
            boolean reservoirBitsMatch, boolean rootBitsMatch,
            MappingKind outputMappingKind) {
        if (fanIn < 0) {
            throw new IllegalArgumentException("receiver fan-in must not be negative");
        }
        if (fanIn == 0) {
            return emptyTagCleared
                    ? BranchWinnerStorageOutcome.EMPTY
                    : BranchWinnerStorageOutcome.EMPTY_DIRTY_REJECT;
        }
        if (!metadataValid || outputMappingKind == null) {
            return BranchWinnerStorageOutcome.METADATA_REJECT;
        }
        if (!reservoirBitsMatch || !rootBitsMatch) {
            return BranchWinnerStorageOutcome.PAIR_REJECT;
        }
        return outputMappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchWinnerStorageOutcome.SELECTED_ACCEPTED
                : BranchWinnerStorageOutcome.RETAINED_ACCEPTED;
    }

    /**
     * Ordered CPU authority for the adjacent-frame winner replay diagnostic. The lifecycle gate is
     * owned by the independent ping-pong state; an accepted record is still diagnostic-only.
     */
    static BranchWinnerPreviousReplayOutcome branchWinnerPreviousReplayOutcome(
            boolean receiverReprojected, boolean previousPresent, boolean metadataValid,
            boolean receiverSurfaceValid, boolean sourceReprojected,
            boolean sourceSurfaceValid, boolean sourceReplayValid,
            MappingKind outputMappingKind) {
        if (!receiverReprojected) {
            return BranchWinnerPreviousReplayOutcome.RECEIVER_REPROJECTION_REJECT;
        }
        if (!previousPresent) {
            return BranchWinnerPreviousReplayOutcome.PREVIOUS_EMPTY;
        }
        if (!metadataValid) {
            return BranchWinnerPreviousReplayOutcome.METADATA_REJECT;
        }
        if (!receiverSurfaceValid) {
            return BranchWinnerPreviousReplayOutcome.RECEIVER_SURFACE_REJECT;
        }
        if (!sourceReprojected) {
            return BranchWinnerPreviousReplayOutcome.SOURCE_REPROJECTION_REJECT;
        }
        if (!sourceSurfaceValid) {
            return BranchWinnerPreviousReplayOutcome.SOURCE_SURFACE_REJECT;
        }
        if (!sourceReplayValid) {
            return BranchWinnerPreviousReplayOutcome.SOURCE_REPLAY_REJECT;
        }
        if (outputMappingKind == null) {
            throw new IllegalArgumentException(
                    "accepted winner replay requires an output mapping kind");
        }
        return outputMappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchWinnerPreviousReplayOutcome.SELECTED_ACCEPTED
                : BranchWinnerPreviousReplayOutcome.RETAINED_ACCEPTED;
    }

    enum BranchWinnerDirectRemapOutcome {
        PREVIOUS_REPLAY_REJECT,
        GUIDE_REJECT,
        EDGE_REJECT,
        GEOMETRY_REJECT,
        PDF_REJECT,
        THROUGHPUT_REJECT,
        READY
    }

    /** Ordered CPU mirror for the previous-winner original-root pre-visibility remap. */
    static BranchWinnerDirectRemapOutcome branchWinnerDirectRemapOutcome(
            BranchWinnerPreviousReplayOutcome replayOutcome,
            boolean receiverGuideValid, boolean sourceEdgeValid, boolean geometryValid,
            boolean pdfValid, boolean throughputValid) {
        boolean replayAccepted = replayOutcome
                == BranchWinnerPreviousReplayOutcome.SELECTED_ACCEPTED
                || replayOutcome == BranchWinnerPreviousReplayOutcome.RETAINED_ACCEPTED;
        if (!replayAccepted) {
            return BranchWinnerDirectRemapOutcome.PREVIOUS_REPLAY_REJECT;
        }
        if (!receiverGuideValid) {
            return BranchWinnerDirectRemapOutcome.GUIDE_REJECT;
        }
        if (!sourceEdgeValid) {
            return BranchWinnerDirectRemapOutcome.EDGE_REJECT;
        }
        if (!geometryValid) {
            return BranchWinnerDirectRemapOutcome.GEOMETRY_REJECT;
        }
        if (!pdfValid) {
            return BranchWinnerDirectRemapOutcome.PDF_REJECT;
        }
        if (!throughputValid) {
            return BranchWinnerDirectRemapOutcome.THROUGHPUT_REJECT;
        }
        return BranchWinnerDirectRemapOutcome.READY;
    }

    enum BranchWinnerDirectVisibilityOutcome {
        DIRECT_REMAP_REJECT,
        CLEAR,
        TINTED,
        OCCLUDED,
        INVALID
    }

    enum BranchWinnerDirectTargetOutcome {
        NOT_ELIGIBLE,
        POSITIVE,
        ZERO,
        INVALID
    }

    record BranchWinnerDirectTargetAudit(
            BranchWinnerDirectVisibilityOutcome visibility,
            BranchWinnerDirectTargetOutcome target) {
    }

    /** CPU mirror for production visibility and shifted target from a ready previous winner. */
    static BranchWinnerDirectTargetAudit branchWinnerDirectTargetAudit(
            boolean directRemapReady, boolean visibilityArithmeticValid,
            boolean anyTransmission, boolean fullyClear,
            boolean targetArithmeticValid, boolean positiveTarget) {
        if (!directRemapReady) {
            return new BranchWinnerDirectTargetAudit(
                    BranchWinnerDirectVisibilityOutcome.DIRECT_REMAP_REJECT,
                    BranchWinnerDirectTargetOutcome.NOT_ELIGIBLE);
        }
        if (!visibilityArithmeticValid) {
            return new BranchWinnerDirectTargetAudit(
                    BranchWinnerDirectVisibilityOutcome.INVALID,
                    BranchWinnerDirectTargetOutcome.NOT_ELIGIBLE);
        }
        BranchWinnerDirectVisibilityOutcome visibility = !anyTransmission
                ? BranchWinnerDirectVisibilityOutcome.OCCLUDED
                : fullyClear
                        ? BranchWinnerDirectVisibilityOutcome.CLEAR
                        : BranchWinnerDirectVisibilityOutcome.TINTED;
        BranchWinnerDirectTargetOutcome target = !targetArithmeticValid
                ? BranchWinnerDirectTargetOutcome.INVALID
                : positiveTarget
                        ? BranchWinnerDirectTargetOutcome.POSITIVE
                        : BranchWinnerDirectTargetOutcome.ZERO;
        return new BranchWinnerDirectTargetAudit(visibility, target);
    }

    enum BranchWinnerDirectWeightOutcome {
        TARGET_REJECT,
        POSITIVE,
        ZERO,
        INVALID
    }

    record BranchWinnerDirectWeightAudit(
            BranchWinnerDirectWeightOutcome weight,
            BranchDirectSourceCountOutcome sourceCount,
            double mergeWeight) {
    }

    /**
     * CPU mirror for the previous-winner GRIS weight. No previous mapping Jacobian is an input:
     * callers must provide only the directly recomputed original-root-to-current-receiver value.
     */
    static BranchWinnerDirectWeightAudit branchWinnerDirectWeightAudit(
            boolean targetReady, double shiftedTarget, double sourceFinalWeight,
            double sourceEffectiveCount, double directPssJacobian) {
        if (!targetReady) {
            return new BranchWinnerDirectWeightAudit(
                    BranchWinnerDirectWeightOutcome.TARGET_REJECT,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, 0.0);
        }
        boolean termsValid = nonNegativeFinite(shiftedTarget)
                && nonNegativeFinite(sourceFinalWeight)
                && nonNegativeFinite(sourceEffectiveCount)
                && positiveFinite(directPssJacobian);
        if (!termsValid) {
            return new BranchWinnerDirectWeightAudit(BranchWinnerDirectWeightOutcome.INVALID,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, 0.0);
        }
        double sourceCount = Math.min(sourceEffectiveCount, MAX_SPATIAL_SOURCE_COUNT);
        double mergeWeight = shiftedTarget * sourceFinalWeight;
        mergeWeight *= sourceCount;
        mergeWeight *= directPssJacobian;
        if (!Double.isFinite(mergeWeight) || mergeWeight < 0.0) {
            return new BranchWinnerDirectWeightAudit(BranchWinnerDirectWeightOutcome.INVALID,
                    BranchDirectSourceCountOutcome.NOT_ELIGIBLE, 0.0);
        }
        return new BranchWinnerDirectWeightAudit(
                mergeWeight > 0.0
                        ? BranchWinnerDirectWeightOutcome.POSITIVE
                        : BranchWinnerDirectWeightOutcome.ZERO,
                sourceEffectiveCount > MAX_SPATIAL_SOURCE_COUNT
                        ? BranchDirectSourceCountOutcome.CAPPED
                        : BranchDirectSourceCountOutcome.UNCAPPED,
                mergeWeight);
    }

    enum BranchWinnerDirectSelectionOutcome {
        WEIGHT_REJECT,
        CURRENT_REJECT,
        ZERO,
        OPEN,
        ONE,
        INVALID
    }

    record BranchWinnerDirectSelectionAudit(
            BranchWinnerDirectSelectionOutcome selection,
            BranchDirectCurrentWeightOutcome currentWeight,
            double probability) {
    }

    /** CPU authority for winner selection using the uncapped overflow-stable relative ratio. */
    static BranchWinnerDirectSelectionAudit branchWinnerDirectSelectionAudit(
            BranchWinnerDirectWeightOutcome weightOutcome,
            double currentWeightSum, double mergeWeight) {
        boolean weightReady = weightOutcome == BranchWinnerDirectWeightOutcome.POSITIVE
                || weightOutcome == BranchWinnerDirectWeightOutcome.ZERO;
        if (!weightReady) {
            return new BranchWinnerDirectSelectionAudit(
                    BranchWinnerDirectSelectionOutcome.WEIGHT_REJECT,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, 0.0);
        }
        if (!nonNegativeFinite(currentWeightSum)) {
            return new BranchWinnerDirectSelectionAudit(
                    BranchWinnerDirectSelectionOutcome.CURRENT_REJECT,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, 0.0);
        }
        if (!nonNegativeFinite(mergeWeight)) {
            return new BranchWinnerDirectSelectionAudit(
                    BranchWinnerDirectSelectionOutcome.INVALID,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, 0.0);
        }

        double probability;
        if (mergeWeight == 0.0) {
            probability = 0.0;
        } else if (currentWeightSum <= mergeWeight) {
            probability = 1.0 / (1.0 + currentWeightSum / mergeWeight);
        } else {
            double ratio = mergeWeight / currentWeightSum;
            probability = ratio / (1.0 + ratio);
        }
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            return new BranchWinnerDirectSelectionAudit(
                    BranchWinnerDirectSelectionOutcome.INVALID,
                    BranchDirectCurrentWeightOutcome.NOT_ELIGIBLE, 0.0);
        }
        return new BranchWinnerDirectSelectionAudit(
                probability == 0.0
                        ? BranchWinnerDirectSelectionOutcome.ZERO
                        : probability == 1.0
                                ? BranchWinnerDirectSelectionOutcome.ONE
                                : BranchWinnerDirectSelectionOutcome.OPEN,
                currentWeightSum == 0.0
                        ? BranchDirectCurrentWeightOutcome.ZERO
                        : BranchDirectCurrentWeightOutcome.POSITIVE,
                probability);
    }

    record BranchWinnerDirectBernoulliAudit(
            BranchDirectBernoulliOutcome outcome,
            BranchDirectBernoulliBoundary boundary,
            double random,
            boolean zeroBoundaryViolation,
            boolean oneBoundaryViolation) {
    }

    /** CPU mirror for the winner-owned deterministic diagnostic draw without reservoir mutation. */
    static BranchWinnerDirectBernoulliAudit branchWinnerDirectBernoulliAudit(
            BranchWinnerDirectSelectionAudit selectionAudit,
            int receiverPixelIndex, int previousPixelIndex,
            int winnerTagFrame, int winnerPriority, int frameIndex) {
        boolean selectionReady = selectionAudit.selection()
                == BranchWinnerDirectSelectionOutcome.ZERO
                || selectionAudit.selection() == BranchWinnerDirectSelectionOutcome.OPEN
                || selectionAudit.selection() == BranchWinnerDirectSelectionOutcome.ONE;
        if (!selectionReady) {
            return new BranchWinnerDirectBernoulliAudit(
                    BranchDirectBernoulliOutcome.SELECTION_REJECT,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, 0.0, false, false);
        }
        double probability = selectionAudit.probability();
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
            return new BranchWinnerDirectBernoulliAudit(BranchDirectBernoulliOutcome.INVALID,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, 0.0, false, false);
        }

        int mixedSeed = receiverPixelIndex
                ^ previousPixelIndex * BRANCH_BERNOULLI_ENTRY_MULTIPLIER
                ^ winnerTagFrame * BRANCH_WINNER_BERNOULLI_TAG_FRAME_MULTIPLIER
                ^ winnerPriority * BRANCH_WINNER_BERNOULLI_PRIORITY_MULTIPLIER
                ^ frameIndex * BRANCH_BERNOULLI_FRAME_MULTIPLIER
                ^ BRANCH_WINNER_BERNOULLI_SALT;
        int hashedSeed = RtPathReplayReference.pathHash(mixedSeed);
        double random = (hashedSeed >>> 8) * (1.0 / 16_777_216.0);
        if (!Double.isFinite(random) || random < 0.0 || random >= 1.0) {
            return new BranchWinnerDirectBernoulliAudit(BranchDirectBernoulliOutcome.INVALID,
                    BranchDirectBernoulliBoundary.NOT_ELIGIBLE, 0.0, false, false);
        }

        boolean selected = random < probability;
        BranchDirectBernoulliBoundary boundary = probability == 0.0
                ? BranchDirectBernoulliBoundary.ZERO
                : probability == 1.0
                        ? BranchDirectBernoulliBoundary.ONE
                        : BranchDirectBernoulliBoundary.OPEN;
        return new BranchWinnerDirectBernoulliAudit(
                selected ? BranchDirectBernoulliOutcome.SELECTED
                        : BranchDirectBernoulliOutcome.RETAINED,
                boundary, random,
                probability == 0.0 && selected,
                probability == 1.0 && !selected);
    }

    record BranchWinnerDirectPostSelectionAudit(
            BranchDirectPostSelectionOutcome outcome,
            BranchDirectStoredCapOutcome weightCap,
            BranchDirectStoredCapOutcome countCap,
            double nextWeightSum,
            double nextEffectiveCount,
            double selectedTarget,
            double finalWeight,
            boolean sourceSelected,
            boolean empty) {
    }

    /** CPU authority for winner post-selection arithmetic without constructing a reservoir. */
    static BranchWinnerDirectPostSelectionAudit branchWinnerDirectPostSelectionAudit(
            BranchDirectBernoulliOutcome bernoulliOutcome,
            double currentWeightSum, double currentEffectiveCount, double currentTarget,
            double mergeWeight, double sourceEffectiveCount, double shiftedTarget) {
        boolean bernoulliReady = bernoulliOutcome == BranchDirectBernoulliOutcome.SELECTED
                || bernoulliOutcome == BranchDirectBernoulliOutcome.RETAINED;
        boolean sourceSelected = bernoulliOutcome == BranchDirectBernoulliOutcome.SELECTED;
        if (!bernoulliReady) {
            return branchWinnerDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.BERNOULLI_REJECT, sourceSelected);
        }
        if (!nonNegativeFinite(currentWeightSum)
                || !nonNegativeFinite(currentEffectiveCount)
                || !nonNegativeFinite(currentTarget)) {
            return branchWinnerDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.CURRENT_REJECT, sourceSelected);
        }
        if (!nonNegativeFinite(mergeWeight)
                || !nonNegativeFinite(sourceEffectiveCount)
                || !nonNegativeFinite(shiftedTarget)) {
            return branchWinnerDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.NEXT_REJECT, sourceSelected);
        }

        double sourceCount = Math.min(sourceEffectiveCount, MAX_SPATIAL_SOURCE_COUNT);
        boolean weightCapped = currentWeightSum > MAX_STORED_WEIGHT_SUM - mergeWeight;
        boolean countCapped = currentEffectiveCount
                > MAX_STORED_EFFECTIVE_COUNT - sourceCount;
        double nextWeightSum = weightCapped
                ? MAX_STORED_WEIGHT_SUM : currentWeightSum + mergeWeight;
        double nextEffectiveCount = countCapped
                ? MAX_STORED_EFFECTIVE_COUNT : currentEffectiveCount + sourceCount;
        if (!nonNegativeFinite(nextWeightSum) || !nonNegativeFinite(nextEffectiveCount)) {
            return branchWinnerDirectPostSelectionReject(
                    BranchDirectPostSelectionOutcome.NEXT_REJECT, sourceSelected);
        }

        BranchDirectStoredCapOutcome weightCap = weightCapped
                ? BranchDirectStoredCapOutcome.CAPPED
                : BranchDirectStoredCapOutcome.UNCAPPED;
        BranchDirectStoredCapOutcome countCap = countCapped
                ? BranchDirectStoredCapOutcome.CAPPED
                : BranchDirectStoredCapOutcome.UNCAPPED;
        double selectedTarget = sourceSelected ? shiftedTarget : currentTarget;
        boolean empty = nextWeightSum == 0.0;
        if (!nonNegativeFinite(selectedTarget) || (!empty && selectedTarget == 0.0)) {
            return new BranchWinnerDirectPostSelectionAudit(
                    sourceSelected
                            ? BranchDirectPostSelectionOutcome.SELECTED_TARGET_REJECT
                            : BranchDirectPostSelectionOutcome.RETAINED_TARGET_REJECT,
                    weightCap, countCap, nextWeightSum, nextEffectiveCount,
                    selectedTarget, 0.0, sourceSelected, false);
        }

        double finalWeight = 0.0;
        if (!empty) {
            if (nextEffectiveCount == 0.0) {
                return new BranchWinnerDirectPostSelectionAudit(
                        sourceSelected
                                ? BranchDirectPostSelectionOutcome.SELECTED_FINAL_REJECT
                                : BranchDirectPostSelectionOutcome.RETAINED_FINAL_REJECT,
                        weightCap, countCap, nextWeightSum, nextEffectiveCount,
                        selectedTarget, 0.0, sourceSelected, false);
            }
            double denominator = nextEffectiveCount * selectedTarget;
            finalWeight = nextWeightSum / denominator;
            if (!positiveFinite(denominator) || !positiveFinite(finalWeight)) {
                return new BranchWinnerDirectPostSelectionAudit(
                        sourceSelected
                                ? BranchDirectPostSelectionOutcome.SELECTED_FINAL_REJECT
                                : BranchDirectPostSelectionOutcome.RETAINED_FINAL_REJECT,
                        weightCap, countCap, nextWeightSum, nextEffectiveCount,
                        selectedTarget, 0.0, sourceSelected, false);
            }
        }
        return new BranchWinnerDirectPostSelectionAudit(
                sourceSelected
                        ? BranchDirectPostSelectionOutcome.SELECTED_READY
                        : BranchDirectPostSelectionOutcome.RETAINED_READY,
                weightCap, countCap, nextWeightSum, nextEffectiveCount,
                selectedTarget, finalWeight, sourceSelected, empty);
    }

    private static BranchWinnerDirectPostSelectionAudit branchWinnerDirectPostSelectionReject(
            BranchDirectPostSelectionOutcome outcome, boolean sourceSelected) {
        return new BranchWinnerDirectPostSelectionAudit(outcome,
                BranchDirectStoredCapOutcome.NOT_ELIGIBLE,
                BranchDirectStoredCapOutcome.NOT_ELIGIBLE,
                0.0, 0.0, 0.0, 0.0, sourceSelected, false);
    }

    record BranchWinnerDirectRecordAudit(
            BranchDirectRecordOutcome outcome,
            BranchDirectLaneOutcome selectedRewrite,
            BranchDirectLaneOutcome selectedPreserve,
            BranchDirectLaneOutcome retainedPreserve,
            BranchDirectLaneOutcome weights,
            BranchDirectLaneOutcome metadata,
            BranchDirectLaneOutcome sourceKey,
            boolean sourceSelected) {
    }

    /** CPU mirror for full previous-winner reservoir assembly without root pairing or storage. */
    static BranchWinnerDirectRecordAudit branchWinnerDirectRecordAudit(
            BranchDirectPostSelectionOutcome postSelectionOutcome,
            boolean empty,
            boolean selectedRewriteReady,
            boolean selectedPreserveReady,
            boolean retainedPreserveReady,
            boolean weightsReady,
            boolean metadataReady,
            boolean sourceKeyReady) {
        boolean postSelectionReady = postSelectionOutcome
                == BranchDirectPostSelectionOutcome.SELECTED_READY
                || postSelectionOutcome == BranchDirectPostSelectionOutcome.RETAINED_READY;
        boolean sourceSelected = postSelectionOutcome
                == BranchDirectPostSelectionOutcome.SELECTED_READY;
        if (!postSelectionReady) {
            return branchWinnerDirectRecordAuditResult(
                    BranchDirectRecordOutcome.POST_SELECTION_REJECT, sourceSelected,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }
        if (empty) {
            return branchWinnerDirectRecordAuditResult(
                    sourceSelected
                            ? BranchDirectRecordOutcome.SELECTED_REJECT
                            : BranchDirectRecordOutcome.EMPTY_READY,
                    sourceSelected,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }

        BranchDirectLaneOutcome weightLane = weightsReady
                ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
        BranchDirectLaneOutcome metadataLane = metadataReady
                ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
        if (sourceSelected) {
            BranchDirectLaneOutcome rewriteLane = selectedRewriteReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            BranchDirectLaneOutcome preserveLane = selectedPreserveReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            BranchDirectLaneOutcome sourceKeyLane = sourceKeyReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            boolean ready = selectedRewriteReady && selectedPreserveReady
                    && weightsReady && metadataReady && sourceKeyReady;
            return branchWinnerDirectRecordAuditResult(
                    ready ? BranchDirectRecordOutcome.SELECTED_READY
                            : BranchDirectRecordOutcome.SELECTED_REJECT,
                    true, rewriteLane, preserveLane,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    weightLane, metadataLane, sourceKeyLane);
        }

        BranchDirectLaneOutcome retainedLane = retainedPreserveReady
                ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
        boolean ready = retainedPreserveReady && weightsReady && metadataReady;
        return branchWinnerDirectRecordAuditResult(
                ready ? BranchDirectRecordOutcome.RETAINED_READY
                        : BranchDirectRecordOutcome.RETAINED_REJECT,
                false,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                retainedLane, weightLane, metadataLane,
                BranchDirectLaneOutcome.NOT_ELIGIBLE);
    }

    private static BranchWinnerDirectRecordAudit branchWinnerDirectRecordAuditResult(
            BranchDirectRecordOutcome outcome,
            boolean sourceSelected,
            BranchDirectLaneOutcome selectedRewrite,
            BranchDirectLaneOutcome selectedPreserve,
            BranchDirectLaneOutcome retainedPreserve,
            BranchDirectLaneOutcome weights,
            BranchDirectLaneOutcome metadata,
            BranchDirectLaneOutcome sourceKey) {
        return new BranchWinnerDirectRecordAudit(outcome, selectedRewrite, selectedPreserve,
                retainedPreserve, weights, metadata, sourceKey, sourceSelected);
    }

    record BranchWinnerDirectPairAudit(
            BranchDirectPairOutcome outcome,
            BranchDirectLaneOutcome selectedKey,
            BranchDirectLaneOutcome selectedRoot,
            BranchDirectLaneOutcome retainedRoot) {
    }

    /** CPU mirror for current-frame root pairing after the previous-winner record gate. */
    static BranchWinnerDirectPairAudit branchWinnerDirectPairAudit(
            BranchDirectRecordOutcome recordOutcome,
            boolean selectedKeyReady,
            boolean selectedRootReady,
            boolean retainedRootReady,
            boolean rootChainValid,
            boolean rootIdentityValid) {
        if (recordOutcome == null) {
            throw new IllegalArgumentException("winner pair audit requires a record outcome");
        }
        if (recordOutcome == BranchDirectRecordOutcome.EMPTY_READY) {
            return branchWinnerDirectPairAuditResult(BranchDirectPairOutcome.EMPTY_READY,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }
        if (recordOutcome != BranchDirectRecordOutcome.SELECTED_READY
                && recordOutcome != BranchDirectRecordOutcome.RETAINED_READY) {
            return branchWinnerDirectPairAuditResult(BranchDirectPairOutcome.RECORD_REJECT,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }

        if (recordOutcome == BranchDirectRecordOutcome.SELECTED_READY) {
            BranchDirectLaneOutcome key = selectedKeyReady
                    ? BranchDirectLaneOutcome.READY : BranchDirectLaneOutcome.REJECT;
            if (!selectedKeyReady) {
                return branchWinnerDirectPairAuditResult(
                        BranchDirectPairOutcome.SELECTED_KEY_REJECT, key,
                        BranchDirectLaneOutcome.NOT_ELIGIBLE,
                        BranchDirectLaneOutcome.NOT_ELIGIBLE);
            }
            boolean completeRoot = selectedRootReady && rootChainValid && rootIdentityValid;
            return branchWinnerDirectPairAuditResult(
                    completeRoot ? BranchDirectPairOutcome.SELECTED_READY
                            : BranchDirectPairOutcome.SELECTED_ROOT_REJECT,
                    key, completeRoot ? BranchDirectLaneOutcome.READY
                            : BranchDirectLaneOutcome.REJECT,
                    BranchDirectLaneOutcome.NOT_ELIGIBLE);
        }

        boolean completeRoot = retainedRootReady && rootChainValid && rootIdentityValid;
        return branchWinnerDirectPairAuditResult(
                completeRoot ? BranchDirectPairOutcome.RETAINED_READY
                        : BranchDirectPairOutcome.RETAINED_ROOT_REJECT,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                BranchDirectLaneOutcome.NOT_ELIGIBLE,
                completeRoot ? BranchDirectLaneOutcome.READY
                        : BranchDirectLaneOutcome.REJECT);
    }

    private static BranchWinnerDirectPairAudit branchWinnerDirectPairAuditResult(
            BranchDirectPairOutcome outcome,
            BranchDirectLaneOutcome selectedKey,
            BranchDirectLaneOutcome selectedRoot,
            BranchDirectLaneOutcome retainedRoot) {
        return new BranchWinnerDirectPairAudit(
                outcome, selectedKey, selectedRoot, retainedRoot);
    }

    record BranchWinnerDirectPairReplayAudit(
            BranchDirectPairReplayOutcome outcome,
            BranchDirectPairReplayComparator comparator,
            int segmentCount) {
    }

    /** CPU mirror for seeded replay of a current-frame previous-winner pair. */
    static BranchWinnerDirectPairReplayAudit branchWinnerDirectPairReplayAudit(
            BranchDirectPairOutcome pairOutcome,
            MappingKind mappingKind,
            int segmentCount,
            boolean mappingSourceReplayMatches,
            int exactReplayMask) {
        if (pairOutcome == null) {
            throw new IllegalArgumentException("winner pair replay requires a pair outcome");
        }
        if (pairOutcome != BranchDirectPairOutcome.SELECTED_READY
                && pairOutcome != BranchDirectPairOutcome.RETAINED_READY) {
            return new BranchWinnerDirectPairReplayAudit(
                    BranchDirectPairReplayOutcome.NOT_ELIGIBLE,
                    BranchDirectPairReplayComparator.NONE, 0);
        }
        if (mappingKind == null || (segmentCount != 1 && segmentCount != 2)) {
            throw new IllegalArgumentException(
                    "eligible winner pair replay requires mapping kind and one or two segments");
        }

        if (pairOutcome == BranchDirectPairOutcome.SELECTED_READY) {
            boolean accepted = mappingKind == MappingKind.DIFFUSE_RECONNECTION
                    && mappingSourceReplayMatches;
            return new BranchWinnerDirectPairReplayAudit(
                    accepted ? BranchDirectPairReplayOutcome.SELECTED_ACCEPTED
                            : BranchDirectPairReplayOutcome.SELECTED_REJECT,
                    BranchDirectPairReplayComparator.MAPPING_SOURCE, segmentCount);
        }

        boolean accepted = mappingKind == MappingKind.IDENTITY && exactReplayMask == 0;
        return new BranchWinnerDirectPairReplayAudit(
                accepted ? BranchDirectPairReplayOutcome.RETAINED_ACCEPTED
                        : BranchDirectPairReplayOutcome.RETAINED_REJECT,
                BranchDirectPairReplayComparator.EXACT, segmentCount);
    }

    /** CPU mirror for bounded post-barrier storage of a replay-approved winner pair. */
    static BranchAgedStorageOutcome branchWinnerPairStorageOutcome(
            boolean metadataMatches,
            boolean receiverIndexValid,
            MappingKind sourceMappingKind,
            MappingKind outputMappingKind,
            MappingKind previousOutputMappingKind,
            int segmentCount,
            boolean reservoirBitsMatch,
            boolean rootBitsMatch) {
        boolean metadataValid = metadataMatches && receiverIndexValid
                && sourceMappingKind != null
                && outputMappingKind != null
                && previousOutputMappingKind != null
                && (segmentCount == 1 || segmentCount == 2);
        if (!metadataValid) {
            return BranchAgedStorageOutcome.METADATA_REJECT;
        }
        if (!reservoirBitsMatch || !rootBitsMatch) {
            return BranchAgedStorageOutcome.PAIR_REJECT;
        }
        return outputMappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchAgedStorageOutcome.SELECTED_ACCEPTED
                : BranchAgedStorageOutcome.RETAINED_ACCEPTED;
    }

    /**
     * CPU mirror for the temporary dense candidate-owner tag. The receiver index is the unique
     * storage owner; all provenance fields must be explicit before payload can be considered.
     */
    static BranchWinnerCandidateOwnerOutcome branchWinnerCandidateOwnerOutcome(
            boolean empty, boolean metadataMatches, boolean receiverIndexMatches,
            MappingKind sourceMappingKind, MappingKind outputMappingKind,
            MappingKind previousOutputMappingKind, int segmentCount) {
        if (empty) {
            return BranchWinnerCandidateOwnerOutcome.EMPTY;
        }
        boolean metadataValid = metadataMatches && receiverIndexMatches
                && sourceMappingKind != null
                && outputMappingKind != null
                && previousOutputMappingKind != null
                && (segmentCount == 1 || segmentCount == 2);
        if (!metadataValid) {
            return BranchWinnerCandidateOwnerOutcome.METADATA_REJECT;
        }
        return outputMappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchWinnerCandidateOwnerOutcome.SELECTED_ACCEPTED
                : BranchWinnerCandidateOwnerOutcome.RETAINED_ACCEPTED;
    }

    /** CPU mirror for bounded exact validation of the receiver-indexed dense payload. */
    static BranchAgedStorageOutcome branchWinnerDensePayloadOutcome(
            boolean ownerTagMatches,
            boolean receiverIndexValid,
            MappingKind sourceMappingKind,
            MappingKind outputMappingKind,
            MappingKind previousOutputMappingKind,
            int segmentCount,
            boolean reservoirBitsMatch,
            boolean rootBitsMatch) {
        return branchWinnerPairStorageOutcome(
                ownerTagMatches, receiverIndexValid,
                sourceMappingKind, outputMappingKind, previousOutputMappingKind,
                segmentCount, reservoirBitsMatch, rootBitsMatch);
    }

    /**
     * Ordered CPU mirror for the post-barrier exact-lane validation sample. Reservoir and root
     * equality are bitwise requirements; acceptance remains diagnostic-only.
     */
    static BranchScratchStorageOutcome branchScratchStorageOutcome(
            boolean metadataValid, boolean reservoirBitsMatch, boolean rootBitsMatch,
            MappingKind mappingKind) {
        if (!metadataValid) {
            return BranchScratchStorageOutcome.METADATA_REJECT;
        }
        if (!reservoirBitsMatch || !rootBitsMatch) {
            return BranchScratchStorageOutcome.PAIR_REJECT;
        }
        if (mappingKind == null) {
            throw new IllegalArgumentException(
                    "accepted branch scratch sample requires a mapping kind");
        }
        return mappingKind == MappingKind.DIFFUSE_RECONNECTION
                ? BranchScratchStorageOutcome.SELECTED_ACCEPTED
                : BranchScratchStorageOutcome.RETAINED_ACCEPTED;
    }

    record BranchCandidateTag(int frameIndex, int control) {
        static BranchCandidateTag capture(
                int frameIndex, int generation, MappingKind mappingKind) {
            if (mappingKind == null) {
                throw new IllegalArgumentException(
                        "branch candidate tag requires a mapping kind");
            }
            int control = BRANCH_CANDIDATE_TAG_VALID_BIT
                    | ((mappingKind.code() & MAPPING_KIND_MASK)
                            << BRANCH_CANDIDATE_TAG_MAPPING_SHIFT)
                    | (generation & BRANCH_CANDIDATE_TAG_GENERATION_MASK);
            return new BranchCandidateTag(frameIndex, control);
        }

        boolean validFor(int currentFrameIndex, int currentGeneration,
                         MappingKind mappingKind) {
            return equals(capture(currentFrameIndex - 1, currentGeneration, mappingKind));
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

    /** Ordered fail-closed receiver policy mirrored by the cross-frame GPU diagnostic counters. */
    enum CrossFrameReceiverReject {
        NONE,
        SURFACE,
        SAMPLE,
        EDGE,
        TOPOLOGY,
        DEPTH,
        TRANSPORT,
        FOOTPRINT
    }

    record CrossFrameReceiverPolicy(boolean surfaceCompatible,
                                    boolean sampleUsable,
                                    boolean edgeAvailable,
                                    boolean topologyMatches,
                                    boolean depthMatches,
                                    boolean transportMatches,
                                    boolean footprintCompatible) {
        CrossFrameReceiverReject firstReject() {
            if (!surfaceCompatible) return CrossFrameReceiverReject.SURFACE;
            if (!sampleUsable) return CrossFrameReceiverReject.SAMPLE;
            return firstRejectAfterSample();
        }

        boolean sampleRescueEligible() {
            return surfaceCompatible && !sampleUsable;
        }

        CrossFrameReceiverReject sampleRescueReject() {
            if (!sampleRescueEligible()) return firstReject();
            return firstRejectAfterSample();
        }

        boolean sampleRescued() {
            return sampleRescueEligible()
                    && sampleRescueReject() == CrossFrameReceiverReject.NONE;
        }

        private CrossFrameReceiverReject firstRejectAfterSample() {
            if (!edgeAvailable) return CrossFrameReceiverReject.EDGE;
            if (!topologyMatches) return CrossFrameReceiverReject.TOPOLOGY;
            if (!depthMatches) return CrossFrameReceiverReject.DEPTH;
            if (!transportMatches) return CrossFrameReceiverReject.TRANSPORT;
            if (!footprintCompatible) return CrossFrameReceiverReject.FOOTPRINT;
            return CrossFrameReceiverReject.NONE;
        }
    }

    enum CrossFrameReceiverEdgeReject {
        NONE,
        MISSING_VALID,
        DEPTH,
        EVENT,
        MAPPING,
        PDF,
        FINITE
    }

    record CrossFrameReceiverEdgePolicy(boolean valid,
                                        boolean depthMatches,
                                        boolean eventMatches,
                                        boolean mappingMatches,
                                        boolean pdfPositive,
                                        boolean vertexThroughputFinite) {
        CrossFrameReceiverEdgeReject firstReject() {
            if (!valid) return CrossFrameReceiverEdgeReject.MISSING_VALID;
            if (!depthMatches) return CrossFrameReceiverEdgeReject.DEPTH;
            if (!eventMatches) return CrossFrameReceiverEdgeReject.EVENT;
            if (!mappingMatches) return CrossFrameReceiverEdgeReject.MAPPING;
            if (!pdfPositive) return CrossFrameReceiverEdgeReject.PDF;
            if (!vertexThroughputFinite) return CrossFrameReceiverEdgeReject.FINITE;
            return CrossFrameReceiverEdgeReject.NONE;
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
