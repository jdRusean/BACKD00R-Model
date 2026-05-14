public class MetricSnapshot {
    private final String fileName;
    private final int linesOfCode;
    private final int methodCount;
    private final int fieldCount;
    private final int cyclomaticComplexity;
    private final int riskScore;
    private final String recommendationPriority;

    public MetricSnapshot(
            String fileName,
            int linesOfCode,
            int methodCount,
            int fieldCount,
            int cyclomaticComplexity,
            int riskScore,
            String recommendationPriority
    ) {
        this.fileName = fileName;
        this.linesOfCode = linesOfCode;
        this.methodCount = methodCount;
        this.fieldCount = fieldCount;
        this.cyclomaticComplexity = cyclomaticComplexity;
        this.riskScore = riskScore;
        this.recommendationPriority = recommendationPriority;
    }

    public String getFileName() {
        return fileName;
    }

    public int getLinesOfCode() {
        return linesOfCode;
    }

    public int getMethodCount() {
        return methodCount;
    }

    public int getFieldCount() {
        return fieldCount;
    }

    public int getCyclomaticComplexity() {
        return cyclomaticComplexity;
    }

    public int getRiskScore() {
        return riskScore;
    }

    public String getRecommendationPriority() {
        return recommendationPriority;
    }
}