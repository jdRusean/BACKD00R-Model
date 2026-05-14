import java.util.List;

public class SourceFile {
    private final String fileName;
    private final String path;
    private final int linesOfCode;
    private final int methodCount;
    private final int fieldCount;
    private final int cyclomaticComplexity;
    private final int externalDataAccessCount;
    private final String lastEditedBy;
    private final List<String> methodNames;

    public SourceFile(
            String fileName,
            String path,
            int linesOfCode,
            int methodCount,
            int fieldCount,
            int cyclomaticComplexity,
            int externalDataAccessCount,
            String lastEditedBy,
            List<String> methodNames
    ) {
        this.fileName = fileName;
        this.path = path;
        this.linesOfCode = linesOfCode;
        this.methodCount = methodCount;
        this.fieldCount = fieldCount;
        this.cyclomaticComplexity = cyclomaticComplexity;
        this.externalDataAccessCount = externalDataAccessCount;
        this.lastEditedBy = lastEditedBy;
        this.methodNames = methodNames;
    }

    public String getFileName() {
        return fileName;
    }

    public String getPath() {
        return path;
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

    public int getExternalDataAccessCount() {
        return externalDataAccessCount;
    }

    public String getLastEditedBy() {
        return lastEditedBy;
    }

    public List<String> getMethodNames() {
        return methodNames;
    }

    public int getLongestMethodLength() {
        if (methodCount == 0) {
            return 0;
        }
        return linesOfCode / methodCount;
    }
}