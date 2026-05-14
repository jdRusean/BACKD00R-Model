import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

 INTENTIONAL GOD CLASS
 This class has too many responsibilities
 1. Calculates metrics
 2. Detects code smells
 3. Generates recommendations
 4. Builds the output report
 5. Creates developer-specific messages
 6. Computes project health scores
public class ProjectAnalyzer {

    private final ListString detectedSmells = new ArrayList();
    private final ListString recommendations = new ArrayList();
    private int criticalCount = 0;
    private int warningCount = 0;
    private int infoCount = 0;

     INTENTIONAL LONG METHOD
     This method does many tasks that should be separated into smaller methods.
    public String analyzeProject(ListSourceFile files, DeveloperProfile developer) {
        StringBuilder report = new StringBuilder();
        MapString, Integer smellCountByFile = new HashMap();
        ListMetricSnapshot snapshots = new ArrayList();

        detectedSmells.clear();
        recommendations.clear();
        criticalCount = 0;
        warningCount = 0;
        infoCount = 0;

        report.append(MINI CODE QUALITY ANALYZER REPORTn);
        report.append(=================================nn);

        if (files == null  files.isEmpty()) {
            report.append(No source files were provided.n);
            return report.toString();
        }

        if (developer == null) {
            report.append(No developer profile was provided. Generic recommendations will be used.nn);
        } else {
            report.append(Developer ).append(developer.getName()).append(n);
            report.append(Experience Level ).append(developer.getExperienceLevel()).append(n);
            report.append(Warnings Reviewed ).append(developer.getWarningsReviewed()).append(n);
            report.append(Fixes Applied ).append(developer.getFixesApplied()).append(nn);
        }

        int totalLoc = 0;
        int totalMethods = 0;
        int totalComplexity = 0;
        int totalExternalAccess = 0;
        int highestLoc = 0;
        int highestComplexity = 0;
        String largestFile = None;
        String mostComplexFile = None;

        for (SourceFile file  files) {
            totalLoc += file.getLinesOfCode();
            totalMethods += file.getMethodCount();
            totalComplexity += file.getCyclomaticComplexity();
            totalExternalAccess += file.getExternalDataAccessCount();

            if (file.getLinesOfCode()  highestLoc) {
                highestLoc = file.getLinesOfCode();
                largestFile = file.getFileName();
            }

            if (file.getCyclomaticComplexity()  highestComplexity) {
                highestComplexity = file.getCyclomaticComplexity();
                mostComplexFile = file.getFileName();
            }
        }

        report.append(PROJECT SUMMARYn);
        report.append(---------------n);
        report.append(Total Files ).append(files.size()).append(n);
        report.append(Total LOC ).append(totalLoc).append(n);
        report.append(Total Methods ).append(totalMethods).append(n);
        report.append(Total Cyclomatic Complexity ).append(totalComplexity).append(n);
        report.append(Total External Data Access Count ).append(totalExternalAccess).append(n);
        report.append(Largest File ).append(largestFile).append( ().append(highestLoc).append( LOC)n);
        report.append(Most Complex File ).append(mostComplexFile).append( ().append(highestComplexity).append( complexity)nn);

        report.append(FILE ANALYSISn);
        report.append(-------------n);

        for (SourceFile file  files) {
            MetricSnapshot snapshot = calculateMetrics(file);
            snapshots.add(snapshot);
            int smellCount = 0;

            report.append(File ).append(file.getFileName()).append(n);
            report.append(Path ).append(file.getPath()).append(n);
            report.append(LOC ).append(file.getLinesOfCode()).append(n);
            report.append(Methods ).append(file.getMethodCount()).append(n);
            report.append(Fields ).append(file.getFieldCount()).append(n);
            report.append(Cyclomatic Complexity ).append(file.getCyclomaticComplexity()).append(n);
            report.append(External Data Access Count ).append(file.getExternalDataAccessCount()).append(n);
            report.append(Method Names ).append(file.getMethodNames()).append(n);

            if (file.getLinesOfCode()  300 && file.getMethodCount()  12 && file.getFieldCount()  8) {
                String smell = Possible God Class detected in  + file.getFileName();
                detectedSmells.add(smell);
                recommendations.add(Split  + file.getFileName() +  into smaller classes based on responsibility.);
                criticalCount++;
                smellCount++;
                report.append(Detected Smell God Classn);
            }

            if (file.getCyclomaticComplexity()  15  file.getLongestMethodLength()  30) {
                String smell = Possible Long Method detected in  + file.getFileName();
                detectedSmells.add(smell);
                recommendations.add(Apply Extract Method to reduce the longest method in  + file.getFileName() + .);
                warningCount++;
                smellCount++;
                report.append(Detected Smell Long Methodn);
            }

            if (file.getExternalDataAccessCount()  5) {
                String smell = Possible Feature Envy detected in  + file.getFileName();
                detectedSmells.add(smell);
                recommendations.add(Move behavior closer to the class whose data is being accessed by  + file.getFileName() + .);
                warningCount++;
                smellCount++;
                report.append(Detected Smell Feature Envyn);
            }

            if (snapshot.getRiskScore() = 80) {
                report.append(Risk Level Criticaln);
            } else if (snapshot.getRiskScore() = 50) {
                report.append(Risk Level Moderaten);
            } else {
                report.append(Risk Level Lown);
                infoCount++;
            }

            if (developer != null) {
                report.append(createPersonalizedDeveloperMessage(developer, file, snapshot));
            }

            smellCountByFile.put(file.getFileName(), smellCount);
            report.append(Estimated Risk Score ).append(snapshot.getRiskScore()).append(n);
            report.append(Recommendation Priority ).append(snapshot.getRecommendationPriority()).append(n);
            report.append(n);
        }

        int healthScore = 100;
        healthScore -= criticalCount  20;
        healthScore -= warningCount  10;
        healthScore -= infoCount  2;

        if (healthScore  0) {
            healthScore = 0;
        }

        report.append(DETECTED SMELLSn);
        report.append(---------------n);

        if (detectedSmells.isEmpty()) {
            report.append(No major smells detected.n);
        } else {
            for (String smell  detectedSmells) {
                report.append(- ).append(smell).append(n);
            }
        }

        report.append(nRECOMMENDATIONSn);
        report.append(---------------n);

        if (recommendations.isEmpty()) {
            report.append(No recommendations generated.n);
        } else {
            for (int i = 0; i  recommendations.size(); i++) {
                report.append(i + 1).append(. ).append(recommendations.get(i)).append(n);
            }
        }

        report.append(nSMELL COUNT BY FILEn);
        report.append(-------------------n);

        for (String fileName  smellCountByFile.keySet()) {
            report.append(fileName).append( ).append(smellCountByFile.get(fileName)).append( smell(s)n);
        }

        report.append(nPROJECT HEALTH SCOREn);
        report.append(--------------------n);
        report.append(Critical Findings ).append(criticalCount).append(n);
        report.append(Warnings ).append(warningCount).append(n);
        report.append(Informational Findings ).append(infoCount).append(n);
        report.append(Overall Health Score ).append(healthScore).append(100n);

        if (healthScore = 80) {
            report.append(Interpretation The project is generally maintainable.n);
        } else if (healthScore = 50) {
            report.append(Interpretation The project needs refactoring attention.n);
        } else {
            report.append(Interpretation The project has serious maintainability risks.n);
        }

        return report.toString();
    }

    private MetricSnapshot calculateMetrics(SourceFile file) {
        int riskScore = 0;

        riskScore += file.getLinesOfCode()  10;
        riskScore += file.getMethodCount()  2;
        riskScore += file.getFieldCount()  3;
        riskScore += file.getCyclomaticComplexity()  2;
        riskScore += file.getExternalDataAccessCount()  4;

        if (riskScore  100) {
            riskScore = 100;
        }

        String priority;
        if (riskScore = 80) {
            priority = High;
        } else if (riskScore = 50) {
            priority = Medium;
        } else {
            priority = Low;
        }

        return new MetricSnapshot(
                file.getFileName(),
                file.getLinesOfCode(),
                file.getMethodCount(),
                file.getFieldCount(),
                file.getCyclomaticComplexity(),
                riskScore,
                priority
        );
    }

     INTENTIONAL FEATURE ENVY
     This method uses too much data from DeveloperProfile and SourceFile.
     It should probably belong partly inside DeveloperProfile, SourceFile, or a separate advisor class.
    private String createPersonalizedDeveloperMessage(
            DeveloperProfile developer,
            SourceFile file,
            MetricSnapshot snapshot
    ) {
        StringBuilder message = new StringBuilder();

        message.append(Developer-Specific Note );

        if (developer.getName().equals(file.getLastEditedBy())) {
            message.append(developer.getName())
                    .append( recently edited )
                    .append(file.getFileName())
                    .append(, so this file may be familiar for refactoring. );
        }

        if (developer.getExperienceLevel().equalsIgnoreCase(Junior)
                && file.getCyclomaticComplexity()  10
                && file.getLongestMethodLength()  25) {
            message.append(Because the developer is junior and )
                    .append(file.getFileName())
                    .append( has high complexity, pair review is recommended. );
        }

        if (developer.getExperienceLevel().equalsIgnoreCase(Intermediate)
                && developer.getWarningsReviewed()  10
                && developer.getFixesApplied() = 5
                && developer.getPreferredRefactorings().contains(Extract Method)) {
            message.append(The developer has enough review history to attempt Extract Method on )
                    .append(file.getFileName())
                    .append(. );
        }

        if (developer.getRecentFilesEdited().contains(file.getFileName())
                && file.getExternalDataAccessCount()  5
                && developer.getPreferredRefactorings().contains(Move Method)) {
            message.append(Move Method may be suitable because )
                    .append(file.getFileName())
                    .append( frequently accesses external data. );
        }

        if (file.getLinesOfCode()  300
                && file.getMethodCount()  12
                && file.getFieldCount()  8
                && developer.getPreferredRefactorings().contains(Extract Class)) {
            message.append(Extract Class is recommended because )
                    .append(file.getFileName())
                    .append( is large and has many methods and fields. );
        }

        if (snapshot.getRiskScore() = 80
                && developer.getWarningsReviewed()  20
                && developer.getFixesApplied()  10) {
            message.append(Manual inspection is advised before applying automated refactoring. );
        }

        message.append(n);
        return message.toString();
    }
}