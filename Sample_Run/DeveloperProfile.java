import java.util.List;

public class DeveloperProfile {
    private final String name;
    private final String experienceLevel;
    private final int warningsReviewed;
    private final int fixesApplied;
    private final List<String> preferredRefactorings;
    private final List<String> recentFilesEdited;

    public DeveloperProfile(
            String name,
            String experienceLevel,
            int warningsReviewed,
            int fixesApplied,
            List<String> preferredRefactorings,
            List<String> recentFilesEdited
    ) {
        this.name = name;
        this.experienceLevel = experienceLevel;
        this.warningsReviewed = warningsReviewed;
        this.fixesApplied = fixesApplied;
        this.preferredRefactorings = preferredRefactorings;
        this.recentFilesEdited = recentFilesEdited;
    }

    public String getName() {
        return name;
    }

    public String getExperienceLevel() {
        return experienceLevel;
    }

    public int getWarningsReviewed() {
        return warningsReviewed;
    }

    public int getFixesApplied() {
        return fixesApplied;
    }

    public List<String> getPreferredRefactorings() {
        return preferredRefactorings;
    }

    public List<String> getRecentFilesEdited() {
        return recentFilesEdited;
    }
}