# Small-Scale Java Project with Intentional Code Smells

## Project Title
**Mini Code Quality Analyzer**

## Purpose
This is a small Java project designed for testing or demonstrating code smell detection. It contains **5 Java classes** and intentionally includes the following code smells:

1. **Class-Level God Class**
2. **Long Method**
3. **Feature Envy**

The code is intentionally written with poor design in selected areas so that evaluators, students, or code smell detection tools can identify and analyze the smells.

---

## Project Structure

```text
MiniCodeQualityAnalyzer/
├── Main.java
├── ProjectAnalyzer.java
├── SourceFile.java
├── MetricSnapshot.java
└── DeveloperProfile.java
```

---

## Code Smell Mapping

| Code Smell | Location | Description |
|---|---|---|
| **God Class** | `ProjectAnalyzer` class | This class handles metric calculation, code smell detection, recommendation generation, report formatting, developer-specific advice, scoring, and summary generation. It has too many responsibilities. |
| **Long Method** | `ProjectAnalyzer.analyzeProject()` | This method performs validation, metric collection, detection, scoring, recommendation generation, and report construction in one large method. |
| **Feature Envy** | `ProjectAnalyzer.createPersonalizedDeveloperMessage()` | This method depends heavily on data from `DeveloperProfile` and `SourceFile`, using many of their getters instead of letting those classes handle their own behavior. |

```

---

## Expected Output Behavior

When executed, the program prints a code quality report showing:

- Project summary
- File-level metrics
- Detected code smells
- Refactoring recommendations
- Developer-specific notes
- Project health score

Because `OrderService.java` has high LOC, many methods, many fields, high complexity, and frequent external data access, it is expected to trigger all three target smells:

1. **God Class**
2. **Long Method**
3. **Feature Envy**

---

## Notes for Evaluation

This project is intentionally small and artificial. It is not meant to represent best practices. Instead, it is useful for:

- Testing code smell detection tools
- Demonstrating refactoring opportunities
- Creating a sample input for thesis evaluation
- Showing how design problems can exist even in a small Java project

Here are the **5 classes** and their **supposed / intentional code smell mapping**:

| Class Name                                             | Intended Code Smell                                             |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| `Main.java`                                            | No main code smell; used only as the program runner/test driver |
| `ProjectAnalyzer.java`                                 | **God Class**                                                   |
| `ProjectAnalyzer.analyzeProject()`                     | **Long Method**                                                 |
| `ProjectAnalyzer.createPersonalizedDeveloperMessage()` | **Feature Envy**                                                |
| `SourceFile.java`                                      | No direct code smell; acts as a data/model class                |
| `MetricSnapshot.java`                                  | No direct code smell; acts as a data/model class                |
| `DeveloperProfile.java`                                | No direct code smell; acts as a data/model class                |

The main smell-heavy class is **`ProjectAnalyzer.java`** because it intentionally contains all three:

1. **God Class** — the whole `ProjectAnalyzer` class has too many responsibilities.
2. **Long Method** — `analyzeProject()` performs too many tasks in one method.
3. **Feature Envy** — `createPersonalizedDeveloperMessage()` uses too much data from `DeveloperProfile` and `SourceFile`.

So the expected detection target is mainly:

| Target Location                                        | Expected Smell |
| ------------------------------------------------------ | -------------- |
| `ProjectAnalyzer`                                      | God Class      |
| `ProjectAnalyzer.analyzeProject()`                     | Long Method    |
| `ProjectAnalyzer.createPersonalizedDeveloperMessage()` | Feature Envy   |
