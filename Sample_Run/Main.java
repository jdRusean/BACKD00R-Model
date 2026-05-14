import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<SourceFile> files = Arrays.asList(
                new SourceFile(
                        "OrderService.java",
                        "src/service/OrderService.java",
                        420,
                        18,
                        9,
                        34,
                        6,
                        "Mika",
                        Arrays.asList("validateOrder", "calculateTotal", "applyDiscount", "saveOrder", "sendReceipt")
                ),
                new SourceFile(
                        "Customer.java",
                        "src/model/Customer.java",
                        90,
                        4,
                        2,
                        6,
                        1,
                        "Luis",
                        Arrays.asList("getName", "getEmail", "updateAddress")
                ),
                new SourceFile(
                        "PaymentGateway.java",
                        "src/payment/PaymentGateway.java",
                        180,
                        8,
                        5,
                        12,
                        3,
                        "Mika",
                        Arrays.asList("connect", "charge", "refund", "disconnect")
                )
        );

        DeveloperProfile developer = new DeveloperProfile(
                "Mika",
                "Intermediate",
                14,
                7,
                Arrays.asList("Extract Method", "Move Method", "Extract Class"),
                Arrays.asList("OrderService.java", "PaymentGateway.java")
        );

        ProjectAnalyzer analyzer = new ProjectAnalyzer();
        String report = analyzer.analyzeProject(files, developer);
        System.out.println(report);
    }
}