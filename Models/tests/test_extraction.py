from backd00r_ai.extraction.expert_signals import ExpertSignalExtractor
from backd00r_ai.extraction.structural_metrics import StructuralMetricExtractor


JAVA_SOURCE = """
package demo;

public class AccountService extends BaseService {
    private int balance;
    private Customer customer;

    public void deposit(int amount) {
        if (amount > 0) {
            this.balance += amount;
            customer.record(amount);
        }
    }

    public void withdraw(int amount) {
        if (amount > 0 && balance >= amount) {
            balance -= amount;
            customer.record(-amount);
        }
    }

    private void helper() {
        System.out.println(customer.name);
    }
}
"""


def test_structural_metrics_extract_required_values():
    metrics = StructuralMetricExtractor().extract(JAVA_SOURCE)
    assert metrics["nom"] == 3
    assert metrics["nof"] == 2
    assert metrics["wmc"] >= 5
    assert metrics["cc_max"] >= 2
    assert metrics["loc"] > 0
    assert metrics["dit"] == 1
    assert metrics["cida"] > 0
    assert 0.0 <= metrics["coa"] <= 1.0
    assert metrics["method_loc_max"] > 0
    assert metrics["method_param_max"] == 1
    assert metrics["foreign_method_calls"] >= 2


def test_expert_signals_are_deterministic_and_bounded():
    metrics = StructuralMetricExtractor().extract(JAVA_SOURCE)
    metrics.update({"code_churn": 10.0, "bug_fix_ratio": 0.2})
    signals = ExpertSignalExtractor().extract(None, metrics)
    for value in signals.values():
        assert 0.0 <= value <= 1.0
    assert "jdeodorant_signal" in signals
    assert "jspirit_signal" in signals
