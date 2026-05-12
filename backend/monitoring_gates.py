import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class MonitoringThresholds:
    min_top_10_capture_rate: float = 0.40
    max_brier_score: float = 0.10
    max_feature_drift_p_value: float = 0.05
    min_expected_positive_rate: float = 0.08
    max_expected_positive_rate: float = 0.12

class ModelMonitor:
    def __init__(self, thresholds: Optional[MonitoringThresholds] = None):
        self.thresholds = thresholds or MonitoringThresholds()

    def check_capture_rate(self, current_capture_rate: float) -> bool:
        if current_capture_rate < self.thresholds.min_top_10_capture_rate:
            logger.warning(f"Capture rate {current_capture_rate:.2%} is below threshold {self.thresholds.min_top_10_capture_rate:.2%}")
            return False
        return True

    def check_calibration(self, current_brier_score: float) -> bool:
        if current_brier_score > self.thresholds.max_brier_score:
            logger.warning(f"Brier score {current_brier_score:.3f} is above threshold {self.thresholds.max_brier_score:.3f}")
            return False
        return True

    def check_population_drift(self, positive_rate: float) -> bool:
        if not (self.thresholds.min_expected_positive_rate <= positive_rate <= self.thresholds.max_expected_positive_rate):
            logger.warning(f"Positive rate {positive_rate:.2%} outside expected bounds [{self.thresholds.min_expected_positive_rate:.2%}, {self.thresholds.max_expected_positive_rate:.2%}]")
            return False
        return True

def evaluate_deployment_readiness(metrics: dict) -> bool:
    monitor = ModelMonitor()
    
    passed = True
    if 'top_10_capture_rate' in metrics:
        passed &= monitor.check_capture_rate(metrics['top_10_capture_rate'])
    if 'brier_score' in metrics:
        passed &= monitor.check_calibration(metrics['brier_score'])
    if 'positive_rate' in metrics:
        passed &= monitor.check_population_drift(metrics['positive_rate'])
        
    return passed
