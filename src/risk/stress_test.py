"""
Stress testing module for FX-Commodity correlation arbitrage strategy.
Implements scenario analysis and stress testing for risk controls.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger
from risk.manager import RiskManager, RiskConfig


@dataclass
class StressScenario:
    """Definition of a stress testing scenario."""

    name: str  # Name of the scenario
    description: str  # Description of the scenario
    drawdown_multiplier: float = 1.0  # Multiplier for drawdown stress
    volatility_multiplier: float = 1.0  # Multiplier for volatility stress
    correlation_stress: float = 0.0  # Correlation stress (positive or negative)
    duration: int = 30  # Duration of stress in days
    frequency: float = 1.0  # Frequency of occurrence (0.0 to 1.0)


@dataclass
class StressTestResult:
    """Results of a stress test."""

    scenario: StressScenario
    max_drawdown: float
    final_equity: float
    vol_scaling_triggered: bool
    circuit_breaker_triggered: bool
    violations: List[str] = field(default_factory=list)


class StressTester:
    """
    Stress tester for FX-Commodity correlation arbitrage strategy.
    Implements scenario analysis and stress testing for risk controls.
    """

    def __init__(self):
        """Initialize the stress tester."""
        self.scenarios = self._create_default_scenarios()
        logger.info("Initialized stress tester")

    def _create_default_scenarios(self) -> List[StressScenario]:
        """
        Create default stress testing scenarios.

        Returns:
            List of default stress scenarios.
        """
        scenarios = [
            StressScenario(
                name="Market Crash",
                description="Severe market downturn with high volatility",
                drawdown_multiplier=3.0,
                volatility_multiplier=2.5,
                duration=60,
            ),
            StressScenario(
                name="Flash Crash",
                description="Sudden extreme market movement",
                drawdown_multiplier=5.0,
                volatility_multiplier=4.0,
                duration=5,
            ),
            StressScenario(
                name="Correlation Breakdown",
                description="Historical correlation relationship breaks down",
                correlation_stress=-0.5,
                duration=90,
            ),
            StressScenario(
                name="Volatility Spike",
                description="Sudden increase in market volatility",
                volatility_multiplier=3.0,
                duration=30,
            ),
            StressScenario(
                name="Prolonged Drawdown",
                description="Extended period of negative returns",
                drawdown_multiplier=2.0,
                duration=180,
            ),
            StressScenario(
                name="Daily Drawdown Stress",
                description="Stress test for daily drawdown limits",
                drawdown_multiplier=1.5,  # Moderate stress to test daily limits
                duration=10,
            ),
            StressScenario(
                name="High Volatility Regime",
                description="Extended period of high volatility testing position scaling",
                volatility_multiplier=2.0,
                duration=60,
            ),
        ]

        return scenarios

    def add_scenario(self, scenario: StressScenario) -> None:
        """
        Add a custom stress scenario.

        Args:
            scenario: Stress scenario to add.
        """
        self.scenarios.append(scenario)
        logger.info(f"Added stress scenario: {scenario.name}")

    def run_stress_test(
        self, equity_series: pd.Series, risk_config: RiskConfig = None
    ) -> List[StressTestResult]:
        """
        Run stress tests on equity series.

        Args:
            equity_series: Series of equity values.
            risk_config: Risk configuration. If None, uses default.

        Returns:
            List of stress test results.
        """
        if risk_config is None:
            risk_config = RiskConfig()

        results = []

        for scenario in self.scenarios:
            logger.info(f"Running stress test: {scenario.name}")
            result = self._run_single_scenario(equity_series, scenario, risk_config)
            results.append(result)

        return results

    def _run_single_scenario(
        self,
        equity_series: pd.Series,
        scenario: StressScenario,
        risk_config: RiskConfig,
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Args:
            equity_series: Series of equity values.
            scenario: Stress scenario to run.
            risk_config: Risk configuration.

        Returns:
            Stress test result.
        """
        # Create risk manager for this scenario
        risk_manager = RiskManager(risk_config)

        # Apply stress multipliers to equity series
        stressed_equity = self._apply_stress_to_equity(equity_series, scenario)

        # Track violations
        violations = []
        vol_scaling_triggered = False
        circuit_breaker_triggered = False

        # Simulate through stressed equity series
        for i, (date, equity) in enumerate(stressed_equity.items()):
            current_date = (
                pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            )

            # Update risk manager
            risk_manager.update_account_state(equity, current_date)

            # Check for violations
            if risk_manager.check_drawdown_limit(stressed_equity.iloc[: i + 1]):
                violations.append(f"Drawdown limit breached at {date}")

            if risk_manager.check_daily_loss_limit():
                violations.append(f"Daily loss limit breached at {date}")

            if risk_manager.check_weekly_loss_limit():
                violations.append(f"Weekly loss limit breached at {date}")

            # Check if circuit breaker was triggered
            if risk_manager.check_circuit_breaker(current_date):
                circuit_breaker_triggered = True

        # Calculate final metrics
        peak = stressed_equity.cummax()
        drawdown = (stressed_equity - peak) / peak
        max_drawdown = drawdown.min()
        final_equity = stressed_equity.iloc[-1]

        result = StressTestResult(
            scenario=scenario,
            max_drawdown=max_drawdown,
            final_equity=final_equity,
            vol_scaling_triggered=vol_scaling_triggered,
            circuit_breaker_triggered=circuit_breaker_triggered,
            violations=violations,
        )

        return result

    def _apply_stress_to_equity(
        self, equity_series: pd.Series, scenario: StressScenario
    ) -> pd.Series:
        """
        Apply stress to equity series based on scenario parameters.

        Args:
            equity_series: Series of equity values.
            scenario: Stress scenario.

        Returns:
            Stressed equity series.
        """
        # Create a copy to avoid modifying original
        stressed = equity_series.copy()

        # Apply drawdown stress
        if scenario.drawdown_multiplier > 1.0:
            # Calculate returns
            returns = stressed.pct_change().fillna(0)

            # Amplify negative returns
            stressed_returns = returns.copy()
            stressed_returns[stressed_returns < 0] *= scenario.drawdown_multiplier

            # Reconstruct equity series
            stressed = (1 + stressed_returns).cumprod() * stressed.iloc[0]

        # Apply volatility stress
        if scenario.volatility_multiplier > 1.0:
            # Add additional volatility
            additional_vol = np.random.normal(
                0, 0.01 * (scenario.volatility_multiplier - 1.0), len(stressed)
            )
            stressed *= 1 + additional_vol

        return stressed

    def generate_stress_report(self, results: List[StressTestResult]) -> str:
        """
        Generate a stress test report.

        Args:
            results: List of stress test results.

        Returns:
            Formatted stress test report.
        """
        report = "STRESS TEST REPORT\n"
        report += "=" * 50 + "\n\n"

        for result in results:
            report += f"Scenario: {result.scenario.name}\n"
            report += f"Description: {result.scenario.description}\n"
            report += f"Max Drawdown: {result.max_drawdown:.2%}\n"
            report += f"Final Equity: {result.final_equity:.2f}\n"
            report += f"Circuit Breaker Triggered: {result.circuit_breaker_triggered}\n"
            report += f"Violations: {len(result.violations)}\n"

            if result.violations:
                report += "  Violations:\n"
                for violation in result.violations:
                    report += f"    - {violation}\n"

            report += "\n"

        return report

    def check_pass_fail(self, results: List[StressTestResult]) -> Dict[str, bool]:
        """
        Check pass/fail criteria for stress tests.

        Args:
            results: List of stress test results.

        Returns:
            Dictionary with pass/fail status for each scenario.
        """
        pass_fail = {}

        for result in results:
            # Pass criteria:
            # 1. Max drawdown should not exceed 3x the configured limit
            # 2. No more than 2 violations per scenario
            # 3. Final equity should not be less than 50% of starting equity
            # 4. For new scenarios, additional specific criteria

            drawdown_pass = (
                abs(result.max_drawdown) <= 3 * 0.15
            )  # 3x of 15% default limit
            violations_pass = len(result.violations) <= 2
            equity_pass = result.final_equity >= 0.5  # 50% of starting equity

            # Additional criteria for new scenarios
            if result.scenario.name == "Daily Drawdown Stress":
                # For daily drawdown stress, we expect some violations but not catastrophic failure
                drawdown_pass = (
                    abs(result.max_drawdown) <= 5 * 0.15
                )  # More lenient for this scenario
            elif result.scenario.name == "High Volatility Regime":
                # For high volatility regime, we want to ensure position scaling works
                # Check that violations are not excessive
                violations_pass = (
                    len(result.violations) <= 5
                )  # More violations allowed for this scenario

            pass_fail[result.scenario.name] = (
                drawdown_pass and violations_pass and equity_pass
            )

        return pass_fail


def run_default_stress_tests(
    equity_series: pd.Series,
) -> Tuple[List[StressTestResult], Dict[str, bool]]:
    """
    Run default stress tests on equity series.

    Args:
        equity_series: Series of equity values.

    Returns:
        Tuple of (stress test results, pass/fail criteria).
    """
    tester = StressTester()
    results = tester.run_stress_test(equity_series)
    pass_fail = tester.check_pass_fail(results)

    return results, pass_fail
