# FX-Commodity Correlation Arbitrage System - Flow Diagrams

## Overview

This document provides detailed system flow diagrams showing the module relationships and data flow after architectural improvements. These diagrams illustrate the clean separation of concerns and reduced coupling achieved through the refactoring process.

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        YahooAPI[Yahoo Finance API]
        EIAAPI[EIA API]
        DataCache[Data Cache]
    end
    
    subgraph "Feature Layer"
        SpreadCalc[Spread Calculation]
        Indicators[Technical Indicators]
        RegimeFilter[Regime Filtering]
    end
    
    subgraph "ML Layer"
        Ensemble[ML Ensemble]
        OLSModel[OLS Model]
        KalmanModel[Kalman Model]
        CorrModel[Correlation Model]
        GBModel[Gradient Boosting]
        LSTMModel[LSTM Model]
    end
    
    subgraph "Strategy Layer"
        SignalGen[Signal Generation]
        PositionSizing[Position Sizing]
        TimeStop[Time Stop Logic]
    end
    
    subgraph "Interface Layer"
        FeaturePrep[Feature Preparation Interface]
        Validation[Input Validation Interface]
    end
    
    subgraph "Risk Layer"
        RiskManager[Risk Manager]
        CircuitBreaker[Circuit Breakers]
        DrawdownMonitor[Drawdown Monitor]
    end
    
    subgraph "Execution Layer"
        ExecutionPolicy[Execution Policy]
        CostModel[Cost Model]
        Slippage[Slippage Model]
    end
    
    subgraph "Backtest Layer"
        BacktestEngine[Backtest Engine]
        PerformanceMetrics[Performance Metrics]
        TradeAnalysis[Trade Analysis]
    end
    
    subgraph "Configuration Layer"
        ConfigLoader[Config Loader]
        ConfigValidator[Config Validator]
        ParameterValidator[Parameter Validator]
    end
    
    %% Data Flow
    YahooAPI --> DataCache
    EIAAPI --> DataCache
    DataCache --> SpreadCalc
    DataCache --> Indicators
    SpreadCalc --> SignalGen
    Indicators --> SignalGen
    RegimeFilter --> SignalGen
    
    %% ML Flow
    FeaturePrep --> Ensemble
    Ensemble --> OLSModel
    Ensemble --> KalmanModel
    Ensemble --> CorrModel
    Ensemble --> GBModel
    Ensemble --> LSTMModel
    Ensemble --> SignalGen
    
    %% Strategy Flow
    SignalGen --> PositionSizing
    SignalGen --> TimeStop
    PositionSizing --> BacktestEngine
    
    %% Interface Layer
    Validation --> SignalGen
    Validation --> BacktestEngine
    FeaturePrep --> Ensemble
    
    %% Risk Flow
    RiskManager --> SignalGen
    RiskManager --> BacktestEngine
    CircuitBreaker --> RiskManager
    DrawdownMonitor --> RiskManager
    
    %% Execution Flow
    ExecutionPolicy --> BacktestEngine
    CostModel --> ExecutionPolicy
    Slippage --> ExecutionPolicy
    
    %% Configuration Flow
    ConfigLoader --> ConfigValidator
    ConfigValidator --> ParameterValidator
    ParameterValidator --> Validation
    
    %% Performance Flow
    BacktestEngine --> PerformanceMetrics
    BacktestEngine --> TradeAnalysis
    
    classDef data fill:#e3f2fd
    classDef feature fill:#f3e5f5
    classDef ml fill:#e8f5e8
    classDef strategy fill:#fff3e0
    classDef interface fill:#fce4ec
    classDef risk fill:#ffebee
    classDef execution fill:#f1f8e9
    classDef backtest fill:#e0f2f1
    classDef config fill:#f9fbe7
    
    class YahooAPI,EIAAPI,DataCache data
    class SpreadCalc,Indicators,RegimeFilter feature
    class Ensemble,OLSModel,KalmanModel,CorrModel,GBModel,LSTMModel ml
    class SignalGen,PositionSizing,TimeStop strategy
    class FeaturePrep,Validation interface
    class RiskManager,CircuitBreaker,DrawdownMonitor risk
    class ExecutionPolicy,CostModel,Slippage execution
    class BacktestEngine,PerformanceMetrics,TradeAnalysis backtest
    class ConfigLoader,ConfigValidator,ParameterValidator config
```

## 2. Signal Generation Flow

```mermaid
graph TD
    Start([Start Signal Generation])
    
    %% Input Validation
    ValidateInputs{Validate Inputs}
    ValidateConfig{Validate Config}
    ValidateSeries{Validate Series Alignment}
    
    %% Data Preparation
    PrepareData[Prepare Market Data]
    CalculateSpread[Calculate Spread<br/>OLS/Kalman/RLS]
    CalculateIndicators[Calculate Technical Indicators<br/>Z-Score, ATR, etc.]
    
    %% Regime Filtering
    RegimeAnalysis[Regime Analysis]
    CorrelationGate[Correlation Gate]
    StationarityTest[ADF Stationarity Test]
    CombineRegime[Combine Regime Filters]
    
    %% ML Enhancement
    DecisionML{Use ML Models?}
    PrepareFeatures[Prepare ML Features]
    RunEnsemble[Run Ensemble Models]
    CombinePredictions[Combine ML Predictions]
    
    %% Signal Logic
    CalculateZScore[Calculate Z-Score Signals]
    ApplyThresholds[Apply Entry/Exit Thresholds]
    PositionLogic[Apply Position Logic]
    TimeStopCheck[Apply Time Stop Logic]
    
    %% Risk Checks
    RiskCheck{Risk Limits OK?}
    ApplyRiskLimits[Apply Risk Limits]
    
    %% Output
    GeneratePositions[Generate Position Sizes]
    OutputSignals[Output Signal DataFrame]
    End([Signal Generation Complete])
    
    %% Error Handling
    ValidationError[Log Validation Error]
    ConfigError[Log Config Error]
    MLError[Log ML Error]
    RiskError[Log Risk Error]
    
    %% Flow connections
    Start --> ValidateInputs
    ValidateInputs -->|Pass| ValidateConfig
    ValidateInputs -->|Fail| ValidationError
    ValidateConfig -->|Pass| ValidateSeries
    ValidateConfig -->|Fail| ConfigError
    ValidateSeries -->|Pass| PrepareData
    ValidateSeries -->|Fail| ValidationError
    
    PrepareData --> CalculateSpread
    CalculateSpread --> CalculateIndicators
    CalculateIndicators --> RegimeAnalysis
    
    RegimeAnalysis --> CorrelationGate
    CorrelationGate --> StationarityTest
    StationarityTest --> CombineRegime
    
    CombineRegime --> DecisionML
    DecisionML -->|Yes| PrepareFeatures
    DecisionML -->|No| CalculateZScore
    PrepareFeatures --> RunEnsemble
    RunEnsemble --> CombinePredictions
    CombinePredictions -->|Success| CalculateZScore
    RunEnsemble -->|Error| MLError
    MLError --> CalculateZScore
    
    CalculateZScore --> ApplyThresholds
    ApplyThresholds --> PositionLogic
    PositionLogic --> TimeStopCheck
    TimeStopCheck --> RiskCheck
    
    RiskCheck -->|Pass| GeneratePositions
    RiskCheck -->|Fail| ApplyRiskLimits
    ApplyRiskLimits --> RiskError
    RiskError --> GeneratePositions
    
    GeneratePositions --> OutputSignals
    OutputSignals --> End
    
    ValidationError --> End
    ConfigError --> End
    
    classDef process fill:#e3f2fd
    classDef decision fill:#fff3e0
    classDef error fill:#ffebee
    classDef start fill:#e8f5e8
    
    class Start,End start
    class ValidateInputs,ValidateConfig,ValidateSeries,DecisionML,RiskCheck decision
    class ValidationError,ConfigError,MLError,RiskError error
```

## 3. Backtest Engine Flow

```mermaid
graph TD
    StartBT([Start Backtest])
    
    %% Input Processing
    ValidateDF{Validate Signals DataFrame}
    ExtractParams[Extract Config Parameters]
    InitRiskManager[Initialize Risk Manager]
    InitExecutionPolicy[Initialize Execution Policy]
    
    %% Signal Processing
    GenerateTradeSignals[Generate Trading Signals<br/>Entry/Exit/Stop]
    ApplyExecutionDelay[Apply One-Bar Execution Delay]
    
    %% PnL Calculation Loop
    StartLoop[Start Daily Processing Loop]
    CalculatePnL[Calculate Position PnL]
    CalculateExecutionCosts[Calculate Execution Costs]
    UpdateEquityCurve[Update Equity Curve]
    
    %% Risk Monitoring
    CheckDrawdown{Check Drawdown Limits}
    CheckRiskLimits{Check Position Limits}
    ApplyCircuitBreaker[Apply Circuit Breaker]
    
    %% Trade Analysis
    IdentifyTrades[Identify Individual Trades]
    CalculateTradeStats[Calculate Trade Statistics]
    CalculateReturns[Calculate Trade Returns]
    
    %% Performance Metrics
    CalculateBasicMetrics[Calculate Basic Metrics<br/>Total Return, CAGR]
    CalculateRiskMetrics[Calculate Risk Metrics<br/>Sharpe, Sortino, MaxDD]
    CalculateTradeMetrics[Calculate Trade Metrics<br/>Win Rate, Profit Factor]
    CalculateDistributionStats[Calculate Distribution Statistics<br/>Skewness, Kurtosis, VaR]
    
    %% Rolling Metrics
    DecisionRolling{Calculate Rolling Metrics?}
    CalculateRollingMetrics[Calculate Rolling Performance<br/>30/60/90 Day Windows]
    
    %% Output Generation
    GenerateReport[Generate Backtest Report]
    OutputResults[Output Results & Metrics]
    EndBT([Backtest Complete])
    
    %% Error Handling
    ValidationErrorBT[Validation Error]
    ConfigErrorBT[Config Error]
    ProcessingErrorBT[Processing Error]
    
    %% Continuation Logic
    MoreDays{More Days to Process?}
    
    %% Flow connections
    StartBT --> ValidateDF
    ValidateDF -->|Pass| ExtractParams
    ValidateDF -->|Fail| ValidationErrorBT
    ExtractParams --> InitRiskManager
    InitRiskManager --> InitExecutionPolicy
    InitExecutionPolicy --> GenerateTradeSignals
    
    GenerateTradeSignals --> ApplyExecutionDelay
    ApplyExecutionDelay --> StartLoop
    
    StartLoop --> CalculatePnL
    CalculatePnL --> CalculateExecutionCosts
    CalculateExecutionCosts --> UpdateEquityCurve
    UpdateEquityCurve --> CheckDrawdown
    
    CheckDrawdown -->|Within Limits| CheckRiskLimits
    CheckDrawdown -->|Breach| ApplyCircuitBreaker
    ApplyCircuitBreaker --> MoreDays
    
    CheckRiskLimits -->|Within Limits| MoreDays
    CheckRiskLimits -->|Breach| ApplyCircuitBreaker
    
    MoreDays -->|Yes| StartLoop
    MoreDays -->|No| IdentifyTrades
    
    IdentifyTrades --> CalculateTradeStats
    CalculateTradeStats --> CalculateReturns
    CalculateReturns --> CalculateBasicMetrics
    
    CalculateBasicMetrics --> CalculateRiskMetrics
    CalculateRiskMetrics --> CalculateTradeMetrics
    CalculateTradeMetrics --> CalculateDistributionStats
    CalculateDistributionStats --> DecisionRolling
    
    DecisionRolling -->|Yes| CalculateRollingMetrics
    DecisionRolling -->|No| GenerateReport
    CalculateRollingMetrics --> GenerateReport
    
    GenerateReport --> OutputResults
    OutputResults --> EndBT
    
    ValidationErrorBT --> EndBT
    ConfigErrorBT --> EndBT
    ProcessingErrorBT --> EndBT
    
    classDef process fill:#e3f2fd
    classDef decision fill:#fff3e0
    classDef loop fill:#f3e5f5
    classDef error fill:#ffebee
    classDef start fill:#e8f5e8
    
    class StartBT,EndBT start
    class ValidateDF,CheckDrawdown,CheckRiskLimits,MoreDays,DecisionRolling decision
    class StartLoop loop
    class ValidationErrorBT,ConfigErrorBT,ProcessingErrorBT error
```

## 4. ML Ensemble Flow

```mermaid
graph TD
    StartML([Start ML Ensemble])
    
    %% Data Preparation
    PrepareFeatures[Prepare Feature Matrix<br/>Prices, Returns, Volatilities]
    ValidateFeatures{Validate Features}
    SplitFeatures[Split Features by Model Requirements]
    
    %% Model Training
    TrainOLS[Train OLS Model<br/>Rolling Window Regression]
    TrainKalman[Train Kalman Filter<br/>Adaptive Parameters]
    TrainCorrelation[Train Correlation Model<br/>Rolling Correlation]
    TrainGradientBoosting[Train Gradient Boosting<br/>Tree-based Ensemble]
    TrainLSTM[Train LSTM Model<br/>Sequential Deep Learning]
    
    %% Model Validation
    ValidateOLS{OLS Training Success?}
    ValidateKalman{Kalman Training Success?}
    ValidateCorr{Correlation Training Success?}
    ValidateGB{GB Training Success?}
    ValidateLSTM{LSTM Training Success?}
    
    %% Prediction Generation
    PredictOLS[Generate OLS Predictions]
    PredictKalman[Generate Kalman Predictions]
    PredictCorr[Generate Correlation Predictions]
    PredictGB[Generate GB Predictions]
    PredictLSTM[Generate LSTM Predictions]
    
    %% Ensemble Combination
    CombinePredictions[Combine Model Predictions<br/>Weighted Average]
    CalculateWeights[Calculate Dynamic Weights<br/>Based on Recent Performance]
    
    %% Feature Importance
    ExtractImportance[Extract Feature Importance<br/>From Applicable Models]
    
    %% Output
    OutputEnsemble[Output Ensemble Predictions]
    OutputImportance[Output Feature Importance]
    EndML([ML Ensemble Complete])
    
    %% Error Handling
    FeatureError[Feature Preparation Error]
    ModelError[Model Training Error]
    PredictionError[Prediction Error]
    
    %% Flow connections
    StartML --> PrepareFeatures
    PrepareFeatures --> ValidateFeatures
    ValidateFeatures -->|Pass| SplitFeatures
    ValidateFeatures -->|Fail| FeatureError
    
    SplitFeatures --> TrainOLS
    SplitFeatures --> TrainKalman
    SplitFeatures --> TrainCorrelation
    SplitFeatures --> TrainGradientBoosting
    SplitFeatures --> TrainLSTM
    
    TrainOLS --> ValidateOLS
    ValidateOLS -->|Success| PredictOLS
    ValidateOLS -->|Fail| ModelError
    
    TrainKalman --> ValidateKalman
    ValidateKalman -->|Success| PredictKalman
    ValidateKalman -->|Fail| ModelError
    
    TrainCorrelation --> ValidateCorr
    ValidateCorr -->|Success| PredictCorr
    ValidateCorr -->|Fail| ModelError
    
    TrainGradientBoosting --> ValidateGB
    ValidateGB -->|Success| PredictGB
    ValidateGB -->|Fail| ModelError
    
    TrainLSTM --> ValidateLSTM
    ValidateLSTM -->|Success| PredictLSTM
    ValidateLSTM -->|Fail| ModelError
    
    PredictOLS --> CombinePredictions
    PredictKalman --> CombinePredictions
    PredictCorr --> CombinePredictions
    PredictGB --> CombinePredictions
    PredictLSTM --> CombinePredictions
    ModelError --> CombinePredictions
    
    CombinePredictions --> CalculateWeights
    CalculateWeights --> ExtractImportance
    ExtractImportance --> OutputEnsemble
    OutputEnsemble --> OutputImportance
    OutputImportance --> EndML
    
    FeatureError --> EndML
    
    classDef process fill:#e8f5e8
    classDef decision fill:#fff3e0
    classDef model fill:#f3e5f5
    classDef error fill:#ffebee
    classDef start fill:#e3f2fd
    
    class StartML,EndML start
    class ValidateFeatures,ValidateOLS,ValidateKalman,ValidateCorr,ValidateGB,ValidateLSTM decision
    class TrainOLS,TrainKalman,TrainCorrelation,TrainGradientBoosting,TrainLSTM,PredictOLS,PredictKalman,PredictCorr,PredictGB,PredictLSTM model
    class FeatureError,ModelError,PredictionError error
```

## 5. Module Dependency Graph (After Refactoring)

```mermaid
graph TB
    subgraph "Layer 1: Infrastructure"
        Config[config/*]
        Interfaces[interfaces/*]
        Utils[utils/*]
    end
    
    subgraph "Layer 2: Data & Features"
        Data[data/*]
        Features[features/*]
    end
    
    subgraph "Layer 3: ML & Models"
        ML[ml/*]
        Models[ml/models/*]
    end
    
    subgraph "Layer 4: Strategy & Risk"
        Strategy[strategy/*]
        Risk[risk/*]
    end
    
    subgraph "Layer 5: Execution & Backtesting"
        Execution[exec/*]
        Backtest[backtest/*]
    end
    
    subgraph "Layer 6: Analysis & Reporting"
        Analysis[analysis/*]
        Reporting[reporting/*]
    end
    
    %% Clean Dependencies (No Circular)
    Config --> Interfaces
    Config --> Data
    Config --> Features
    Config --> Strategy
    Config --> Risk
    Config --> Backtest
    
    Interfaces --> Strategy
    Interfaces --> ML
    Interfaces --> Backtest
    
    Data --> Features
    Features --> Strategy
    Features --> ML
    
    ML --> Strategy
    
    Strategy --> Execution
    Strategy --> Backtest
    
    Risk --> Strategy
    Risk --> Backtest
    
    Execution --> Backtest
    
    Backtest --> Analysis
    Backtest --> Reporting
    
    classDef layer1 fill:#e3f2fd
    classDef layer2 fill:#f3e5f5
    classDef layer3 fill:#e8f5e8
    classDef layer4 fill:#fff3e0
    classDef layer5 fill:#ffebee
    classDef layer6 fill:#f1f8e9
    
    class Config,Interfaces,Utils layer1
    class Data,Features layer2
    class ML,Models layer3
    class Strategy,Risk layer4
    class Execution,Backtest layer5
    class Analysis,Reporting layer6
```

## 6. Interface Layer Architecture

```mermaid
graph TD
    subgraph "Interface Layer"
        FeaturePrep[Feature Preparation Interface]
        Validation[Input Validation Interface]
        ErrorHandling[Error Handling Interface]
    end
    
    subgraph "Validation Components"
        ParameterValidator[Parameter Validator]
        SeriesValidator[Series Validator]
        ConfigValidator[Config Validator]
        ValidationRegistry[Validation Registry]
    end
    
    subgraph "Feature Preparation Components"
        FeaturePreparator[Feature Preparator Interface]
        DefaultPreparator[Default Feature Preparator]
        CustomPreparator[Custom Feature Preparators]
    end
    
    subgraph "Client Modules"
        StrategyModule[Strategy Module]
        BacktestModule[Backtest Module]
        MLModule[ML Module]
        RiskModule[Risk Module]
    end
    
    %% Interface connections
    Validation --> ParameterValidator
    Validation --> SeriesValidator
    Validation --> ConfigValidator
    Validation --> ValidationRegistry
    
    FeaturePrep --> FeaturePreparator
    FeaturePreparator --> DefaultPreparator
    FeaturePreparator --> CustomPreparator
    
    %% Client connections
    StrategyModule --> Validation
    StrategyModule --> FeaturePrep
    BacktestModule --> Validation
    BacktestModule --> FeaturePrep
    MLModule --> FeaturePrep
    RiskModule --> Validation
    
    classDef interface fill:#fce4ec
    classDef component fill:#e8f5e8
    classDef client fill:#fff3e0
    
    class FeaturePrep,Validation,ErrorHandling interface
    class ParameterValidator,SeriesValidator,ConfigValidator,ValidationRegistry,FeaturePreparator,DefaultPreparator,CustomPreparator component
    class StrategyModule,BacktestModule,MLModule,RiskModule client
```

## Key Architectural Improvements

### 1. Eliminated Tight Coupling
- **Before**: Backtest engine directly imported private functions from strategy layer
- **After**: Uses clean interface layer for feature preparation and validation

### 2. Standardized Error Handling
- **Before**: Inconsistent validation patterns across modules
- **After**: Centralized validation interface with consistent error types

### 3. Clean Layered Architecture
- Clear dependency hierarchy from infrastructure to analysis layers
- No circular dependencies between major components
- Interface layer provides abstraction between core modules

### 4. Improved Separation of Concerns
- **Data Layer**: Focuses purely on data loading and caching
- **Feature Layer**: Handles technical calculations and indicators
- **ML Layer**: Manages model training and prediction
- **Strategy Layer**: Implements trading logic and signal generation
- **Interface Layer**: Provides clean abstractions and validation
- **Execution Layer**: Handles trade execution and cost modeling
- **Backtest Layer**: Manages backtesting and performance analysis

### 5. Enhanced Modularity
- Each module has well-defined responsibilities
- Clear interfaces between modules
- Easy to test and maintain individual components
- Supports plugin architecture for extending functionality

## Usage Examples

### Using Validation Interface
```python
from src.interfaces.validation import validate_trading_config, ValidationError

try:
    validate_trading_config(config)
except ValidationError as e:
    logger.error(f"Configuration invalid: {e}")
```

### Using Feature Preparation Interface
```python
from src.interfaces.feature_preparation import prepare_features_for_model

features = prepare_features_for_model(fx_series, comd_series, config)
```

This architectural design provides a solid foundation for the FX-Commodity Correlation Arbitrage trading system with clear separation of concerns, standardized interfaces, and maintainable code structure.