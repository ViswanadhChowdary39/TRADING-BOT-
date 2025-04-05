# Trading Bot System Design and Implementation

![GitHub](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8%2B-green)

A robust trading bot system that collects real-time market data, implements a threshold-based trading strategy, and includes special rules for end-of-simulation handling. Designed for the Roostoo platform.

## Features

- **Real-time Data Collection**: Secure API communication with authentication and error handling
- **Dynamic Threshold Strategy**: Adaptive trading threshold based on recent price averages
- **Intelligent Trading Rules**: 
  - Buy when price falls below threshold
  - Sell with minimum 0.3% profit margin
- **End-of-Simulation Handling**: Special rules for last 10 seconds to maximize returns
- **Risk Management**: Portfolio tracking and Sharpe ratio calculation
- **Comprehensive Logging**: Detailed trade history and performance metrics

## System Architecture

```mermaid
graph TD
    A[API Client] -->|Fetch Data| B[Strategy Engine]
    B -->|Generate Signals| C[Simulation Bot]
    C -->|Execute Trades| D[Risk Manager]
    D -->|Update Metrics| E[Reporting]
