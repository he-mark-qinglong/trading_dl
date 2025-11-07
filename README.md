# trade_rl

A cryptocurrency trading system built with reinforcement learning and technical analysis.

## Project Structure

```
trade_rl/
├── strategy/              # Trading strategies and backtesting logic
│   ├── scaling_strategy.py # Main scaling strategy implementation
│   └── dash_app.py         # Dashboard visualization
├── data/                  # Market data storage
├── models/                # Trained models
├── utils/                 # Utility functions
├── config/                # Configuration files
├── tests/                 # Test suite
├── requirements.txt       # Dependencies
└── system.md              # System architecture diagram
```

## Features

- Reinforcement learning based trading strategies
- Implementation of scaling trading strategy with dynamic stop-loss
- Technical analysis indicators (MACD, ATR, etc.)
- Real-time market data integration
- Backtesting framework
- Interactive dashboard for monitoring

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run backtest:
   ```bash
   python strategy/scaling_strategy.py
   ```

3. Start dashboard:
   ```bash
   python strategy/dash_app.py
   ```

## Key Components

- **scaling_strategy.py**: Implements the core scaling trading strategy with dynamic risk management
- **dash_app.py**: Provides real-time dashboard for strategy visualization
- **backtest_module.py**: Handles historical data backtesting
- **risk_backtester.py**: Risk assessment and portfolio management

## Requirements

- Python 3.8+
- PyTorch
- Ray RLlib
- Pandas
- NumPy
- ccxt (for cryptocurrency exchange integration)
- Dash (for dashboard)
- Plotly (for charting)

## License

MIT
