# Quantum Neural Network (QNN), 0DTE retail algorithmic trading strategy
Open-Source Algorithmic Trading Code (Retail)

IV % mean-reversion off highest minute SPX volume (retail sales), Uses:
  - Institutional Market Maker Hedge: Long volatility hedge to HFT alpha (options sales biased), which is undiversified across time (infinitesimally small holding periods)
  - Open-source access for aspiring traders, quantitative developers, etc to high-quality code, testable statistical arbitrage investment
      - A coding sandbox: Learn how to build your own profitable algorithmic trading programs
      - Sentiment analysis (via forums, WSB, etc) of strikes, DTE, etc to identify when market liquidity is most needed.

Strategy developed from my own experiences trading derivatives on retail exchanges and competing for premia with market makers.

Position Details:

Purchase Debit reverse IB/IC AFTER 11 AM, Sell Credit reverse IB/IC BEFORE 2 PM

Buy to Open Reverse Iron Butterfly/Iron Condor: Buy close-to-ATM strikes P/C (>35 Delta), then sell OTM strikes P/C
Sell to Close Reverse Iron Butterfly/Iron Condor: Sell close-to-ATM strikes P/C (>35 Delta), then purchase back OTM strikes P/C


Benefits:
Provides emergency liquidity and better pricing for retail AND institutional market makers at their instantaneous points of risk (position entry/exit)

P/L: Width of spread and premium paid defines capital at risk/profit. Strategy should be defined around investor preferences/objectives (hedging/speculation)

*Disclaimer: Sophisticated Trading Strategy. Not Investment Advice. Please seek your professional financial advisor.
