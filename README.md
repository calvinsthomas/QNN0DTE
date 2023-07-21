# Quantum Neural Network (QNN), 0DTE retail algorithmic trading strategy
Institutional Algorithmic Trading Code (Retail)

IV % mean-reversion off highest minute SPX volume (retail sales; self-financing), Uses:
  - Institutional Market Maker Hedge: Long volatility hedge to HFT alpha (options sales biased), which is undiversified across time (infinitesimally small holding periods; static bid/ask spread arbitrage)

Speculation (Only Institutional, Not investment advice, Institutions: limit exposure to <2.5% of total capital):
  - Fractional-Kelly Criterion Betting: Self-financing using a fraction of winnings for subsequent betting on volatile auto-correlated days
      - Greater $ winnings overall/self-financing greater future bets for potentially exponential portfolio growth
        - Make a few greater bets early for chance of multiplying the initial base early-on, then quickly reduce future bettings to secure winnings
          - Close mini-portfolio for week once 3-5X target is hit; rinse-repeat
    - Deploy programmable batches of capital. To be deployed ("server kicks on") on certain volatile days, with a focus on Friday's expiration
    - Feel free to contact me if you or your firm would like fractional-kelly integrated into this program

For aspiring traders, quantitative developers, etc to access high-quality code, testable statistical arbitrage investment
      - A coding sandbox: Learn how to build your own profitable algorithmic trading programs
      - Sentiment analysis (via forums, WSB, etc) of strikes, DTE, etc to identify when market liquidity is most needed.

Programming to trade on only certain volatile days, (i.e. auto BOT at 12 PM, only if SPX is greater than |0.3%| after 9:30AM open and maintains greater than |0.3%| before 11 PM) to increase sharpe ratio/reduce weekly capital exposure to returns
    
  - Feel free to reach out if you or your firm would like this specific feature added to the program.

Strategy developed from my own experiences trading derivatives on retail exchanges and competing for premia with market makers.

Position Details:

Purchase Debit reverse IB/IC AFTER 11 AM, Sell Credit reverse IB/IC BEFORE 2 PM

Buy to Open Reverse Iron Butterfly/Iron Condor: Buy close-to-ATM strikes P/C (>35 Delta), then sell OTM strikes P/C
Sell to Close Reverse Iron Butterfly/Iron Condor: Sell close-to-ATM strikes P/C (>35 Delta), then purchase back OTM strikes P/C


Benefits:
Provides emergency liquidity and better pricing for retail AND institutional market makers at their instantaneous points of risk (position entry/exit)

P/L: Width of spread and premium paid defines capital at risk/profit. Strategy should be defined around investor preferences/objectives (hedging/speculation)

*Disclaimer: Sophisticated Trading Strategy. Not Investment Advice. Please seek your professional financial advisor.
