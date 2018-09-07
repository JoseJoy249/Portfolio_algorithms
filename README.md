This repository contains the implementation of constant rebalance portfolio (CRP), and an approximate version of Covers Universal Portfolio.

A constant rebalanced portfolio is a sequential investment strategy that maintains fixed through time, trading period by trading period, the wealth distribution among a set of assets.

The universal portfolio algorithm rebalances the portfolio at the beginning of each trading period. At the beginning of the first trading period it starts with a naive diversification. In the following trading periods the portfolio composition depends on the historical total return of all possible constant-rebalanced portfolios.


The repository contains stock data obtain from yahoo finance, for 14 companies from Jan 2008 to Dec 2017
