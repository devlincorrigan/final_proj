# python3 rolling_baseline/evaluate_rolling_baseline.py
Rows: 31162
Pushes: 0

Directional Accuracy
mean_pick: correct=15953 decided=31003 eligible=31162 accuracy=51.46%
median_pick: correct=14973 decided=29091 eligible=31162 accuracy=51.47%

Projection Error
rolling_mean_points: count=31162 mae=5.497 rmse=7.059
rolling_median_points: count=31162 mae=5.542 rmse=7.155

Betting ROI
mean_pick: bets=31000 wins=15952 win_rate=51.46% avg_price=1.874 total_profit=-1140.080 roi=-3.68%
median_pick: bets=29089 wins=14972 win_rate=51.47% avg_price=1.874 total_profit=-1076.840 roi=-3.70%

Accuracy By Edge Size
mean:
  [0, 1): count=11905 accuracy=50.55%
  [1, 2): count=8857 accuracy=51.58%
  [2, 3): count=5107 accuracy=51.87%
  [3, +inf): count=5134 accuracy=52.94%
median:
  [0, 1): count=7916 accuracy=49.95%
  [1, 2): count=9018 accuracy=51.60%
  [2, 3): count=5548 accuracy=52.29%
  [3, +inf): count=6609 accuracy=52.43%

ROI By Edge Size
mean:
  [0, 1): count=11904 profit=-617.410 roi=-5.19%
  [1, 2): count=8855 profit=-305.730 roi=-3.45%
  [2, 3): count=5107 profit=-162.110 roi=-3.17%
  [3, +inf): count=5134 profit=-54.830 roi=-1.07%
median:
  [0, 1): count=7915 profit=-511.900 roi=-6.47%
  [1, 2): count=9017 profit=-304.420 roi=-3.38%
  [2, 3): count=5548 profit=-129.540 roi=-2.33%
  [3, +inf): count=6609 profit=-130.980 roi=-1.98%

Rows By Season
2023-24: 7950
2024-25: 9912
2025-26: 13300

Rows By Window
40: 31162

Rows By Bookmaker
BetMGM: 8
BetOnline.ag: 40
Caesars: 43
DraftKings: 30753
FanDuel: 310
Fanatics: 5
PointsBet (US): 3

# python3 empirical\ baseline/evaluate_empirical_baseline.py
Rows: 30509
Pushes: 0

Directional Accuracy
empirical_pick: correct=15196 decided=29466 eligible=30509 accuracy=51.57%

Betting ROI
empirical_pick: bets=29464 wins=15195 win_rate=51.57% avg_price=1.874 total_profit=-1023.080 roi=-3.47%

Accuracy By Edge Size
  [0.00, 0.10): count=6195 accuracy=50.36%
  [0.10, 0.20): count=7032 accuracy=52.03%
  [0.20, 0.30): count=6027 accuracy=50.34%
  [0.30, +inf): count=10212 accuracy=52.71%

ROI By Edge Size
  [0.00, 0.10): count=6195 profit=-341.280 roi=-5.51%
  [0.10, 0.20): count=7032 profit=-182.440 roi=-2.59%
  [0.20, 0.30): count=6025 profit=-350.270 roi=-5.81%
  [0.30, +inf): count=10212 profit=-149.090 roi=-1.46%

Rows By Season
2023-24: 7771
2024-25: 9737
2025-26: 13001

Rows By Window
80: 30509

Rows By Bookmaker
BetMGM: 8
BetOnline.ag: 40
Caesars: 43
DraftKings: 30104
FanDuel: 306
Fanatics: 5
PointsBet (US): 3

# python3 context_model_benchmark.py
Rows
  baseline_rows=31162
  joined_usable_rows=31162
  train_rows=21813
  test_rows=9349
  train_date_range=2023-10-24 to 2025-12-09
  test_date_range=2025-12-09 to 2026-04-12

Coefficients
  intercept=0.319314
  opp_defensive_rating_roll=0.313335
  opp_pace_roll=0.236162
  player_minutes_roll=-0.390857
  player_usage_roll=-0.124192
  isHomeInt=0.108167

Train
  Rolling baseline (selected rows are top 1 lines per event)
  selected_rows=2716 events=2716 correct=1443 accuracy=53.13% profit=-20.510 roi=-0.76%
  Context-adjusted
  selected_rows=2716 events=2716 correct=1475 accuracy=54.31% profit=33.100 roi=1.22%

Test
  Rolling baseline (selected rows are top 1 lines per event)
  selected_rows=843 events=843 correct=452 accuracy=53.62% profit=1.500 roi=0.18%
  Context-adjusted
  selected_rows=843 events=843 correct=455 accuracy=53.97% profit=6.770 roi=0.80%