# Why Condition Number Thresholds Don't Change the RMSE

## The Core Issue

Your original observation was correct - **the condition numbers and RMSE ARE staying the same** across all threshold values (1e6, 1e8, 1e10, etc.). Here's why:

## The Complete Picture

### The RMSE Landscape

For **Gaussian RBF**, the full exploration shows:

```
Epsilon Range:        0.0001 → 10000
RMSE Range:           0.6498 → 0.6886 (spread: 0.0388)

Optimal epsilon:      ε ≈ 0.1136
- RMSE at optimum:    0.6498
- Condition Number:   4.29e+03

Performance at boundaries:
- ε = 0.0001 (tiny):  RMSE = 0.6886, cond# = 1e+00 (excellent stability, poor fit)
- ε = 10000 (huge):   RMSE = 0.6886, cond# = 1e+09 (terrible stability, poor fit)
```

### What Happens Across Thresholds

```
1e6  threshold → Selects ε = 0.1136 (cond# = 4.3e+03) → RMSE = 0.6498
1e8  threshold → Selects ε = 0.1136 (cond# = 4.3e+03) → RMSE = 0.6498  ← SAME
1e10 threshold → Selects ε = 0.1136 (cond# = 4.3e+03) → RMSE = 0.6498  ← SAME
1e12 threshold → Selects ε = 0.1136 (cond# = 4.3e+03) → RMSE = 0.6498  ← SAME
```

**Why?** Because the algorithm finds the **globally optimal epsilon** (0.1136), which naturally produces a condition number of 4.3e+03. Since 4.3e+03 < 1e6, it passes all thresholds equally.

## The Key Insight

**The condition number constraint is NOT the limiting factor here.**

The RMSE doesn't improve with higher thresholds because:

1. ✓ **The optimal epsilon (0.1136) is already well below the 1e6 threshold**
2. ✓ **Relaxing to 1e8, 1e10, etc. doesn't allow a better epsilon**
3. ✓ **The algorithm already found the globally best fit**

The condition number threshold only matters when it would **force selection of a sub-optimal epsilon**.

## Illustration

```
                        RMSE Curve (Gaussian)
                        
        0.690 ────────┐
                      │  (Large ε, unstable)
        0.670 ────────┼─────────────┐
                      │             │
        0.650 ────────┼─────┬───────┘
                      │     ↑ OPTIMUM
                      │    (ε=0.1136, cond≈4e3)
        0.630 ────────┴─────┴───────────
              0.0001  0.01  0.1  1   100  1000
                    Shape Parameter ε (log scale)
                    
        All thresholds (1e6, 1e8, 1e10, ...)
        can select this optimum ↑
```

## Why We See Flat Performance

The condition number threshold values tested (1e6, 1e8, 1e10, etc.) are so high that they're **never constraining the optimization**. The natural optimum always respects the threshold.

To see the threshold make a difference, we'd need to test thresholds like **1, 10, 100, 1000** which would start cutting off the optimal region and force selection of inferior (but more stable) epsilon values.

## Practical Takeaway

Your original instinct was right - we **can** improve the fit by allowing higher condition numbers. But the improvement hits a **natural limit** around condition number 4000-9000 (depending on kernel). Beyond that, increasing the condition number doesn't help because:

1. The fit plateaus at the global optimum
2. Further increases in condition number come from moving away from optimal epsilon
3. The tradeoff stops being worthwhile

## The Real Question to Ask

Instead of "what threshold gives best performance?", ask:

**"What's the absolute best fit we can achieve, and what condition number does it require?"**

Answer: **~0.6498 RMSE at ~4000 condition number** (for Gaussian)

This is already a major improvement from the conservative ε=10.0 (RMSE=0.6886, cond≈1).

## Recommendation

**Use Gaussian RBF with ε ≈ 0.115**
- Provides 5.6% better RMSE than conservative approach
- Condition number of 4300 is still numerically excellent
- This is near the natural optimum (can't improve much further)
