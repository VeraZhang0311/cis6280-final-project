# Planning Screen Results

30 scenes; policy seeds [0, 2, 3]; 90 episodes per variant.

| Method | Mean boxes | >=2 | >=3 | 4 | Ground fall | ms/block | Delta vs baseline | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| policy | 1.233 | 32.2% | 11.1% | 3.3% | 3.3% | 58.2 | - | - |
| baseline | 1.422 | 42.2% | 11.1% | 2.2% | 4.4% | 124.2 | - | - |
| clipbaseline | 1.478 | 46.7% | 15.6% | 0.0% | 2.2% | 124.9 | +0.056 | [-0.167, 0.256] |
| trajectory | 1.622 | 56.7% | 18.9% | 6.7% | 2.2% | 135.4 | +0.200 | [-0.022, 0.411] |
| lookahead | 1.478 | 47.8% | 13.3% | 2.2% | 2.2% | 135.7 | +0.056 | [-0.144, 0.267] |
| adaptive | 1.567 | 51.1% | 20.0% | 4.4% | 1.1% | 136.2 | +0.144 | [-0.100, 0.378] |
| combined | 1.444 | 45.6% | 12.2% | 2.2% | 1.1% | 137.1 | +0.022 | [-0.222, 0.267] |
| budget32 | 1.322 | 42.2% | 7.8% | 1.1% | 3.3% | 143.1 | -0.100 | [-0.333, 0.122] |

Best at 16 candidates: trajectory.

Development results require independent confirmation; smoke is a pipeline check.
