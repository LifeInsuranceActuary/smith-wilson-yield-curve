# smith-wilson-yield-curve
Smith-Wilson method for yield curve interpolation and extrapolation

A Python implementation of the Smith-Wilson method for yield curve interpolation and extrapolation. 
This library takes annual swap data and converts it into a spot rate curve.

Usage - Example:
    Swap_data = np.array([
            [1, 1.0, 0.0092],
            [1, 2.0, 0.0106],
            [1, 3.0, 0.011],
            [1, 4.0, 0.011],
            [1, 5.0, 0.0138], 
            [1, 6.0, 0.0169],
            [1, 10.0, 0.0219],
            [1, 20.0, 0.0295])
    
    sw = SmithWilson()

    result,alfa = sw.smith_wilson_brute_force(
            data=sample_data,         # swap data
            cra=0,                    # Credit Risk Adjustment (10 basis points)
            ufr_ac=0.042,             # Ultimate Forward Rate (4.2%)
            alfa_min=0.05,            # Minimum alpha value
            tau=1,                    # Tau  (basis points)
            t2=60                     # Convergence maturity (60 years)
        )

Output - 6 dimensions:
- discount
- yld_intensity
- zero_ac
- fw_intensity
- forward_cc
- forward_ac
