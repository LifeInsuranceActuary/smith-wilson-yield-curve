import numpy as np
from typing import Literal, Union, Tuple, List

class SmithWilson:
    def __init__(self):
        pass
    
    @staticmethod
    def hh(z: float) -> float:
        """Help function for Hmat function"""
        return (z + np.exp(-z)) / 2
    
    @staticmethod
    def hmat(u: float, v: float) -> float:
        """The big h-function according to 139 of the specs"""
        return SmithWilson.hh(u + v) - SmithWilson.hh(abs(u - v))

    
    def smith_wilson_brute_force(
        self,
        data: np.ndarray,
        cra: float,
        ufr_ac: float,
        alfa_min: float,
        tau: float,
        t2: int
    ) -> np.ndarray:
        """
        Implement Smith-Wilson interpolation method.
        
        Args:
            data: Array with shape (n, 3) containing:
                - Column 1: Indicator vector (1 if maturity is DLT qualified as credible input)
                - Column 2: Maturity vector
                - Column 3: Rate vector
            cra: Credit Risk Adjustment in basis points
            ufr_ac: Ultimate Forward Rate annual compounded (per unit)
            alfa_min: Minimum alpha value
            tau: Tau value in basis points
            t2: Convergence maturity
            
        Returns:
            Array containing calculated curves:
            - Discount factors
            - Yield intensities
            - Zero rates (annual compounding)
            - Forward intensities
            - Forward rates (continuous compounding)
            - Forward rates (annual compounding)
        """
        # Convert basis points to decimal
        tau = tau / 10000
        
        # Extract liquid rates and maturities
        mask = data[:, 0].astype(bool) #create true of false where liquid data (as first row is liquid indicator)
        maturity_vector = data[mask, 1]
        spot_rate_vector = data[mask, 2] - cra / 10000
        
        num_rates = len(maturity_vector) #Count of how many swaps there are
        max_liquid_term = np.max(maturity_vector) #max term swap, i.e., a swap at year 20 -> 20
        
        # Calculate natural logarithm of UFR 
        ln_ufr = np.log(1 + ufr_ac)
        

        # Sores values for coupon payments, with the cash flows spread over the period until maturity
        Q = np.zeros((num_rates, int(max_liquid_term )))
        
        #Prices of all instruments are set to 1 by construction
        #Swap i there are pay-offs of r(i) at time 1, 2, ... u(i)-1
        #Plus a final pay-off of 1+r(i) at time u(i)
        for i in range(num_rates):
            for j in range(int(maturity_vector[i] )):
                if j == int(maturity_vector[i] ) - 1:
                    Q[i, j] = np.exp(-ln_ufr * (j+1) ) * (1 + spot_rate_vector[i] )
                else:
                    Q[i, j] = np.exp(-ln_ufr * (j+1) ) * spot_rate_vector[i] 
              

        # Find optimal alpha and gamma
        g_alfa_output = self._g_alfa(alfa_min, Q, num_rates, max_liquid_term, t2, tau)
        
        #'g(alfa)<=tau => alfamin is optimal
        if g_alfa_output[0] <= 0:
            alfa = alfa_min
            gamma = g_alfa_output[1]
        else:
        #else -> scanning for the optimal alfa is based on the scan-procedure taken from Eiopa
        # in each for-next loop the next optimal alfa decimal is scanned for, starting with an stepsize of 0.1 (first decimal) 
        # followed by the a next decimal through stepsize = stepsize/10
            alfa, gamma = self._find_optimal_alfa(
                alfa_min, Q, num_rates, max_liquid_term, t2, tau
            )
        
        # Calculate matrices H and G
        h_matrix = np.zeros((122, int(max_liquid_term )))
        g_matrix = np.zeros((122, int(max_liquid_term )))
        
        for i in range(122):
            for j in range(int(max_liquid_term )):
                h_matrix[i, j] = self.hmat(alfa * i, alfa * (j+1) )
                if (j+1)  > i:
                    g_matrix[i, j] = alfa * (1 - np.exp(-alfa * (j+1) ) * np.cosh(alfa * i))
                else:
                    g_matrix[i, j] = alfa * np.exp(-alfa * i) * np.sinh(alfa * (j+1) )
        
        # Optimal alfa and corresponding gamma have been determined
        #Now the SW-present value function can be calculated: 
        # # p(v)=exp(-lnUFR*v)*(1+H(v,u)*Qb)

        # Calculate discount factors and rates
        temp_discount = h_matrix @ gamma
        temp_intensity = g_matrix @ gamma
        
        # Initialize arrays to store various financial calculations  
        # Each array has 122 elements, likely corresponding to time points (e.g., months or quarters)
        discount = np.zeros(122)       # Discount factors for future cash flows  
        fw_intensity = np.zeros(122)   # Forward rate intensities (instantaneous rates for future periods)  
        yld_intensity = np.zeros(122)  # Yield curve intensities (spot rate representation)  
        forward_ac = np.zeros(122)     # Forward rates in annualized compounding form  
        zero_ac = np.zeros(122)        # Zero-coupon bond yields in annualized compounding form  
        forward_cc = np.zeros(122)     # Forward rates in continuously compounded form  
        zero_cc = np.zeros(122)        # Zero-coupon bond yields in continuously compounded form  

        
        # Calculate initial values
        temp = np.sum((1 - np.exp(-alfa * np.arange(1, max_liquid_term  + 1) )) * gamma)
        yld_intensity[0] = ln_ufr - alfa * temp
        fw_intensity[0] = yld_intensity[0]
        discount[0] = 1
        
        # Calculate remaining values
        for i in range(1, 121):
            yld_intensity[i] = ln_ufr - np.log(1 + temp_discount[i]) / i
            fw_intensity[i] = ln_ufr - temp_intensity[i] / (1 + temp_discount[i])
            discount[i] = np.exp(-ln_ufr * i) * (1 + temp_discount[i])
            zero_ac[i] = (1 / discount[i]) ** (1 / i) - 1
            forward_ac[i] = discount[i-1] / discount[i] - 1
            
        # Calculate continuous compound rates
        forward_cc[1:121] = np.log(1 + forward_ac[1:121])
        zero_cc[1:121] = np.log(1 + zero_ac[1:121])
        
       
        return np.column_stack([
            discount,
            yld_intensity,
            zero_ac,
            fw_intensity,
            forward_cc,
            forward_ac
        ]), alfa
    
    def _g_alfa(
        self, 
        alfa: float, 
        Q: np.ndarray, 
        mm: int, 
        max_liquid_term: float, 
        t2: int, 
        tau: float
    ) -> Tuple[float, np.ndarray]:  
        
        """Calculate g(alfa)-tau and gamma according to specs"""
        # Create H matrix
        h = np.zeros((int(max_liquid_term ), int(max_liquid_term )))
        for i in range(int(max_liquid_term )):
            for j in range(int(max_liquid_term )):
                h[i, j] = self.hmat(alfa * (i+1) , alfa * (j+1) )
        
        # Calculate temporary values
        temp1 = 1 - np.sum(Q, axis=1)
        
        # Calculate b according to specs
        QHQ = Q @ h @ Q.T
        b = np.linalg.inv(QHQ) @ temp1
        
        # Calculate gamma (Qb)
        gamma = Q.T @ b
        
        # Calculate kappa
        temp2 = np.sum(gamma * np.arange(1, len(gamma) + 1) )
        temp3 = np.sum(gamma * np.sinh(alfa * np.arange(1, len(gamma) + 1) ))
        
        kappa = (1 + alfa * temp2) / temp3
        
        return (alfa / abs(1 - kappa * np.exp(t2 * alfa)) - tau, gamma)
    
    def _find_optimal_alfa(
        self,
        alfa_min: float,
        Q: np.ndarray,
        mm: int,
        max_liquid_term: float,
        t2: int,
        tau: float,
        precision: int = 6
    ) -> Tuple[float, np.ndarray]:
        """Find optimal alfa value through scanning procedure"""
        step_size = 0.1
        alfa = alfa_min + step_size
        
        # Find initial alfa estimate
        while self._g_alfa(alfa, Q, mm, max_liquid_term, t2, tau)[0] > 0 and alfa <= 20:
            alfa += step_size
            
        # Refine alfa estimate
        for _ in range(precision - 1):
            alfa_range = np.arange(
                alfa - step_size,
                alfa + step_size / 10,
                step_size / 10
            )
            for a in alfa_range:
                g_alfa_output = self._g_alfa(a, Q, mm, max_liquid_term, t2, tau)
                if g_alfa_output[0] <= 0:
                    alfa = a
                    gamma = g_alfa_output[1]
                    break
            step_size /= 10

        return alfa, gamma
    

if __name__ == "__main__":
    #
    
    # Initialize Smith-Wilson calculator
    sw = SmithWilson()

    sample_data = np.array( [
        [1, 1.0, 0.0092],
        [1, 2.0, 0.0106],
        [1, 3.0, 0.011],
        [1, 4.0, 0.011],
        [1, 5.0, 0.0138], 
        [1, 6.0, 0.0169],
        [1, 10.0, 0.0219],
        [1, 20.0, 0.0295]
    ], dtype=np.float64)


    # Calculate curves
    result,alfa = sw.smith_wilson_brute_force(
            data=sample_data,           # Our input data          # Annual payments
            cra=0,                    # Credit Risk Adjustment (10 basis points)
            ufr_ac=0.042,              # Ultimate Forward Rate (4.2%)
            alfa_min=0.05,             # Minimum alpha value
            tau=1,                   # Tau (100 basis points)
            t2=60                      # Convergence maturity (60 years)
        )
    
    print(alfa)
    for year in range(30):
            discount = result[year, 2]
            print(f"{year:4d} | {discount:14.6f}")

