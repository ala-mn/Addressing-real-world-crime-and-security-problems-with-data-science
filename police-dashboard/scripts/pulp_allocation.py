import pulp
import pandas as pd

def run_allocation_with_floor(
    ward_risk: pd.DataFrame,
    total_budget: float,
    max_hours_per_ward: float = 800
) -> pd.DataFrame:
    """
    Allocate officer-hours with a guaranteed floor in every ward, then LP the remainder.

    Parameters
    ----------
    ward_risk : pd.DataFrame
        ['ward_code','risk'] – risk scores (e.g. burglary counts) per ward.
    total_budget : float
        Total hours to allocate across all wards.
    max_hours_per_ward : float, default=800
        Cap on hours any single ward can receive.

    Returns
    -------
    pd.DataFrame with ['ward_code','allocated_hours'].
    """
    df = ward_risk.copy()
    wards = df['ward_code'].tolist()
    risk_dict = dict(zip(df['ward_code'], df['risk']))
    N = len(wards)

    # Baseline floor
    min_hours = total_budget / N

    # Decision vars
    hrs = pulp.LpVariable.dicts(
        'hrs',
        wards,
        lowBound=min_hours,
        upBound=max_hours_per_ward
    )

    # Build problem
    prob = pulp.LpProblem('alloc_with_floor', pulp.LpMaximize)
    prob += pulp.lpSum(risk_dict[w]*hrs[w] for w in wards)          # maximize impact
    prob += pulp.lpSum(hrs[w] for w in wards) <= total_budget       # budget

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Gather results
    alloc = pd.DataFrame({
        'ward_code': wards,
        'allocated_hours': [hrs[w].value() for w in wards]
    })

    return alloc
