import pandas as pd
import numpy as np
import random

if __name__ == "__main__":
    # Create a dummy DataFrame
    employee_email = ["erdenebileg.b@techpack.mn", "itgel.o@techpack.mn"]

    # Create a DataFrame of worktime clock in and clock out times for the two employees above by randomly generating their arrival timestamps between 8:30 AM and 10:30 AM and their departure timestamps between 6:30 PM and 7:30 PM for each working day until 23rd of January 2026.

    date_range = pd.date_range(start="2025-12-01", end="2026-01-23", freq='B')  # Business days only
    data = []
    for date in date_range:
        for email in employee_email:
            clock_in = date + pd.Timedelta(hours=random.randint(8, 10), minutes=random.randint(0, 59))
            clock_out = date + pd.Timedelta(hours=random.randint(18, 19), minutes=random.randint(0, 59))
            data.append({"employee_email": email, "clock_in": clock_in, "clock_out": clock_out})
    
    df = pd.DataFrame(data)
    # Now create two new columns "total working hours" and "total salary hours", which is the total working hours within the arrival of 8-10 AM and the departure of 5-7 PM respectively.
    df["total_working_hours"] = (df["clock_out"] - df["clock_in"]).dt.total_seconds() / 3600.0
    df["total_salary_hours"] = df["total_working_hours"].apply(lambda x: min(x, 8))  # Cap at 8 hours for salary calculation

    # Round the hours to 2 decimal places
    df["total_working_hours"] = df["total_working_hours"].round(2)
    df["total_salary_hours"] = df["total_salary_hours"].round(2)
    # Save to CSV
    df.to_csv("dummy_employee_worktime_data.csv", index=False)