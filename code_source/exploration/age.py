from code_source.dataset import Dataset
import pandas as pd


class AgeContainer:
    df = Dataset.df

    age_less18_pct = round(df[df.Age < 18]['Customer_WID'].count() \
                           / df['Customer_WID'].count() * 100, 3)
    age_more80_pct = round(df[(df.Age > 80) & (df.Age < 117)]['Customer_WID'] \
                           .count() / df['Customer_WID'].count() * 100, 3)
    age_117_pct = round(df[df.Age == 117]['Customer_WID'].count() \
                        / df['Customer_WID'].count() * 100, 3)
    age_118_count = df[df.Age == 118]['Customer_WID'].count()
    age_18_80 = df[(df.Age >= 18) & (df.Age <= 80)]
