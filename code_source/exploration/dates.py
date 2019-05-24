import datetime

from code_source.dataset import Dataset
import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd
class DateContainer:

    df=Dataset.get_data()

    min_date=df.DisbursementDate.min()
    max_date=df.DisbursementDate.max()
    min_month=calendar.month_name[min_date.month]
    max_month=calendar.month_name[max_date.month]
    min_month_abbr=calendar.month_abbr[min_date.month]
    max_month_abbr=calendar.month_abbr[max_date.month]
    min_year=min_date.year
    max_year=max_date.year
    if min_year==max_year:
        year=min_year
    else: year='%s - %s'%(min_year,max_year)
    month_day_count=relativedelta(max_date,min_date)