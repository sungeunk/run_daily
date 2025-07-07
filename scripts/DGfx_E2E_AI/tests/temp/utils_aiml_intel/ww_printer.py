from datetime import datetime


def date_to_work_week():
    # Get the current system date
    date = datetime.now()
    
    # Get the ISO calendar week and weekday
    iso_calendar = date.isocalendar()
    year = date.strftime('%y')
    week_number = iso_calendar[1]
    weekday = iso_calendar[2]
    
    # Format the result as yyWWxx.y
    return f'{year}WW{week_number:02}.{weekday}'

# Example usage
print(date_to_work_week())