import re


def parse_model_key(model_key):
    """
    Decode the model_key into:
      - name (e.g. 'C-T-LGBM-XGB'),
      - weather = 'Yes'/'No',
      - holiday = 'Yes'/'No'.
    """
    # Aggregation level mapping: L_3 -> W, L_4 -> C, L_6 -> T
    agg_map = {'L_3': 'W', 'L_4': 'C', 'L_6': 'T'}
    agg_pattern = r"(L_3|L_4|L_6)"
    agg_found = re.findall(agg_pattern, model_key)
    agg_letters = [agg_map[a] for a in agg_found]
    agg_string = "-".join(agg_letters)

    # Forecasting methods (case-insensitive)
    methods_pattern = r"(sarimax|sarima|ets|naive_drift|naive_seasonal|lstm|nhits|lgbm|xgb)"
    methods_found = re.findall(methods_pattern, model_key.lower())
    methods_up = [m.upper() for m in methods_found]
    methods_string = "-".join(methods_up)

    # Check for weather and holiday flags
    weather_flag = "Yes" if "_weather_" in model_key.lower() else "No"
    holiday_flag = "Yes" if ("_holiday_" in model_key.lower() or "sarimax" in model_key.lower()) else "No"

    # Combine aggregator and methods
    if agg_string and methods_string:
        final_name = f"{agg_string}-{methods_string}"
    else:
        final_name = agg_string if methods_string == "" else methods_string

    return final_name, weather_flag, holiday_flag
