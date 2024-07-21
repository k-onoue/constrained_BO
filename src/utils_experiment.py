import re

def extract_info_from_filename(filename):
    # Extract date and objective function from the filename
    match = re.match(r'.*(\d{4}-\d{2}-\d{2})_(\w+)_v\d+\.py', filename)
    if match:
        date = match.group(1)
        objective_function = match.group(2)
        return date, objective_function
    return None, None


