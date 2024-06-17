
def process_deserialized_json(data):
    if isinstance(data, dict):
        return {int(key): value for key, value in data.items()}
    elif isinstance(data, list):
        return [{int(key): value for key, value in d.items()} for d in data]
    else:
        raise TypeError("Input must be a dictionary or a list of dictionaries")
