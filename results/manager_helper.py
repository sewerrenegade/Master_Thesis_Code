def convert_int_keys_to_str_keys(input_dict):
    """
    Converts a dictionary with integer keys to a dictionary with string keys.

    Parameters:
    - input_dict (dict): The input dictionary with integer keys.

    Returns:
    - dict: A new dictionary with the same values but with string keys.
    """
    return {str(key): value for key, value in input_dict.items()}

def convert_str_keys_to_int_keys(input_dict):
    """
    Converts a dictionary with string keys to a dictionary with integer keys.

    Parameters:
    - input_dict (dict): The input dictionary with string keys.

    Returns:
    - dict: A new dictionary with the same values but with integer keys.
    """
    # Use a try-except block to handle cases where the string cannot be converted to an integer
    output_dict = {}
    for key, value in input_dict.items():
        try:
            int_key = int(key)
            output_dict[int_key] = value
        except ValueError:
            raise ValueError(f"Key '{key}' cannot be converted to an integer.")
    return output_dict
