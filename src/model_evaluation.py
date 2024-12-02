# Functions to evaluate the model
def calculate_cost(token_count: int) -> float:
    """
    Calculate the cost of a model interaction based on the number of input and output tokens.

    Based on Google Gemini pricing model https://ai.google.dev/pricing#1_5flash
    """
    cost_per_token = 0.00000008 if token_count < 128000 else 0.00000015

    return token_count * cost_per_token
