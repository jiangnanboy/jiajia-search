import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_relevance_level(score: float) -> str:
    if 0.8 <= score <= 1.0:
        return "Highly relevant"
    elif 0.5 <= score < 0.8:
        return "Moderately relevant"
    elif 0.2 <= score < 0.5:
        return "Somewhat relevant"
    else:
        return "Low relevance"