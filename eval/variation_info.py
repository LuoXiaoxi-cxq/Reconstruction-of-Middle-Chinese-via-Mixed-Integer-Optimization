"""
The following code comes from Binglun Li's dissertation.
"""

from collections import Counter

import numpy as np


def calculate_entropy(cluster):
    """Calculate the entropy of a clustering."""
    total_points = len(cluster)
    if total_points == 0:
        return 0
    label_counts = Counter(cluster)
    probabilities = [count / total_points for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def calculate_mutual_information(U, V):
    """Calculate the mutual information between two clusterings."""
    total_points = len(U)
    mutual_info = 0
    U_labels, V_labels = set(U), set(V)
    for u in U_labels:
        for v in V_labels:
            intersection_size = sum(1 for i in range(total_points) if U[i] == u and V[i] == v)
            if intersection_size == 0:
                continue
            p_u = sum(1 for x in U if x == u) / total_points
            p_v = sum(1 for x in V if x == v) / total_points
            p_uv = intersection_size / total_points
            mutual_info += p_uv * np.log2(p_uv / (p_u * p_v))
    return mutual_info


def calculate_variation_of_information(U, V):
    """Calculate the variation of information between two clusterings."""
    entropy_U = calculate_entropy(U)
    entropy_V = calculate_entropy(V)
    mutual_information = calculate_mutual_information(U, V)
    variation_of_information = entropy_U + entropy_V - 2 * mutual_information
    return variation_of_information, variation_of_information / (entropy_U + entropy_V)
