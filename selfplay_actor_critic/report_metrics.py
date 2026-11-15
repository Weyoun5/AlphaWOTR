"""Offline report arithmetic: unavailable statistics serialize as JSON null.

These helpers do not change losses, fitting admission, or inference.
"""
import math

import numpy as np


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def ratio(numerator, denominator):
    if denominator is None or denominator <= 0:
        return None
    return finite_or_none(numerator / denominator)


def explained_fraction(error, reference):
    value = ratio(error, reference)
    return None if value is None else 1. - value


def r2_score(target, prediction):
    target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
    # Comparing the values also catches constant decimal labels whose
    # floating-point mean can differ slightly from every element.
    if target.size < 2 or np.all(target == target.flat[0]):
        return None
    return explained_fraction(float(np.mean((target - prediction) ** 2)), float(target.var()))


def correlation(left, right):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if (left.size < 2 or right.size < 2 or np.all(left == left.flat[0])
            or np.all(right == right.flat[0])):
        return None
    return finite_or_none(np.corrcoef(left, right)[0, 1])


def metric_text(value, spec=".3f", *, scale=1., suffix=""):
    value = finite_or_none(value)
    return "unavailable" if value is None else format(scale * value, spec) + suffix
