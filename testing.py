"""
File to test all the objectives in objectives.py
"""

import pytest
from sorting import *
test_data = np.loadtxt("test1.csv", delimiter=",")

# testing our 5 objectives with assert statements
def test_overallocation():
    assert overallocation(test_data) == 37, "Overallocation incorrectly calculated"


def test_conflicts():
    assert conflicts(test_data) == 8, "Conflicts incorrectly calculated"


def test_undersupport():
    assert under_support(test_data) == 1, "Undersupport incorrectly calculated"


def test_unwilling():
    assert unwilling(test_data) == 53, "Unwilling incorrectly calculated"


def test_unpreferred():
    assert unprefered(test_data) == 15, "Unpreferred incorrectly calculated"
