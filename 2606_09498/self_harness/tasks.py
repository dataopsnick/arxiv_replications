"""Minimal verifiable task suite (a stand-in for Terminal-Bench-2.0).

Each task asks the agent to produce a Python solution that writes a required
*artifact* (a result), and a deterministic verifier checks it. The tasks are
designed so that a *minimal* initial harness tends to fail in characteristic,
clusterable ways (missing artifact, wrong format, ignoring edge cases) that a
harness edit can plausibly fix -- exactly the regime Self-Harness targets.

A task is solved by the agent emitting a single python function `solve()` that
returns the answer. The verifier calls solve(*args) in a sandboxed subprocess
and compares against the expected output. Failures carry a verifier-grounded
`cause` used for weakness mining.

Case format (unambiguous): each case is [args, expected], where `args` is ALWAYS
the full list of positional arguments to solve(). So a single list argument is
written as [[1,2,3]] (one positional arg that happens to be a list).
"""

TASKS = [
    {
        "id": "sum_list",
        "prompt": "Write solve(nums) that returns the sum of a list of integers. Return 0 for an empty list.",
        "cases": [[[[1, 2, 3]], 6], [[[]], 0], [[[-1, 1]], 0], [[[10]], 10]],
    },
    {
        "id": "reverse_string",
        "prompt": "Write solve(s) that returns the reversed string.",
        "cases": [[["abc"], "cba"], [[""], ""], [["a"], "a"], [["racecar"], "racecar"]],
    },
    {
        "id": "fizzbuzz_n",
        "prompt": "Write solve(n) that returns a list of length n where index i (1-based) is 'FizzBuzz' if divisible by 15, 'Fizz' if by 3, 'Buzz' if by 5, else the number as a string.",
        "cases": [[[3], ["1", "2", "Fizz"]], [[5], ["1", "2", "Fizz", "4", "Buzz"]], [[1], ["1"]]],
    },
    {
        "id": "max_subarray",
        "prompt": "Write solve(nums) that returns the maximum sum of any contiguous non-empty subarray (Kadane's algorithm). Handle all-negative inputs.",
        "cases": [[[[-2, 1, -3, 4, -1, 2, 1, -5, 4]], 6], [[[-1, -2, -3]], -1], [[[5]], 5]],
    },
    {
        "id": "count_vowels",
        "prompt": "Write solve(s) that returns the number of vowels (a,e,i,o,u, case-insensitive) in s.",
        "cases": [[["hello"], 2], [["XYZ"], 0], [["AEIOU"], 5], [[""], 0]],
    },
    {
        "id": "is_palindrome_num",
        "prompt": "Write solve(n) that returns True if the integer n reads the same forwards and backwards, else False. Negative numbers are never palindromes.",
        "cases": [[[121], True], [[-121], False], [[10], False], [[0], True]],
    },
    {
        "id": "merge_sorted",
        "prompt": "Write solve(a, b) that merges two sorted integer lists into one sorted list.",
        "cases": [[[[1, 3], [2, 4]], [1, 2, 3, 4]], [[[], [1]], [1]], [[[5], []], [5]]],
    },
    {
        "id": "roman_to_int",
        "prompt": "Write solve(s) that converts a Roman numeral string to an integer (supports I,V,X,L,C,D,M and subtractive notation like IV, IX).",
        "cases": [[["III"], 3], [["IV"], 4], [["IX"], 9], [["LVIII"], 58], [["MCMXCIV"], 1994]],
    },
    {
        "id": "two_sum",
        "prompt": "Write solve(nums, target) that returns indices [i,j] (i<j) of two numbers summing to target. Assume exactly one solution.",
        "cases": [[[[2, 7, 11, 15], 9], [0, 1]], [[[3, 2, 4], 6], [1, 2]], [[[3, 3], 6], [0, 1]]],
    },
    {
        "id": "gcd",
        "prompt": "Write solve(a, b) that returns the greatest common divisor of two positive integers.",
        "cases": [[[12, 8], 4], [[17, 5], 1], [[100, 10], 10]],
    },
    {
        "id": "flatten",
        "prompt": "Write solve(nested) that flattens a list that may contain nested lists (arbitrary depth) into a flat list of integers, preserving order.",
        "cases": [[[[1, [2, [3]]]], [1, 2, 3]], [[[]], []], [[[1, 2, [3, 4]]], [1, 2, 3, 4]]],
    },
    {
        "id": "anagram",
        "prompt": "Write solve(a, b) that returns True if strings a and b are anagrams of each other (ignoring case), else False.",
        "cases": [[["listen", "Silent"], True], [["abc", "abd"], False], [["", ""], True]],
    },
]

# Fixed split assignment (held-in vs held-out), fixed across harness variants.
HELD_IN_IDS = {
    "sum_list", "reverse_string", "fizzbuzz_n", "max_subarray",
    "count_vowels", "is_palindrome_num",
}
HELD_OUT_IDS = {
    "merge_sorted", "roman_to_int", "two_sum", "gcd", "flatten", "anagram",
}


def held_in():
    return [t for t in TASKS if t["id"] in HELD_IN_IDS]


def held_out():
    return [t for t in TASKS if t["id"] in HELD_OUT_IDS]
