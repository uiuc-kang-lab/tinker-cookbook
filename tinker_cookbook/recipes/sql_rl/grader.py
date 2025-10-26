import sqlite3

def grade_basic(result1, result2):
    # should have same number of columns:
    if len(result1) == 0:
        if len(result2) == 0:
            return True
        else:
            return False
    elif len(result2) == 0:
        return False
    if len(result1[0]) == len(result2[0]):
        return True

    return False

def grade_multiset(result1, result2):
    # check if length matches
    if len(result1) != len(result2):
        return False, {"message": "Number of rows do not match"}
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    # check if the sorted results match as multisets
    if sorted(sorted_result1) != sorted(sorted_result2):
        return False, {"message": "Results do not match as multisets"}
    return True, {"message": "Results match as multisets"}

def grade_subset(all_results, matching_results, strict_row_count: int = None, minimum_row_count: int = None):
    assert strict_row_count is None or minimum_row_count is None, "Only one of strict_row_count or minimum_row_count should be provided"
    if strict_row_count is not None:
        if len(matching_results) != strict_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} does not equal required {strict_row_count}"}
    if minimum_row_count is not None:
        if len(matching_results) < minimum_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} is less than required minimum {minimum_row_count}"}
    # check if all matching_results are in all_results
    set_all = set([tuple(sorted(row)) for row in all_results])
    set_matching = set([tuple(sorted(row)) for row in matching_results])
    if not set_matching.issubset(set_all):
        return False, {"message": "Not all matching rows are in the full result set"}
    return True, {"message": "All matching rows are in the full result set"}

def grade_list(result1, result2):
    # check if length matches
    if len(result1) != len(result2):
        return False, {"message": "Number of rows do not match"}
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    if sorted_result1 != sorted_result2:
        return False, {"message": "Results do not match in order"}
    return True, {"message": "Results match in order"}

def grade(ground_truth_result, generated_result, grading_method: str = "multiset"):
    if not grade_basic(ground_truth_result, generated_result):
        return False, {"message": "Basic grading failed: number of columns does not match"}

    if "multiset" in grading_method:
        return grade_multiset(ground_truth_result, generated_result)
    elif "subset" in grading_method:
        _, method, row_count = grading_method.split(",")
        if method == "=":
            return grade_subset(ground_truth_result, generated_result, strict_row_count=int(row_count))
        elif method == ">=":
            return grade_subset(ground_truth_result, generated_result, minimum_row_count=int(row_count))
        else:
            return False, {"message": f"Unknown subset method: {method}"}
    elif "list" in grading_method:
        return grade_list(ground_truth_result, generated_result)
    else:
        return False, {"message": f"Unknown grading method: {grading_method}"}
    