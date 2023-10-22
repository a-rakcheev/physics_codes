# modify operators if L=l: in this case some operators that are different for larger L are permutations of each other
from contextlib import suppress
import time


# translate string to the right
def translate_string(string, steps):
        return string[-steps:] + string[:-steps]


# for l in range(10, 11):
#     print("l:", l)
#
#     # read in full operators
#     name = "operators/operators_TPY_l" + str(l) + ".txt"
#     with open(name, "r") as file:
#         op_list = [line.rstrip('\n') for line in file]
#
#
#     # print(op_list)
#     print("Size Full:", len(op_list))
#
#
#     # lists for new operators
#     op_list_new = []
#
#
#     # loop through original operators and check for permutations within the list
#     start_time = time.time()
#     while len(op_list) > 0:
#
#         operator = op_list.pop(0)
#         op_list_new.append(operator)
#
#         # print("operator:", operator)
#
#         # translations
#         for i in range(1, l):
#             operator_permuted = translate_string(operator, i)
#             # print("permuted operator:", operator_permuted)
#
#             # remove permuted operator if it is contained (otherwise there is an error, but this is suppressed here)
#             with suppress(ValueError, AttributeError):
#                 op_list.remove(operator_permuted)
#
#         # reflection and translations
#         if operator != operator[::-1]:
#             operator = operator[::-1]
#
#             # translations (start with 0 since reflection itself a a permutation)
#             for i in range(0, l):
#                 operator_permuted = translate_string(operator, i)
#                 # print("permuted operator:", operator_permuted)
#
#                 # remove permuted operator if it is contained (otherwise there is an error, but this is suppressed here)
#                 with suppress(ValueError, AttributeError):
#                     op_list.remove(operator_permuted)
#
#
#     end_time = time.time()
#     print("Time:", end_time - start_time)
#
#
#     # print(op_list_new)
#     print("Size:", len(op_list_new))
#
#     # save new file
#     name = "operators/operators_TPY_exact_l" + str(l) + ".txt"
#     with open(name, "w") as writefile:
#         for op in op_list_new:
#             writefile.write(op + "\n")
#
#     # save size
#     name = "operators/operators_TPY_exact_size.txt"
#     with open(name, "a") as writefile:
#             writefile.write(str(len(op_list_new)) + "\n")


#######################################################################################################################


# second run to check if some operator with smaller length is included in l=L list through the identity
for l in range(5, 11):
    print("l:", l)

    # read in full operators
    name = "operators/operators_TPY_exact_l" + str(l) + ".txt"
    with open(name, "r") as file:
        op_list = [line.rstrip('\n') for line in file]

    # read in full operators of all smaller sizes
    op_list_full = []
    for r in range(2, l):
        name = "operators/operators_TPY_l" + str(r) + ".txt"
        with open(name, "r") as file:
            for line in file:
                op_list_full.append((line.rstrip('\n')).ljust(l, "1"))


    print(op_list_full)
    print("Size Shorter:", len(op_list_full))

    print(op_list)
    print("Size Original:", len(op_list))

    # loop through short operators and check for permutations within the list
    start_time = time.time()
    while len(op_list_full) > 0:

        operator = op_list_full.pop(0)

        # translations
        for i in range(1, l):
            operator_permuted = translate_string(operator, i)
            # print("permuted operator:", operator_permuted)

            # remove permuted operator if it is contained (otherwise there is an error, but this is suppressed here)
            with suppress(ValueError, AttributeError):
                op_list.remove(operator_permuted)

        # reflection and translations
        if operator != operator[::-1]:
            operator = operator[::-1]

            # translations (start with 0 since reflection itself a a permutation)
            for i in range(0, l):
                operator_permuted = translate_string(operator, i)
                # print("permuted operator:", operator_permuted)

                # remove permuted operator if it is contained (otherwise there is an error, but this is suppressed here)
                with suppress(ValueError, AttributeError):
                    op_list.remove(operator_permuted)


    end_time = time.time()
    print("Time:", end_time - start_time)

    print(op_list)
    print("Size:", len(op_list))

    # save new file
    name = "operators/operators_TPY_exact_reduced_l" + str(l) + ".txt"
    with open(name, "w") as writefile:
        for op in op_list:
            writefile.write(op + "\n")

    # save size
    name = "operators/operators_TPY_exact_reduced_size.txt"
    with open(name, "a") as writefile:
            writefile.write(str(len(op_list)) + "\n")