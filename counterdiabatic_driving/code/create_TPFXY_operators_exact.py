# modify operators if L=l: in this case some operators that are different for larger L are permutations of each other
from contextlib import suppress
import time


# translate string to the right
def translate_string(string, steps):
        return string[-steps:] + string[:-steps]


for l in range(11, 12):
    print("l:", l)

    # read in full operators
    name = "operators/operators_TPFXY_l" + str(l) + ".txt"
    with open(name, "r") as file:
        op_list = [line.rstrip('\n') for line in file]


    # print(op_list)
    print("Size Full:", len(op_list))


    # lists for new operators
    op_list_new = []


    # loop through original operators and check for permutations within the list
    start_time = time.time()
    while len(op_list) > 0:

        operator = op_list.pop(0)
        op_list_new.append(operator)

        # print("operator:", operator)

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


    # print(op_list_new)
    print("Size:", len(op_list_new))

    # save new file
    name = "operators/operators_TPFXY_exact_l" + str(l) + ".txt"
    with open(name, "w") as writefile:
        for op in op_list_new:
            writefile.write(op + "\n")

    # save size
    name = "operators/operators_TPFXY_exact_size.txt"
    with open(name, "a") as writefile:
            writefile.write(str(len(op_list_new)) + "\n")