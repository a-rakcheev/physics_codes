# find the indices of the available operators from a higher symmetry list
# the base list is the TPY list
import numpy as np

l = 8


# read in basis operators of TPY
num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
size = num_op[l - 1]
base_ops = []
count = 0

# extra case for l=1 since file not read in as a list but as single string
names = ["y"]
for op_name in names:
    base_ops.append(op_name)

# larger ranges
for r in range(2, l + 1, 1):

    filename = "operators/operators_TPY_l" + str(r) + ".txt"
    names = np.genfromtxt(filename, dtype='str')

    for op_name in names:
        base_ops.append(op_name)

base_ops = np.array(base_ops)
# check size
print("Check Size TPY:", size, size - len(base_ops))


# # read in basis operators of TPFY
# num_op = np.loadtxt("operators/operators_TPFY_size_full.txt").astype(int)
# size2 = num_op[l - 1]
# trial_ops = []
# count = 0
#
# # no operator for l=1
#
# # extra case for l=2 since file not read in as a list but as single string
# names = ["yz"]
# for op_name in names:
#     trial_ops.append(op_name)
#
# # larger ranges
# for r in range(3, l + 1, 1):
#
#     filename = "operators/operators_TPFY_l" + str(r) + ".txt"
#     names = np.genfromtxt(filename, dtype='str')
#
#     for op_name in names:
#         trial_ops.append(op_name)
#
# trial_ops = np.array(trial_ops)
# # check size
# print("Check Size TPFY:", size2, size2 - len(trial_ops))
#
#
# # find which operators are included
# start = 0
# idx = []
# for i, op in enumerate(trial_ops):
#     r = len(op)
#     for j, op2 in enumerate(base_ops[start:]):
#         r2 = len(op2)
#
#         # test if the same, if true shift start, since ops are lex ordered
#         if op == op2:
#             idx.append(start + j)
#             start += j
#
#         # test if the range of the base operator is larger, if so the trial operator is not included
#         # => break inner loop
#         elif r2 > r:
#             break
#
#         else:
#             continue
#
#
# print(len(idx))
# # print(base_ops[idx])
# # print(trial_ops)
# # print(base_ops)
#
# np.savez("conversion_matrix_TPY_TPFY_l=" + str(l), np.array(idx))


# read in basis operators of TPFXY
num_op = np.loadtxt("operators/operators_TPFXY_size_full.txt").astype(int)
size2 = num_op[l - 1]
trial_ops = []
count = 0

# no operator for l=1 and l=2

# larger ranges
for r in range(3, l + 1, 1):

    filename = "operators/operators_TPFXY_l" + str(r) + ".txt"
    names = np.genfromtxt(filename, dtype='str')

    for op_name in names:
        trial_ops.append(op_name)

trial_ops = np.array(trial_ops)
# check size
print("Check Size TPFXY:", size2, size2 - len(trial_ops))


# find which operators are included
start = 0
idx = []
for i, op in enumerate(trial_ops):
    r = len(op)
    for j, op2 in enumerate(base_ops[start:]):
        r2 = len(op2)

        # test if the same, if true shift start, since ops are lex ordered
        if op == op2:
            idx.append(start + j)
            start += j

        # test if the range of the base operator is larger, if so the trial operator is not included
        # => break inner loop
        elif r2 > r:
            break

        else:
            continue


print(len(idx))
# print(base_ops[idx])
# print(trial_ops)
# print(base_ops)

np.savez("conversion_matrix_TPY_TPFXY_l=" + str(l), np.array(idx))





