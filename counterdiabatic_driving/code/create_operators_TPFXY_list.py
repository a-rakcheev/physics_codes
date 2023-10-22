# create all operators with translational and reflection (parity) or short TP symmetry with range l > 1
import itertools as iter

l = 10                                                                   # range of paulis strings
op_list_x = []                                                          # strings starting with x
op_list_y = []                                                          # strings starting with y
op_list_z = []                                                          # strings starting with z

# create all 4^(l -2) operators between the first and last characters
bulk_strings = []
[bulk_strings.append("".join(x)) for x in iter.product('1xyz', repeat=(l - 2))]

# creating the lists:
# for each initial x, y, z go through all bulk strings and last characters x, y, z
# if the first and last characters equal, check if the reversed string is already in the list (reflection symmetry)
# therefore start with equal characters as suffix
# if not add the new string
# use gauge to select only strings with an overall odd number of y
# the spin flip symmetry leads to an odd number of z as well

# x
# equal suffix
for bulk_str in bulk_strings:

    if bulk_str.count("x") % 2 == 1:

        # check if number of Y, Z operators is odd
        if bulk_str.count("y") % 2 == 1 and bulk_str.count("z") % 2 == 1:

            x_str = "x" + bulk_str + "x"

            # check if reflection in list
            if x_str[::-1] in op_list_x:
                continue
            else:
                op_list_x.append(x_str)


# others (x...y or x...z)
for bulk_str in bulk_strings:

    if bulk_str.count("x") % 2 == 0:

        # check if number of Y operators is odd
        if bulk_str.count("y") % 2 == 1 and bulk_str.count("z") % 2 == 0:
                x_str = "x" + bulk_str + "z"
                op_list_x.append(x_str)

        elif bulk_str.count("y") % 2 == 0 and bulk_str.count("z") % 2 == 1:
                x_str = "x" + bulk_str + "y"
                op_list_x.append(x_str)


# y
# equal suffix
for bulk_str in bulk_strings:
    if bulk_str.count("y") % 2 == 1 and bulk_str.count("z") % 2 == 1 and bulk_str.count("x") % 2 == 1:

        y_str = "y" + bulk_str + "y"

        # check if reflection in list
        if y_str[::-1] in op_list_y:
            continue
        else:
            op_list_y.append(y_str)

# others
for bulk_str in bulk_strings:
    if bulk_str.count("y") % 2 == 0 and bulk_str.count("z") % 2 == 0 and bulk_str.count("x") % 2 == 1:

        y_str = "y" + bulk_str + "z"
        op_list_y.append(y_str)

# z
# equal suffix
for bulk_str in bulk_strings:

    if bulk_str.count("y") % 2 == 1 and bulk_str.count("z") % 2 == 1 and bulk_str.count("x") % 2 == 1:

        z_str = "z" + bulk_str + "z"

        # check if reflection in list
        if z_str[::-1] in op_list_z:
            continue
        else:
            op_list_z.append(z_str)


# print(size, size / 3)
# print(len(op_list_x) + len(op_list_y) + len(op_list_z), len(op_list_x), len(op_list_y), len(op_list_z) )
# print(op_list_x)
# print(op_list_y)
# print(op_list_z)

with open("operators_TPFXY_l" + str(l) + ".txt", "w") as writefile:

    for op in op_list_x:
        writefile.write(op + "\n")
    for op in op_list_y:
        writefile.write(op + "\n")
    for op in op_list_z:
        writefile.write(op + "\n")

writefile.close()