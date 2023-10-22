# -*- coding: utf-8 -*-
"""

"""
from numpy import *
import scipy
import json
import core

comm = core.core()

class equation(dict):
    '''
    Dict class which describes equations of operators
    Keys are pauli strings
    values are the numerical prefactors of the strings.
     values can be an array, in which case the equation
     can be seen as a functional of those values

    Has all the standard operations for a Vector Field:
    - Addition / Subtraction
    - Multiplication by scalar
    - Multiplication by self-elements (in XYZ1 format)
    '''

    def __call__(self, index):
        '''
        Returns the equation evaluated at a certain index
        !! Only works if the thing is a functional !!
        ( don't be dumb )
        '''
        return equation({a: b[tuple(index)] for a, b in self.iteritems()})

    def tol(self, tol=1e-10, action='rel', verbose=False):
        '''
        action: 'rel' - Scrubs all values which are [ tol ] smaller then the largest value
        action: 'abs' - Scrubs all values which are smaller then [ tol ]
        '''

        Ninitial = len(self)
        if Ninitial == 0:
            return equation({})
        amin = tol * {'rel': (abs(array(self.values())).max()), 'abs': 1}[action]
        dout = {}
        for key, dd in self.iteritems():
            if type(dd) != ndarray:
                if abs(dd) > amin:
                    dout[key] = dd
            else:
                if average(abs(dd)) > amin:  # array(abs(dd)>1e-15).prod():#
                    dout[key] = dd
        # out =  {a:b for a,b in self.iteritems() if average(abs(b))>amin}
        if verbose:
            print Ninitial - len(dout), 'terms killed'
        return equation(dout)

    def __add__(self, B):
        # print type(B),isinstance(B,equation)
        if isinstance(B, dict):
            return equation(comm.add(self, B))
        else:
            raise TypeError("Adding by the wrong type!")

    def __sub__(self, B):
        if isinstance(B, dict):
            return equation(comm.add(self, B, b=-1))
        else:
            raise TypeError("Adding by the wrong type!")

    def __rmul__(self, L):
        # L * self
        if isinstance(L, dict):
            return equation(comm.product(L, self))
        else:  # Multiplication by a scalar
            return equation(comm.add(self, equation({}), a=L))

    def __mul__(self, R):
        # self * R
        # print 'MUL:', type(R),isinstance(R,equation)
        if isinstance(R, dict):
            return equation(comm.product(self, R))
        else:  # Multiplication by a scalar
            return equation(comm.add(self, equation({}), a=R))

    def __hash__(self):
        return hash(json.dumps(self))

    def __str__(self):
        strout = ''
        for x in argsort(abs(array(self.values()))):
            strout += self.keys()[x] + '\t' + str(self.values()[x]) + '\n'
        return strout

    def conjugate(self):
        return equation({a: conj(b) for a, b in self.iteritems()})

    def astype(self, type_):
        return equation({a: type_(b) for a, b in self.iteritems()})

    def printit(self):
        '''
        Print a formula, latex style
        '''
        C = self
        argva = argsort(abs(array(C.values())))[-1::-1]
        strout = ''
        linebreak = 1
        for a_ in argva:
            payload = ''
            kk = 0
            for x in C.keys()[a_]:
                if x != '1':
                    payload = payload + r"\hat\sigma^{" + "{:0.0f}".format(kk) + "}_" + x
                kk += 1
            if abs(C.values()[a_]) < 1e-10:
                pass
            elif abs(imag(C.values()[a_])) < 1e-15:
                strout += '({:1.2e})'.format(float(real(C.values()[a_]))) + payload + '+'
            elif abs(real(C.values()[a_])) < 1e-10:
                strout += 'i*({:1.2e})'.format(float(imag(C.values()[a_]))) + payload + '+'
            else:
                strout += '({:1.2e} + i*{:1.2e})'.format(real(C.values()[a_]), imag(C.values()[a_])) + payload + '+'
            if len(strout) > 150 * linebreak:
                strout += r"\\"
                linebreak += 1
        print strout[0:-1]

    # get trace of given equation
    # since only the has a finite trace one only needs to check if it exists
    # if yes the trace is the coefficient times 2 ** N
    def trace(self):

        if len(self.keys()):

            number_of_spins = len(self.keys()[0])
            id_string = number_of_spins * "1"

            # check and return
            if id_string in self.keys():

                return self[id_string] * (2 ** number_of_spins)

            else:

                return 0.

        else:

            return 0.


class maths:
    def __init__(self):
        '''
        Handy Math things!
        '''
        pass  # hehe

    def c(self, A, B):
        '''
        Commutes two equations A and B
        '''
        # return equation(comm.fastcommute(A,B)).tol(1e-15) # Faster by a factor of 10!
        return A * B - (B * A)

    def ac(self, A, B):
        '''
        Anti-Commute two equations A and B
        '''
        return A * B + (B * A)

    def tracedot(self, A, B):
        '''
        Calculates Tr[AB]/D
        Assumes XYZ1 format!!
        '''
        # Check if it is XYZ1 format... Comment if slow?
        if len(set(''.join(A.keys()) + ''.join(B.keys())).difference(['1', 'x', 'y', 'z'])) > 0:
            raise BaseException("Equation is of the wrong format!")
        dout = 0
        for common in set(A.keys()).intersection(B.keys()):
            dout += A[common] * B[common]
        return dout

    def BCH(self, A, H, order=5, precision=1e-10):
        '''
        Does the Baker Campbell Hausdorff espansion of some rotation A on
         some operator H

        e^{i*[ A ]} [ H ] e^{-i*[ A ]}

        To an order specified. All terms which are less then [ precision ]
        are truncated at every order.
        If order is too large, the BCH is run out until the rotation error is
        of the order of 1e-5.
        '''
        trH = sum(real(H.values()) ** 2)
        # Copied then modified from commute_stringgroups.py
        dout2 = equation({})
        B = H
        for oord in range(order):
            print '({:0.0f}, {:0.0f})'.format(oord, len(dout2)),
            dout2 += ((1j) ** oord / scipy.special.factorial(oord)) * B

            if abs((trH - sum(real(dout2.values()) ** 2)) / trH) < 1e-7 and oord > 1:
                print (trH - sum(real(dout2.values()) ** 2)) / trH, 'BCH STOPPING EARLY!'
                return dout2
            B = self.c(A, B)
            B = B.tol(precision)

        dout2 += ((1j) ** (oord + 1) / scipy.special.factorial(oord + 1)) * B

        return dout2

    def magnus(self, A, t_max, order=3, precision=1e-10):
        '''
        Calculates the following operator via the Magnus expansion,
         up to 4th order

        B = i LOG(T e^{i*integrate(A)})

        - A is a dict equation, with values being vectors at different times
        - t_max is the maximum evolution time which sets normalization of the integrals

        From:
            Blanes, S., Casas, F., Oteo, J. A., & Ros, J. (2009).
            The Magnus expansion and some of its applications.
             Physics Reports, 470(5–6), 151–238.
             https://doi.org/10.1016/j.physrep.2008.11.001
        '''

        # Copied then Modified from find_Alambda.py
        Omega = equation({})
        print 'First Order...'
        # First Order...
        omega_1 = equation({a: cumsum(b) * t_max / len(b) for a, b in A.iteritems()})

        omega1_end = equation({a: b[-1] for a, b in omega_1.iteritems()})
        Omega += omega1_end
        if order == 1:
            return Omega

        # Quell small terms...
        Omega = Omega.tol(precision)
        omega_1 = omega_1.tol(precision)

        print 'Second Order...'
        # Second Order...
        omega1A = 1j * self.c(omega_1, A)
        omega_2_sum = -0.5 * omega1A
        omega_2 = equation({a: cumsum(b) * t_max / len(b) for a, b in omega_2_sum.iteritems()})

        omega2_end = equation({a: b[-1] for a, b in omega_2.iteritems()})
        Omega += omega2_end
        if order == 2:
            return Omega

        # Quell small terms...
        Omega = Omega.tol(precision)
        omega1A = omega1A.tol(precision)
        omega_2 = omega_2.tol(precision)

        print 'Third Order...'
        # Third Order...
        omega2A = 1j * self.c(omega_2, A)
        omega11A = 1j * self.c(omega_1, omega1A)
        omega_3_sum = -0.5 * omega2A + 1. / 12 * omega11A
        omega_3 = equation({a: cumsum(b) * t_max / len(b) for a, b in omega_3_sum.iteritems()})

        omega3_end = equation({a: b[-1] for a, b in omega_3.iteritems()})
        Omega += omega3_end
        if order == 3:
            return Omega

        # Quell small terms...
        Omega = Omega.tol(precision)
        omega2A = omega2A.tol(precision)
        omega11A = omega11A.tol(precision)
        omega_3 = omega_3.tol(precision)

        print 'Fourth Order...'
        # Fourth Order!
        omega3A = 1j * self.c(omega_3, A)
        omega12A = 1j * self.c(omega_1, omega2A)
        # print type(omega_2),type(omega1A)
        omega21A = 1j * self.c(omega_2, omega1A)
        omega_4_sum = -0.5 * omega3A + 1. / 12 * omega12A + 1. / 12 * omega21A
        omega_4 = equation({a: cumsum(b) * t_max / len(b) for a, b in omega_4_sum.iteritems()})

        omega4_end = equation({a: b[-1] for a, b in omega_4.iteritems()})
        Omega += omega4_end

        # Quell small terms...
        Omega = Omega.tol(precision)
        # Other terms are never used again...

        if order == 4:
            return Omega
        # Fith order?
        #  ...lol

        raise NotImplemented("I have only done up to fourth order!")


# list of all shifts of a given string

def shifts(string):
    sl = []
    l = len(string)
    for i in range(l):
        # split string and recombine
        left = string[0: l - i]
        right = string[l - i:]
        sl.append(right + left)

    return sl


def shifts_and_reflection(string):
    sl = []
    l = len(string)
    for i in range(l):
        # parity
        sl.append(string[::-1])

        # shift
        # split string and recombine
        left = string[0: l - i]
        right = string[l - i:]

        shifted = right + left
        sl.append(shifted)

        # shift and parity
        sl.append(shifted[::-1])

    return sl


def x_flip(string):
    mul = 1.
    for s in string:

        if s == "1" or s == "x":
            continue
        else:
            mul *= -1.

    return mul


# split equation into periodic operators
def split_equation_shifts(eq):
    # list of operators, that the list is split into
    # each will consists of all translations of a pauli string (could be  periodic or obc)
    # for each op we take the first occurrence as the representative for this operator

    # need to put in some thought for non-translationally invariant systems !!!

    op_list = []
    rep_list = []

    # split the equation

    # print(equation({eq.keys()[0]: 1.}))

    # add the first equation entry to the list
    op_list.append(equation({eq.keys()[0]: 1.}))
    rep_list.append(eq.keys()[0])

    for i in range(len(eq.keys()) - 1):

        op = eq.keys()[i + 1]
        # print(i, op)
        # check if the operator is a translated version of all known representatives
        check = 0
        for j, rep in enumerate(rep_list):

            # print(j, rep, shifts(rep))

            if op in shifts(rep):
                op_list[j] += equation({op: 1.})
                check = 1
                # print("similar to rep", op_list)
                break

        # none of the representatives are similar
        # add the key as new representative and new op group
        if check == 0:
            op_list.append(equation({eq.keys()[i + 1]: 1.}))
            rep_list.append(eq.keys()[i + 1])
            # print("not similar", op_list)
            # print(rep_list)

    # set all values to 1

    for op in op_list:
        for string in op:
            op[string] = 1.

    return op_list


# compute the length of a pauli string, which is the smallest distance enclosed by pauli operators
# for example 1XY11 has length 2, while 1X1Y1 has length 3
# the length though can depend on the boundary conditions (periodic or open)
# for example X111Y has length 5 with open boundaries but length 2 with periodic boundaries

def string_range(pauli_string, boundary_condition):
    l = len(pauli_string)
    if boundary_condition == "obc":

        left_ones = 0
        right_ones = 0

        for char in pauli_string:

            if char == "1":

                left_ones += 1

            else:

                break

        if left_ones == l or left_ones == l - 1:

            count = l - left_ones

        else:

            for char in pauli_string[::-1]:

                if char == "1":

                    right_ones += 1

                else:

                    break

            count = l - left_ones - right_ones

        # print("left:", left_ones)
        # print("right:", right_ones)

    elif boundary_condition == "pbc":

        # check if identity
        if pauli_string == "1" * l:

            count = l

        else:
            double_string = pauli_string * 2

            count = 0
            position = 0

            while position < l:

                # print(position)
                # print(count)
                running_count = 0
                for char in double_string[position::]:

                    if char == "1":

                        running_count += 1

                    else:

                        position += running_count + 1
                        if running_count > count:
                            count = running_count

                        break

        count = l - count

    else:

        raise ValueError("Unknown Boundary Condition:", boundary_condition)

    return count


# reduce an equation to pauli strings up to maximal range
def split_equation_length_cutoff(eq, cutoff, boundary_condition):
    op_list = []
    for op in eq.keys():

        # check if the operator is shorter or equal to the maximal length

        op_range = string_range(op, boundary_condition)
        if op_range <= cutoff:

            op_list.append(equation({op: 1.}))

        else:

            continue

    return op_list


# reduce an equation to pauli strings up to maximal range
def split_equation_periodically_length_cutoff(eq, cutoff, boundary_condition):
    # list of operators, that the list is split into
    # each will consists of all translations of a pauli string (could be  periodic or obc)
    # for each op we take the first occurrence as the representative for this operator

    # need to put in some thought for non-translationally invariant systems !!!

    op_list = []
    rep_list = []

    # split the equation

    # print(equation({eq.keys()[0]: 1.}))

    # add the first equation entry to the list
    op_list.append(equation({eq.keys()[0]: 1.}))
    rep_list.append(eq.keys()[0])

    for i in range(len(eq.keys()) - 1):

        op = eq.keys()[i + 1]
        # print(i, op)

        # check length
        op_range = string_range(op, boundary_condition)
        if op_range <= cutoff:

            # check if the operator is a translated version of all known representatives
            check = 0
            for j, rep in enumerate(rep_list):

                # print(j, rep, shifts(rep))

                if op in shifts(rep):
                    op_list[j] += equation({op: 1.})
                    check = 1
                    # print("similar to rep", op_list)
                    break

            # none of the representatives are similar
            # add the key as new representative and new op group
            if check == 0:
                op_list.append(equation({eq.keys()[i + 1]: 1.}))
                rep_list.append(eq.keys()[i + 1])
                # print("not similar", op_list)
                # print(rep_list)

        else:

            continue

    # set all values to 1

    for op in op_list:
        for string in op:
            op[string] = 1.

    return op_list


# reduce an equation to pauli strings up to maximal range
def split_equation_TP_length_cutoff(eq, cutoff, boundary_condition):
    # list of operators, that the list is split into
    # each will consists of all translations of a pauli string (could be  periodic or obc)
    # for each op we take the first occurrence as the representative for this operator

    # need to put in some thought for non-translationally invariant systems !!!

    op_list = []
    rep_list = []

    # split the equation

    # print(equation({eq.keys()[0]: 1.}))

    # add the first equation entry to the list
    op_list.append(equation({eq.keys()[0]: 1.}))
    rep_list.append(eq.keys()[0])

    for i in range(len(eq.keys()) - 1):

        op = eq.keys()[i + 1]
        # print(i, op)

        # check length
        op_range = string_range(op, boundary_condition)

        if op_range <= cutoff:

            # check if the operator is a translated and reflected version of all known representatives
            check = 0
            for j, rep in enumerate(rep_list):

                # print(j, rep, shifts(rep))

                if op in shifts_and_reflection(rep):
                    op_list[j] += equation({op: 1.})
                    check = 1
                    # print("similar to rep", op_list)
                    break

            # none of the representatives are similar
            # add the key as new representative and new op group
            if check == 0:
                op_list.append(equation({eq.keys()[i + 1]: 1.}))
                rep_list.append(eq.keys()[i + 1])
                # print("not similar", op_list)
                # print(rep_list)

        else:

            continue

    # set all values to 1

    for op in op_list:
        for string in op:
            op[string] = 1.

    return op_list


# reduce an equation to pauli strings up to maximal range
def split_equation_TPF_length_cutoff(eq, cutoff, boundary_condition):
    # list of operators, that the list is split into
    # each will consists of all translations of a pauli string (could be  periodic or obc)
    # for each op we take the first occurrence as the representative for this operator

    # need to put in some thought for non-translationally invariant systems !!!

    op_list = []
    rep_list = []

    # split the equation

    # print(equation({eq.keys()[0]: 1.}))

    # add the first equation entry to the list
    op_list.append(equation({eq.keys()[0]: 1.}))
    rep_list.append(eq.keys()[0])

    for i in range(len(eq.keys()) - 1):

        op = eq.keys()[i + 1]
        # print(i, op)

        # check length
        op_range = string_range(op, boundary_condition)

        if op_range <= cutoff:

            # check spi flip symmetry
            F = x_flip(op)

            if F == 1.:

                # check if the operator is a translated and reflected version of all known representatives
                check = 0
                for j, rep in enumerate(rep_list):

                    # print(j, rep, shifts(rep))

                    if op in shifts_and_reflection(rep):
                        op_list[j] += equation({op: 1.})
                        check = 1
                        # print("similar to rep", op_list)
                        break

                # none of the representatives are similar
                # add the key as new representative and new op group
                if check == 0:
                    op_list.append(equation({eq.keys()[i + 1]: 1.}))
                    rep_list.append(eq.keys()[i + 1])
                    # print("not similar", op_list)
                    # print(rep_list)

            else:

                continue

    # set all values to 1

    for op in op_list:
        for string in op:
            op[string] = 1.

    return op_list