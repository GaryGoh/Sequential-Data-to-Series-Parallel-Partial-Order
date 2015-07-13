__author__ = 'GaryGoh'


def average_waiting_time(customer):
    """ A typical queueing bank system problem
    For simplify, this function will not provide random number to empirical probability distribution, instead, only provide a user-input as below describing.
    And no optimise to the algorithm.

    in this way, we also have the time log of each staff, and further sorting can obtain the log of each customer.


    parameter
    ---------
    customer: int tuple list, (arrive_time, process_time)

    return
    ---------
    average_waiting_time: int

    """
    # Initialize parameters
    waiting_time = 0
    staff = {1: [0], 2: [0], 3: [0], 4: [0]}

    for i in customer:
        # (arrive_time, process_time)
        total_time = customer[0] + customer[1]

        min_endtime_staff = min(staff.items(), key = lambda x: x[1][0])

        # if the earliest end time of one of the staff < arrive_time
        if min_endtime_staff[1][-1] < customer[0]:
            staff[min_endtime_staff[0]].append(total_time)
        else:
            waiting_time += customer[0] - min_endtime_staff[1][-1]
            staff[min_endtime_staff[0]].append(total_time)

    return float(waiting_time/(len(customer)))

