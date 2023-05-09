from checkpoint_schedules.hrevolve_sequence import \
        (disk_revolve, revolve_1d, periodic_disk_revolve, hrevolve)


def test_hrevolve_l2():
    # Check that we reproduce Listing 2 from Herrmann and Pallez (2020).
    sequence = hrevolve(25, [1, 1, 0], [0, 2, 3], [0, 2, 3])
    print(sequence)
    # assert sequence.storage[0] == [17, 14, 12, 9, 7, 3, 0]
    # assert sequence.storage[1] == [7, 12]
    # assert sequence.storage[2] == [0]
    # assert sequence.makespan == 89
    # compare_sequences(
    #     str(sequence), '[W^2_0, F_0->6, W^1_7, F_7->11, W^1_12,'
    #     ' F_12->16, W^0_17, F_17->19, B_20, R^0_17, F_17->18, B_19, R^0_17,'
    #     ' F_17, B_18, R^0_17, B_17, D^0_17, R^1_12, F_12->13, W^0_14,'
    #     ' F_14->15, B_16, R^0_14, F_14, B_15, R^0_14, B_14, D^0_14, R^1_12,'
    #     ' W^0_12, F_12, B_13, R^0_12, B_12, D^0_12, R^1_7, F_7->8, W^0_9,'
    #     ' F_9->10, B_11, R^0_9, F_9, B_10, R^0_9, B_9, D^0_9, R^1_7, W^0_7,'
    #     ' F_7, B_8, R^0_7, B_7, D^0_7, R^2_0, F_0->2, W^0_3, F_3->5, B_6,'
    #     ' R^0_3, F_3->4, B_5, R^0_3, F_3, B_4, R^0_3, B_3, D^0_3, R^2_0,'
    #     ' W^0_0, F_0->1, B_2, R^0_0, F_0, B_1, R^0_0, B_0, D^0_0]'
    # )


def test_disk_revolve_l3():
    # Check that we reproduce Listing 3 from Herrmann and Pallez (2020).
    sequence = disk_revolve(l=10, cm=2, wd=2, rd=1, ub=0)
    print(sequence)
    # assert sequence.memory == [5, 8, 6, 0, 3, 1]
    # assert sequence.disk == [0]
    # compare_sequences(
    #     str(sequence), '[WD_0, F_0->4, WM_5, F_5->7, WM_8, F_8->9, B_10,'
    #     ' RM_8, F_8, B_9, RM_8, B_8, DM_8, RM_5, F_5, WM_6, F_6, B_7, RM_6,'
    #     ' B_6, DM_6, RM_5, B_5, DM_5, RD_0, WM_0, F_0->2, WM_3, F_3, B_4,'
    #     ' RM_3, B_3, DM_3, RM_0, F_0, WM_1, F_1, B_2, RM_1, B_1, DM_1, RM_0,'
    #     ' B_0, DM_0]'
    # )
    # assert sequence.makespan == 22


def test_disk_revolve_l4():
    # Check that we reproduce Listing 4 from Herrmann and Pallez (2020).
    sequence = disk_revolve(l=100, cm=2, wd=10, rd=2, ub=0, concat=2)
    print(sequence)
    # compare_sequences(
    #     str(sequence), '[WD_0, F_0->15, WD_16, F_16->31, WD_32,'
    #     ' F_32->47, WD_48, F_48->63, WD_64, F_64->79, WD_80, F_80->90,'
    #     ' Revolve(9, 2), RD_80, revolve_1d1D-Revolve(10, 2), RD_64, 1D-Revolve(15, 2),'
    #     ' RD_48, 1D-Revolve(15, 2), RD_32, 1D-Revolve(15, 2), RD_16,'
    #     ' 1D-Revolve(15, 2), RD_0, 1D-Revolve(15, 2)]'
    # )


def test_revolve_1d_l4():
    # Check that we reproduce Listing 4 from Herrmann and Pallez (2020).
    sequence = revolve_1d(l=15, cm=2, rd=2, concat=1)
    print(sequence)
    # compare_sequences(
    #     str(sequence), '[F_0->5, Revolve(9, 2), RD_0, Revolve(5, 2)]'
    # )


def test_disk_revolve_l5():
    # Check that we reproduce Listing 5 from Herrmann and Pallez (2020).
    sequence = disk_revolve(l=100, cm=2, wd=5, rd=2, ub=0, concat=3)
    print(sequence)
    # compare_sequences(
    #     str(sequence), "(16, 16, 16, 16, 16, 11; 10)"
    # )


def test_periodic_disk_revolve_l6():
    # Check that we reproduce Listing 6 from Herrmann and Pallez (2020).
    sequence = periodic_disk_revolve(l=10, cm=2, wd=2, rd=1, ub=0)
    print(sequence)
    # compare_sequences(
    #     str(sequence), '[WD_0, F_0->2, WD_3, F_3->5, WD_6, F_6->8, WM_9,'
    #     ' F_9, B_10, RM_9, B_9, DM_9, RD_6, WM_6, F_6, WM_7, F_7, B_8, RM_7,'
    #     ' B_7, DM_7, RM_6, B_6, DM_6, RD_3, WM_3, F_3, WM_4, F_4, B_5, RM_4,'
    #     ' B_4, DM_4, RM_3, B_3, DM_3, RD_0, WM_0, F_0, WM_1, F_1, B_2, RM_1,'
    #     ' B_1, DM_1, RM_0, B_0, DM_0]'
    # )
# test_hrevolve_l2()test_disk_revolve_l3
# test_disk_revolve_l3()
# test_disk_revolve_l4()


def test_periodic_disk_revolve_l7():
    # Check that we reproduce Listing 7 from Herrmann and Pallez (2020).
    sequence = periodic_disk_revolve(l=10, cm=2, wd=2, rd=1, ub=0,
                                     one_read_disk=True)
    print(sequence)
    # compare_sequences(
    #     str(sequence), '[WD_0, F_0->2, WD_3, F_3->5, WD_6, F_6->8, WM_9,'
    #     ' F_9, B_10, RM_9, B_9, DM_9, RD_6, WM_6, F_6, WM_7, F_7, B_8, RM_7,'
    #     ' B_7, DM_7, RM_6, B_6, DM_6, RD_3, WM_3, F_3, WM_4, F_4, B_5, RM_4,'
    #     ' B_4, DM_4, RM_3, B_3, DM_3, RD_0, WM_0, F_0, WM_1, F_1, B_2, RM_1,'
    #     ' B_1, DM_1, RM_0, B_0, DM_0]'
    # )
    # assert sequence.memory == [9, 6, 7, 3, 4, 0, 1]
    # assert sequence.disk == [0, 3, 6]
    # assert sequence.makespan == 25

test_hrevolve_l2()
# test_disk_revolve_l3()
# test_disk_revolve_l4()
# test_revolve_1d_l4()
# test_disk_revolve_l5()
# test_periodic_disk_revolve_l6()
# test_periodic_disk_revolve_l7()
