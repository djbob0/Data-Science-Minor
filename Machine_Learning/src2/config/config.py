import pprint

class config:
    debug = True
    """ graph """
    graph = True
    
    AF= True
    EL= True
    AB= True
    RF= True
    EH= True

    exer_list = {'AF': True,
            'EL' : True,
            'AB' : True,
            'RF' : True,
            'EH' : True}

    pg1 = True
    pg2 = True
    pg3 = True
    

    basepath = "D:\\Hochschule\\5_Semester\\Orthoeyes\\Python\\src2\\data\\Catagory_{groupid}"
    pp = pprint.PrettyPrinter(indent=4)
    exercisetypes = ['AF', 'EL', 'AB', 'RF', 'EH']

    columns = ["thorax_r_x", "thorax_r_y", "thorax_r_z",
               "clavicula_r_x", "clavicula_r_y", "clavicula_r_z",
               "scapula_r_x", "scapula_r_y", "scapula_r_z",
               "humerus_r_x", "humerus_r_y", "humerus_r_z",
               "ellebooghoek_r",
               "thorax_l_x", "thorax_l_y", "thorax_l_z",
               "clavicula_l_x", "clavicula_l_y", "clavicula_l_z",
               "scapula_l_x", "scapula_l_y", "scapula_l_z",
               "humerus_l_x", "humerus_l_y", "humerus_l_z",
               "ellebooghoek_l"]
    frames_counts = 5

    if debug:
        test_selections = { '1': [1,6,8,10],
                            '2': [1, 14],
                            '3': [4,11],
                            '4': [1,5,18]
                            }
    else:
        test_selections = { '1': [6],
                            '2': [],
                            '3': [4],
                            '4': []
                            }