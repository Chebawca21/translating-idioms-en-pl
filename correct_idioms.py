from process_data_files import  full_path

def get_list_from_file(file_name:str):
    listed = list[str]
    with open(file_name) as file:
        listed = file.readlines()
    return listed

if __name__ == '__main__':
    dir =       'data_pre'
    idom_file = 'idom_eng.txt'
    idom_trans= 'idom_trans.txt'
    eng_sent =  'processed_data.txt'
    pl_sent =   'translated_data.txt'
    
    idoms = get_list_from_file(full_path(idom_file, dir))
    trans = get_list_from_file(full_path(idom_trans, dir))
    eng = get_list_from_file(full_path(eng_sent, dir))
    pl = get_list_from_file(full_path(pl_sent, dir))
    
    
