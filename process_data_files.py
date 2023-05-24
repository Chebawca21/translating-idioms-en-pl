#Generate one idiom file 
import os

def full_path(file_name: str, dir: str ) -> str:
    return os.path.join("", dir, file_name)


def fetch_sent_from_file(full_name: str):
    sent_list = []
    eng_list = []
    pl_list = []
    with open(full_name, encoding = 'utf8') as file:
        for id, line in enumerate(file, 1):
            print("Process line:" + str(id))
            if id%4==1:
                eng_list.append(line)
            if id%4==2:
                pl_list.append(line)
            if id%4==0:
                sent_list.append(line)
    return sent_list, eng_list, pl_list
    
def save_to_file( sent_list:list[str], file_name:str = "processed_data.txt") -> None:
    with open(file_name, mode="w",  encoding = 'utf8') as file:
        file.writelines(sent_list)
    
    
if __name__ == '__main__':
    dir = 'data_pre'
    file_list = ['data1.txt','data2.txt','data4.txt','data5.txt','data6.txt', 'data3.txt']
    
    all_sent = []
    all_idioms = []
    all_trans = []
    
    for file in file_list:
        print("Process file:" + file)
        full_name = full_path(file, dir)
        sent_list, eng_list, pl_list  = fetch_sent_from_file(full_name)
        all_sent.extend(sent_list)
        all_idioms.extend(eng_list)
        all_trans.extend(pl_list)
    
    #save_to_file(all_sent, "ex_eng.txt")
    save_to_file(all_idioms, "data_after\idom_eng.txt")
    save_to_file(all_trans, "data_after\idom_trans.txt")