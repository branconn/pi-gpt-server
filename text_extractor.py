from PyPDF2 import PdfReader
import hashlib, glob, string, pickle

def _get_paths(re_path:str) ->list[str]:
    files = glob.glob(re_path)
    return files


def _hash(file):
    BUF_SIZE = 65536 # 64 kB
    sha256 = hashlib.sha256()

    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def extract_all(punctuation=True):
    hash_table = {}
    
    with open("raw_text.txt",'w') as txt_file:

        # check against recorded hashes

        for file in _get_paths():
            reader = PdfReader(file)
            for page in reader.pages:
                extracted_txt = page.extract_text()
                if not punctuation:
                    extracted_txt = extracted_txt.translate(str.maketrans('', '', string.punctuation))
                txt_file.write(extracted_txt)

            hash_table[file] = _hash(file)

    pickle.dump(hash_table, open('f_hashes.p', 'wb'))
    print('success')


def checkHashes(pik:str) -> int:
    p_path = 'pickles/'+pik
    diff = 0
    try:
        h_dict = pickle.load(p_path)
        for file in _get_paths():
            if file not in h_dict:
                diff += 1
                print(file,'not tracked.')
            elif h_dict[file] != _hash(file):
                diff += 1
                print(file,'changed')
            
    except:
        print('pickle not found in pickles directory')
        return -1

def combine_txts():
    file_list = _get_paths('training_texts/*.txt')
    with open('training_data.txt','wb') as dest:
        for file in file_list:
            with open(file, 'rb') as source:
                dest.write(source.read())
                # dest.write('\n')

def clean_txt(path):
    with open(path, 'r') as source_f:
        new_f = path[:-4]+'_cleaned.txt'
        data = source_f.read()
        lowers = set(range(97,123))
        text_list = list(data)

        for i, char in enumerate(text_list):
            if char == '\n':
                print('newline',text_list[i+1],ord(text_list[i+1]))
                if ord(text_list[i+1]) in lowers or text_list[i+1] == '\n':
                    if text_list[i-1] == '-':
                        del text_list[i-1:i+1]
                    else: 
                        del text_list[i]

        new_text = ''.join(text_list)

        with open(new_f, 'w') as dest_f:
            dest_f.write(new_text)


if __name__ == "__main__":
    # combine_txts()
    # if checkHashes('f_hashes.p') != 0:
    #     extract_all(punctuation=True)
    # clean_txt('training_data.txt')
    # print('success')
    pass