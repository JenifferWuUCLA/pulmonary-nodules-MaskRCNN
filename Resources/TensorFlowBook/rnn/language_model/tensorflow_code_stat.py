import os


# source code file count result:
#     cc  1309
#     md  1242
#     py  1235
#     h 734
#     html  107
#     ts  95
#     sh  81
#     proto 57
#     json  49
#     java  34
def get_file_suffix(file):
    index = file.rfind('.')
    return file[index + 1:] if index >= 0 else ''


# char frequency in py source code
def tensorflow_code_stat(data_path):
    tensorflow_code_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            code_file = os.path.join(root, file)
            file_suffix = get_file_suffix(code_file)
            if file_suffix == 'py':
                tensorflow_code_files.append(code_file)

    dict_c = {}
    for code_file in tensorflow_code_files:
        file_r = open(code_file, 'r')
        for data in file_r:
            for c in data:
                if c in dict_c:
                    dict_c[c] += 1
                else:
                    dict_c[c] = 1
        file_r.close()

    for c, c_freq in dict_c.iteritems():
        print(c + '\t' + str(ord(c)) + '\t' + str(c_freq))


if __name__ == "__main__":
    tensorflow_code_stat('../../../../../tensorflow/tensorflow/')
