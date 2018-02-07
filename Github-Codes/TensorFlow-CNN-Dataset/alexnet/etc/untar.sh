#!bin/bash
for file in *.tar
do
    mkdir${file%.*} && tar xvf ${file} -C ${file%.*}
done
