
archive_name=master.zip
dataset_name=mnist_png.tar.gz
temp_dir=tmp

wget https://github.com/myleott/mnist_png/archive/refs/heads/$archive_name
mkdir $temp_dir
unzip -j "$archive_name" "mnist_png-master/$dataset_name" -d "."
tar -xf "mnist_png.tar.gz" -C "$temp_dir/."

rm $archive_name $dataset_name