
#!/bin/bash
python train_test.py --city_name Glas --batch_size 8 --n_poiCat 8 --n_traj_len 9 | tail -n 20 > glas_output.txt

python train_test.py --city_name Osak --batch_size 8 --n_poiCat 5 --n_traj_len 7 | tail -n 20 > osak_output.txt

python train_test.py --city_name Melb --batch_size 8 --n_poiCat 10 --n_traj_len 21 | tail -n 20 > melb_output.txt

python train_test.py --city_name Toro --batch_size 8 --n_poiCat 7 --n_traj_len 14 | tail -n 20 > toro_output.txt

python train_test.py --city_name Edin --batch_size 16 --n_poiCat 7 --n_traj_len 14 | tail -n 20 > edin_output.txt

python train_test.py --city_name TKY_split200 --batch_size 8 --n_poiCat 3 --n_traj_len 8 | tail -n 20 > tky_output.txt


echo "All experiments are complete."



