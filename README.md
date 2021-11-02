# CSE-515
Task 1:
Command Syntax: python/python3 task1.py -fp [folder_path] -f [feature_descriptor] -x [image_type] -k [latent_semantics_num] -t [dimensionality_reduction_technique]
Example: python3 task1.py -fp all -f cm -x cc -k 10 -t lda

Task 2:
Command Syntax: python/python3 task2.py -fp [folder_path] -f [feature_descriptor] -y [subject] -k [latent_semantics_num] -t [dimensionality_reduction_technique]
Example: python3 task2.py -fp all -f cm -y 2 -k 10 -t lda

Task 3:
Command Syntax: python/python3 task3.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -t [dimensionality_reduction_technique]
Example: python3 task3.py -fp all -f cm -k 10 -t lda

Task 4:
Command Syntax: python/python3 task4.py -fp [folder_path] -f [feature_descriptor] -k [latent_semantics_num] -t [dimensionality_reduction_technique]
Example: python3 task4.py -fp all -f cm -k 10 -t lda

Task 5:
Command Syntax: python/python3 task5.py -fp [folder_path] -lp [latent_semantic_path] -k [top_k_num]
Example: python3 task5.py -fp all -lp ‘../all/latent_semantics_elbp_lda_1_10.json’ -k 10

Task 6:
Command Syntax: python/python3 task6.py -fp [folder_path] -lp [latent_semantic_path]
Example: python3 task6.py -fp all -lp ‘../all/latent_semantics_elbp_lda_1_10.json’

Task 7:
Command Syntax: python/python3 task7.py -fp [folder_path] -lp [latent_semantic_path]
Example: python3 task7.py -fp all -lp ‘../all/latent_semantics_elbp_lda_1_10.json’

Task 8:
Command Syntax: python task8.py -sp=[subject-subject similarity matrix path] -n =[number of neighbors] -m=[top m significant subjects to be found]
Example: python task8.py -sp=C:\Users\rshah35\Downloads\all\sub_sub_hog.json -n=30 -m=5

Task 9: 
Command Syntax: python task9.py -sp=[subject-subject similarity matrix path] -n=[top similar images] -m=[top important images] -subid=[3 subject IDs]
Example: python /Users/zealpatel/CSE-515/task9.py -sp="/Users/zealpatel/Documents/JupyterFiles/Phase2_data/all/sub_sub_cm.json" -n=10 -m=5 -subid=1-2-3

