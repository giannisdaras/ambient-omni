cats_dataset_path=./data/afhqv2-64x64-partitioned/0 # cats, replace with your own
dogs_dataset_path=./data/afhqv2-64x64-partitioned/1 # dogs, replace with your own
./scripts/ood_dataset_utils/merge_datasets_for_train_crops_classifier.sh ./data/cats-100_and_dogs-100 $cats_dataset_path $dogs_dataset_path