import random
#SEED = 1 for places_LT_class_order_seed_1.txt


SEED=1
random.seed(SEED)

class_order_text_file = 'places_LT_class_order_original.txt'
### load in user desired order from text file
with open(class_order_text_file) as f:
    lines = [line.rstrip() for line in f]  # grab each class name from line in text file

random.shuffle(lines)

with open(f'places_LT_class_order_seed_{SEED}.txt', 'w') as f:
    for item in lines:
        f.write("%s\n" % item)