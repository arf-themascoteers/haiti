import train
import test
from haiti_dataset import HaitiDataset
from haiti_mod_machine import HaitiModMachine

train_ds = HaitiDataset(is_train=True, ctype="mod_hsv")
test_ds = HaitiDataset(is_train=False, ctype="mod_hsv")
model = HaitiModMachine()
print("Starting training")
model = train.train(train_ds, model)

print("Starting testing")
percent = test.test(test_ds, model)
print(f'Correct:{percent:.2f}')