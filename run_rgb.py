import train
import test
from haiti_dataset import HaitiDataset
from haiti_machine import HaitiMachine

train_ds = HaitiDataset(is_train=True, ctype="rgb")
test_ds = HaitiDataset(is_train=False, ctype="rgb")
model = HaitiMachine()
print("Starting training")
model = train.train(train_ds, model)

print("Starting testing")
percent = test.test(test_ds, model)
print(f'Correct:{percent:.2f}')