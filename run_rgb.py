import train
import test
from haiti_machine import HaitiMachine
from dataset_manager import get_ds

accuracy = []

for train_ds, test_ds in get_ds(ctype="rgb"):
    model = HaitiMachine()
    print("Starting training")
    model = train.train(train_ds, model)

    print("Starting testing")
    percent = test.test(test_ds, model)
    accuracy.append(percent)
    print(f'Correct:{percent:.2f}')

accuracy = sum(accuracy)/len(accuracy)
print(f'Final Result:{accuracy:.2f}')