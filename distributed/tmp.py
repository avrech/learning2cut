import ray
import threading
import time

@ray.remote
class Actor(object):
    def __init__(self):
        self.value = 0
        self.t = threading.Thread(target=self.update, args=())
        self.t.start()

    def update(self):
        while True:
            time.sleep(0.01)
            self.value += 1

    def get_value(self):
        return self.value

    def print_value(self):
        while True:
            time.sleep(1)
            print(self.value)

ray.init()

# Create the actor. This will start a long-running thread in the background
# that updates the value.
a = Actor.remote()

# Get the value a couple times.
a.print_value.remote()
print('finished')
# while True:
#     time.sleep(1)
#     print(ray.get(a.get_value.remote()))
#     print(ray.get(a.get_value.remote()))