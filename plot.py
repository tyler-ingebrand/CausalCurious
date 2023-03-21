import numpy as np
import matplotlib.pyplot as plt
import numpy.random

x = numpy.linspace(0, 10, 10)
y1 = numpy.random.random(size=(10,))
y2 = numpy.random.random(size=(10,))
y3 = numpy.random.random(size=(10,)) * 1000

# plot result of learning
l1, = plt.plot(x, y1, color='b')
l2, = plt.plot(x, y2, color='orange')
plt.xlabel("Env Interactions")
plt.ylabel("Euclidean Distance")
# plot result of learning
plt.twinx()
l3, = plt.plot(x, y3, color="g")
plt.xlabel("Env Interactions")
plt.ylabel("Correct Identification Percentage")
plt.title("Success of clustering during training")
plt.legend([l1, l2, l3] ,["Cluster Size", "Cluster Separation","Classification Success rate" ])
plt.savefig("test.png", dpi=300, bbox_inches='tight')