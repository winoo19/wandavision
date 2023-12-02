import numpy as np
import matplotlib.pyplot as plt

mypattern = np.load("patterns/heart0-prueba.npy")
pattern = np.load("patterns/inf0.npy")

plt.plot(mypattern[:, 0], mypattern[:, 1], "o", label="mypattern")
plt.plot(pattern[:, 0], pattern[:, 1], "o", label="pattern")
plt.gca().set_aspect("equal")
plt.legend()
plt.show()
