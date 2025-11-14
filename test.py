import myconet.network
from myconet.layers.populated import FullyPopulated


net = myconet.network.Network((
    FullyPopulated(100, 50, 1),
    FullyPopulated(50, 2, 2)
))

net.save("test.pyn")

net2 = myconet.network.Network.load("test.pyn")