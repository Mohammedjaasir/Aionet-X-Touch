from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel

def create_topology():
    net = Mininet(controller=RemoteController, switch=OVSKernelSwitch)
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    nodes = []
    for i in range(1, 201):
        node = net.addHost(f'h{i}')
        nodes.append(node)

    switch = net.addSwitch('s1')

    for i in range(1, 201):
        net.addLink(nodes[i-1], switch)

    net.start()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_topology()

