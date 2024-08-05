#########################################################################3
#############################################################################
#########(integrated sector all as ((200-Deivices)))
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
import numpy as np
from sklearn.metrics import f1_score, recall_score


# Machine Learning Setup
def run_ml_models(X_train, X_test, y_train, y_test):
    # Create a decision tree classifier
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    dt_y_pred = dt_clf.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    dt_f1 = f1_score(y_test, dt_y_pred)
    dt_recall = recall_score(y_test, dt_y_pred)
    dt_prediction = " ".join(map(str, dt_y_pred))
    #print(f"Decision Tree Accuracy: {dt_accuracy}, F-measure: {dt_f1}, Recall: {dt_recall}, Prediction: {dt_prediction}")

    # Create a KNN classifier
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    knn_y_pred = knn_clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_y_pred)
    knn_f1 = f1_score(y_test, knn_y_pred)
    knn_recall = recall_score(y_test, knn_y_pred)
    knn_prediction = " ".join(map(str, knn_y_pred))
    #print(f"KNN Accuracy: {knn_accuracy}, F-measure: {knn_f1}, Recall: {knn_recall}, Prediction: {knn_prediction}")


    # Create an ANN classifier
    ann_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    ann_clf.fit(X_train, y_train)
    ann_y_pred = ann_clf.predict(X_test)
    ann_accuracy = accuracy_score(y_test, ann_y_pred)
    ann_f1 = f1_score(y_test, ann_y_pred)
    ann_recall = recall_score(y_test, ann_y_pred)
    ann_prediction = " ".join(map(str, ann_y_pred))
    #print(f"ANN Accuracy: {ann_accuracy}, F-measure: {ann_f1}, Recall: {ann_recall}, Prediction: {ann_prediction}")
    print("Overall Pperformance")
    print("F-measure:",dt_f1+knn_f1+ann_f1)
    print("Recall:",knn_recall+ann_recall+dt_recall)
    print("Prediction:",knn_prediction+ann_prediction+dt_prediction)
# Grasshopper Optimization Algorithm
def grasshopper_optimization_algorithm(num_nodes, max_iter):
    # Initialize grasshoppers' positions
    x = np.random.uniform(-10, 10, num_nodes)
    y = np.random.uniform(-10, 10, num_nodes)

    # Initialize the best solution
    best_solution = float('inf')
    best_position = None

    for _ in range(max_iter):
        # Calculate fitness values
        fitness = objective_function(x, y)

        # Find the best grasshopper
        if np.min(fitness) < best_solution:
            best_solution = np.min(fitness)
            best_position = (x[np.argmin(fitness)], y[np.argmin(fitness)])

        # Calculate distances between grasshoppers
        distances = np.sqrt((x[:, None] - x)**2 + (y[:, None] - y)**2)

        # Update grasshoppers' positions
        for i in range(num_nodes):
            x[i] = x[i] + np.sum((x - x[i]) / distances[i]**2)
            y[i] = y[i] + np.sum((y - y[i]) / distances[i]**2)

    return best_position, best_solution

def objective_function(x, y):
    return x**2 + y**2

# ECC key exchange simulation
def simulate_ecc_key_exchange(num_nodes):
    key_pairs = {}
    for i in range(num_nodes):
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()
        key_pairs[i] = (private_key, public_key)

    for i in range(num_nodes):
        sender_id = i
        receiver_id = np.random.randint(0, num_nodes)
        while receiver_id == sender_id:
            receiver_id = np.random.randint(0, num_nodes)
        sender_private_key, sender_public_key = key_pairs[sender_id]
        receiver_private_key, receiver_public_key = key_pairs[receiver_id]

        # Simulate key exchange
        shared_secret_sender = sender_private_key.exchange(ec.ECDH(), receiver_public_key)
        shared_secret_receiver = receiver_private_key.exchange(ec.ECDH(), sender_public_key)

        # Verify shared secret
        if shared_secret_sender == shared_secret_receiver:
            print(f"Key exchange successful between Node {sender_id} and Node {receiver_id}")
        else:
            print(f"Key exchange failed between Node {sender_id} and Node {receiver_id}")

# Implement RPL logic here
def run_rpl_logic(nodes):
    # This is a simplified example, you should replace this with your actual RPL implementation
    for node in nodes:
        # For each node, you would need to implement the RPL logic
        # such as managing parent-child relationships, sending and receiving control messages,
        # and updating routing tables based on RPL rules

        # For demonstration purposes, we'll print a message for each node
        print(f"Node {node} is running RPL protocol")

# Create Mininet topology
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

    # Run RPL logic
    run_rpl_logic(nodes)

    # Run Grasshopper Optimization Algorithm
    best_position, _ = grasshopper_optimization_algorithm(len(nodes), max_iter=100)

    # Run ECC key exchange simulation
    simulate_ecc_key_exchange(len(nodes))

    # Generate random data for ML models
    X = np.random.rand(200, 2)  # 200 samples, 2 features
    y = np.random.randint(0, 2, 200)  # Binary classification labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run ML models
    run_ml_models(X_train, X_test, y_train, y_test)

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_topology()

