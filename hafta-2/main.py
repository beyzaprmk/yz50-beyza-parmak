from engine import Value
from nn import MLP
from numerical import numerical_derivative
from graphviz import Digraph


def trace(root):
    

    nodes = set()
    edges = set()

    def build(node):

        if node not in nodes:

            nodes.add(node)

            for child in node._children:

                edges.add((child, node))

                build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    
    dot = Digraph(format="png")

    # Graph soldan sağa ilerlesin
    dot.attr(rankdir="LR")

    nodes, edges = trace(root)

   
    for node in nodes:

        node_id = str(id(node))

        label = (
            f"data {node.data:.4f} | "
            f"grad {node.grad:.4f}"
        )

        dot.node(
            node_id,
            label=label,
            shape="box"
        )

        # Eğer node bir işlem sonucu oluşmuşsa
        # operation node oluştur.
        if node._op:

            op_id = node_id + "_op"

            dot.node(
                op_id,
                label=node._op,
                shape="circle"
            )

            # Operation -> Value
            dot.edge(
                op_id,
                node_id
            )

   
    for child, parent in edges:

        child_id = str(id(child))
        parent_id = str(id(parent))

        op_id = parent_id + "_op"

        dot.edge(
            child_id,
            op_id
        )

    return dot



def run_value_experiments():

   
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")

   
    c = a.__add__(b)
    d = c * b
    f = d ** 2

    print("Forward:")

    print("a =", a.data)
    print("b =", b.data)
    print("c =", c.data)
    print("d =", d.data)
    print("f =", f.data)

   
    f.backward()

    print("\nGradients:")

    print("df/da =", a.grad)
    print("df/db =", b.grad)

   
    def f_a(x):

        return ((x + b.data) * b.data) ** 2

    numerical_a = numerical_derivative(
        f_a,
        a.data
    )

    print(
        "Numerical derivative df/da =",
        numerical_a
    )

   
    dot = draw_dot(f)

    dot.render(
        "f_graph",
        view=True
    )



def train_mlp():

   
    X = [
        [2.0, 3.0],
        [3.0, -1.0],
        [-1.0, -2.0],
        [4.0, 1.0],
    ]

    Y = [
        [1.0],
        [-1.0],
        [-1.0],
        [1.0],
    ]


    model = MLP(
        2,
        [4, 4, 1]
    )

    learning_rate = 0.01
    epochs = 100

   
    for epoch in range(epochs):

        predictions = []

       
        for x in X:

            inputs = [
                Value(xi)
                for xi in x
            ]

            prediction = model(inputs)

            predictions.append(prediction)

       
        loss = Value(0.0)

        for prediction, target in zip(predictions, Y):

            error = prediction - target[0]

            loss = loss + error ** 2

        # Mean Squared Error
        loss = loss / len(X)

        
        for parameter in model.parameters():

            parameter.grad = 0.0

       
        loss.backward()

       
        for parameter in model.parameters():

            parameter.data -= (
                learning_rate
                * parameter.grad
            )


        if epoch % 10 == 0:

            print(
                f"epoch {epoch:3d} | "
                f"loss = {loss.data:.6f}"
            )



def main():

    run_value_experiments()

    train_mlp()


if __name__ == "__main__":
    main()