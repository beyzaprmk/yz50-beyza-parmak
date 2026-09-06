import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0

        # Operandların sırasını koruyoruz.
        self._children = tuple(_children)

        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

   
    def __add__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        return Value(
            self.data + other.data,
            (self, other),
            "+"
        )

    
    def __mul__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        return Value(
            self.data * other.data,
            (self, other),
            "*"
        )

  
    def __pow__(self, other):
        return Value(
            self.data ** other,
            (self,),
            f"**{other}"
        )

   
    def __neg__(self):
        return Value(
            -self.data,
            (self,),
            "neg"
        )

   
    def __sub__(self, other):
        return self + (-other)

   
    def __truediv__(self, other):
        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        return Value(
            self.data / other.data,
            (self, other),
            "/"
        )

   
    def tanh(self):
        value = math.tanh(self.data)

        return Value(
            value,
            (self,),
            "tanh"
        )

   
    def exp(self):
        value = math.exp(self.data)

        return Value(
            value,
            (self,),
            "exp"
        )


    def backward(self):

       
        topo = []

        # Aynı node'u birden fazla kez ziyaret etmemek için
        visited = set()

        def build(node):

            if node not in visited:

                visited.add(node)

               
                for parent in node._children:
                    build(parent)

                topo.append(node)

        build(self)

        #outputun gradyani
        self.grad = 1.0

        
        for node in reversed(topo):

           
            if node._op == "+":

                a, b = node._children

                a.grad += node.grad
                b.grad += node.grad

           
            elif node._op == "*":  

                a, b = node._children

                a.grad += b.data * node.grad
                b.grad += a.data * node.grad

          
            elif node._op == "neg":

                (a,) = node._children

                a.grad += -node.grad

           
            elif node._op == "/":  

                a, b = node._children

                a.grad += (
                    (1 / b.data)
                    * node.grad
                )

                b.grad += (
                    (-a.data / (b.data ** 2))
                    * node.grad
                )

          
            elif node._op.startswith("**"): 

                (a,) = node._children

                exponent = float(
                    node._op[2:]
                )

                a.grad += (
                    exponent
                    * (
                        a.data
                        ** (exponent - 1)
                    )
                    * node.grad
                )

         
            elif node._op == "tanh":

                (a,) = node._children

                a.grad += (
                    (1 - node.data ** 2)
                    * node.grad
                )

          
            elif node._op == "exp":

                (a,) = node._children

                a.grad += (
                    node.data
                    * node.grad
                )
