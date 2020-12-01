import numpy as np
import matplotlib.pyplot as plt
import random



class FunctionTree:
    def __init__(self, root_node, sub_trees):
        # unary function has one leaf, which is another FunctionTree
        # binary function has two leaves, '' '' ''
        # etc.
        # nullary is a FunctionTree with no leaves, so it is the end of its line
        # needs evaluate() method to get value of the tree by running the root function on its leaf arguments
        assert type(root_node) is FunctionNode, root_node
        assert type(sub_trees) is list, sub_trees
        for sub_tree in sub_trees:
            assert type(sub_tree) is FunctionTree, sub_tree
        self.function_node = root_node
        self.children = sub_trees

    def evaluate(self, outside_args):
        children_values = [sub_tree.evaluate(outside_args) for sub_tree in self.children]
        root_value = self.function_node.evaluate(children_values, outside_args)
        return root_value

    def get_str(self):
        root_symbol = self.function_node.symbol
        arg_strs = [sub_tree.get_str() for sub_tree in self.children]
        if len(arg_strs) == 0:
            return root_symbol
        args_str = ",".join(arg_strs)
        return "{}({})".format(root_symbol, args_str)

    @staticmethod
    def from_str(s, component_functions):
        print("from_str {}".format(s))
        if "(" in s:
            outer_func_symbol, *rest_lst = s.split("(")
            rest = "(".join(rest_lst)
            func_node = FunctionNode.from_symbol(outer_func_symbol, component_functions)
            assert rest[-1] == ")", rest
            arg_strs = rest[:-1].split(",")
            sub_trees = [FunctionTree.from_str(arg_s, component_functions) for arg_s in arg_strs]
            return FunctionTree(func_node, sub_trees)
        else:
            func_symbol = s
            func_node = FunctionNode.from_symbol(s, component_functions)
            sub_trees = []
            return FunctionTree(func_node, sub_trees)

    @staticmethod
    def random(component_functions, arity):
        outside_argument_getter_components = FunctionNode.create_outside_argument_getters(arity)
        all_component_functions = component_functions + outside_argument_getter_components

        node_choice = random.choice(all_component_functions)
        n_children = node_choice.leaf_arity
        children = [FunctionTree.random(all_component_functions, arity) for i in range(n_children)]
        return FunctionTree(node_choice, children)

    def plot(self, arity):
        if arity == 2:
            xs = list(range(256))
            # X, Y = np.meshgrid(xs, xs)
            # Z = f(X, Y)
            Z = []
            for x in xs:
                row = []
                for y in xs:
                    args = (x, y)
                    val = self.evaluate(args)
                    row.append(val)
                Z.append(row)
            Z = np.array(Z).T  # since X values were row numbers, need to transpose so X is instead column number, i.e. X axis is horizontal
            plt.imshow(Z, origin="lower")
            plt.colorbar()
            plt.title(self.get_str())
            plt.show()
        elif arity == 1:
            xs = list(range(256))
            ys = [self.evaluate((x,)) for x in xs]
            plt.plot(xs, ys)
            plt.title(self.get_str())
            plt.show()
        else:
            raise Exception("unsupported arity {} for plotting".format(arity))


class FunctionNode:
    def __init__(self, symbol, func, is_outside_arg_getter=False):
        self.symbol = symbol
        self.func = func

        # leaf arity is how many children this node needs in the function tree
        # for normal functions, like + or *, it's automatically the number of args
        # but for outside argument getters, it needs to be overwritten to zero, so the tree won't create children
        self.leaf_arity = 0 if is_outside_arg_getter else func.__code__.co_argcount
        self.is_outside_arg_getter = is_outside_arg_getter

    def evaluate(self, leaf_args, outside_args):
        # in future, if want to optimize, can memoize results for funcs with smaller numbers of args
        if self.is_outside_arg_getter:
            return self.func(*outside_args)
        else:
            assert len(leaf_args) == self.leaf_arity
            if not all(np.isfinite(x) for x in leaf_args):
                # if there are nans anywhere, return another one
                return np.nan
            try:
                return self.func(*leaf_args)
            except ZeroDivisionError:
                return np.nan
    
    @staticmethod
    def create_outside_argument_getters(arity):
        # create a list of FunctionNodes, one for each index in range(arity), which gets that arg from the input args
        # e.g. when arity is 2, this creates a function lambda x, y: x and another lambda x, y: y
        res = []
        for i in range(arity):
            symbol = "x{}".format(i)
            func = lambda *args, i=i: args[i]
            f = FunctionNode(symbol, func, is_outside_arg_getter=True)
            res.append(f)
        return res

    @staticmethod
    def from_symbol(s, component_functions):
        candidates = [fn for fn in component_functions if fn.symbol == s]
        if len(candidates) == 0:
            raise Exception("no function found for {}".format(repr(s)))
        elif len(candidates) == 1:
            return candidates[0]
        else:
            raise Exception("more than one function found for {}".format(repr(s)))


def evaluate_tuple(tup):
    assert type(tup) is tuple, tup
    func, arg_exprs = tup
    assert type(arg_exprs) is tuple, arg_exprs
    evaluated_args = (evaluate_tuple(arg_expr) for arg_expr in arg_exprs)
    return func(*evaluated_args)



if __name__ == "__main__":
    nullary_operations = [
        FunctionNode("-1", lambda: -1),
        FunctionNode("1", lambda: 1),
        FunctionNode("2", lambda: 2),
        FunctionNode("3", lambda: 3),
        FunctionNode("7", lambda: 7),
    ]
    
    unary_operations = [
        FunctionNode("~", lambda x: ~x),
        FunctionNode("-", lambda x: -x),
        FunctionNode("_/", lambda x: 0 if x <= 0 else x),
    ]
    
    binary_operations = [
        FunctionNode("+", lambda x, y: x + y),
        FunctionNode("-", lambda x, y: x - y),
        FunctionNode("*", lambda x, y: x * y),
        FunctionNode("//", lambda x, y: x // y),  # often makes things too small
        FunctionNode("&", lambda x, y: x & y),
        FunctionNode("|", lambda x, y: x | y),
        FunctionNode("^", lambda x, y: x ^ y),
        FunctionNode("%", lambda x, y: x % y),
        # FunctionNode("**", lambda x, y: int(int(x) ** int(y))),  # try to avoid typeerrors and making floats
        # FunctionNode("<<", lambda x, y: x << y if y >= 0 else x >> (-y)),  # creates errors
    ]
    
    component_functions = nullary_operations + unary_operations + binary_operations

    # f = create_function(components)
    # plot_func(f)

    f0 = lambda: 4
    f1 = lambda x: x + 2
    f2 = lambda x, y: x * y
    fx = lambda x, y: x
    fy = lambda x, y: y

    args0 = ()
    t0 = (f0, args0)

    four = evaluate_tuple(t0)
    # print(four)

    args1 = (t0,)  # a single arg, whose value is gotten by evaluating tuple 0
    t1 = (f1, args1)
    six = evaluate_tuple(t1)
    # print(six)

    args2 = (t0, t1)
    t2 = (f2, args2)
    x24 = evaluate_tuple(t2)
    # print(x24)

    arity = 2  # will make function of x and y
    # arity = 1  # will make function of x
    function_tree = FunctionTree.random(component_functions, arity)
    print("f(x0, x1) = {}".format(function_tree.get_str()))
    # print("f(x0) = {}".format(function_tree.get_str()))
    function_tree.plot(arity=arity)

    # s = "^(7,+(2,&(x0,-(x0,3))))"
    # function_tree = FunctionTree.from_str(s, component_functions)
    # function_tree.plot(arity=1)


