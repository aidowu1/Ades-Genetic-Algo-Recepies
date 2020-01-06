
import logging
import sys
import itertools
import operator
import collections

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import random
import numpy as np

NEW_LINE = '\n'

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

random.seed(10)

def constructPrimitiveSet():
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 57), bool, "IN")

    # boolean operators
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)

    # floating point operators
    # Define a safe division function
    def safeDiv(left, right):
        try: return left / right
        except ZeroDivisionError: return 0

    pset.addPrimitive(operator.add, [float,float], float)
    pset.addPrimitive(operator.sub, [float,float], float)
    pset.addPrimitive(operator.mul, [float,float], float)
    pset.addPrimitive(safeDiv, [float,float], float)

    # logic operators
    # Define a new if-then-else function
    def if_then_else(input, output1, output2):
        if input: return output1
        else: return output2

    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    # terminals
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    pset.addTerminal(0, bool)
    pset.addTerminal(1, bool)
    return pset
    
def constructCreator(primitive_set):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=primitive_set)
    return creator
    
def constructPreCompileToolBox(primitive_set):
    creator = constructCreator(primitive_set)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitive_set)    
    return toolbox

def constructPostCompileToolBox(eval_func, data, toolbox, primitive_set):
    toolbox.register("evaluate", eval_func, data, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)
    return toolbox

def runPreCompileState():
    primitive_set = constructPrimitiveSet()
    toolbox = constructPreCompileToolBox(primitive_set)
    return primitive_set, toolbox

def countNumberOActualClasses(data):
    data_matrix = np.array(data)
    actual_class_count = collections.Counter(data_matrix[:,-1])
    return actual_class_count
    
def runPostCompileState(eval_func, data, toolbox, primitive_set):
    toolbox = constructPostCompileToolBox(eval_func, data, toolbox, primitive_set)
    actual_class_count = countNumberOActualClasses(data)
    logging.info(f"Class count:{NEW_LINE}{actual_class_count}")
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof)
    #logging.info("Best individual is %s, %s", gp.evaluate(hof[0]), hof[0].fitness)
    logging.info(f"best individual is {hof[0]}{NEW_LINE}Best Fitness is:{hof[0].fitness}")
    
    return pop, stats, hof

