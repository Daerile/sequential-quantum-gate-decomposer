#!/bin/bash
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Bfgs
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Pso
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Genetic
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Multistart
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=iPso
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Price
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=gende
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=de
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=Tmlsl
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=gcrs
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=IntegerGenetic
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=ParallelGenetic
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=DoubleGenetic
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=NeuralMinimizer
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=ParallelDe
hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=parallelPso
