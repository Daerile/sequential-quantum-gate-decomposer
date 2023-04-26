TEMPLATE=lib
SOURCES+=squander.cc \
    interval.cpp

INCLUDEPATH += \
    ../../sequential-quantum-gate-decomposer/decomposition/include \
    ../../sequential-quantum-gate-decomposer/gates/include \
    ../../sequential-quantum-gate-decomposer/common/include
    
LIBS     += "/home/morse/squander/sequential-quantum-gate-decomposer/qgd_python/libqgd.so"

HEADERS += \
    interval.h
