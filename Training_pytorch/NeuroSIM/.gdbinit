# 跳过标准库
skip -rfu ^std::
skip -rfu ^__gnu_cxx::

# 跳过常用数学函数
skip function ceil
skip function floor
skip function sqrt
skip function pow

# 跳过 vector 等容器的内部实现
skip file /usr/include/c++/*
skip file vector
skip file bits/stl_vector.h