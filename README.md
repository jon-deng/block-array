# BlockTensor

BlockTensor is a package for working with tensors interpreted as being partitioned into blocks (or subtensors). Common examples are block matrices and block vectors; for example, the matrix 
$$
A = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
    \end{bmatrix},
$$
can be interpreted in a block form as
$$
A = 
\begin{bmatrix}
A_{0,0} & A_{0, 1} \\
A_{1, 0} & A_{1, 1}
\end{bmatrix} = 
\begin{bmatrix}
    \begin{bmatrix}
    1 & 2 \\
    4 & 5
    \end{bmatrix} &
    \begin{bmatrix}
    3 \\ 
    6
    \end{bmatrix} \\

    \begin{bmatrix}
    7 & 8
    \end{bmatrix} &
    \begin{bmatrix}
    9
    \end{bmatrix}
\end{bmatrix},
$$
while a vector
$$
x = 
\begin{bmatrix}
    1 \\ 2 \\ 3
\end{bmatrix},
$$
can be interpreted as
$$
x = 
\begin{bmatrix}
    x_{0} \\ x_{1}
\end{bmatrix}
=
\begin{bmatrix}
    \begin{bmatrix}
    1 \\ 2
    \end{bmatrix}\\
    \begin{bmatrix}
    3
    \end{bmatrix}
\end{bmatrix},
$$
