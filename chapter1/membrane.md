
# Deflection of a membrane
Authors: Hans Petter Langtangen and Anders Logg.

Modified to dolfin-X by Jørgen S. Dokken

In the first FEniCS program, we solved a simple problem which we could easily use to verify the implementation. In this section, we will turn our attentition to a physically more relevant problem with solutions of a somewhat more exciting shape.

We would like to compute the deflection $D(x,y)$ of a two-dimensional, circular membrane of radius $R$, subject to a load $p$ over the membrane. The appropriate PDE model is 
\begin{align}
     -T \nabla^2D&=p \quad\text{in }\quad \Omega=\{(x,y)\vert x^2+y^2\leq R \}.
\end{align}
Here, $T$ is the tension in the membrane (constant), and  $p$ is the external pressure load. The boundary of the membrane has no deflection. This implies that $D=0$ is the boundary condition. We model a localized load as a Gaussian function:
\begin{align}
     p(x,y)&=\frac{A}{2\pi\sigma}e^{-\frac{1}{2}\left(\frac{x-x_0}{\sigma}\right)^2-\frac{1}{2}\left(\frac{y-y_0}{\sigma}\right)^2}
\end{align}
The parameter $A$ is the amplitude of the pressure, $(x_0, y_0)$ the localization of the maximum point of the load, and $\sigma$ the "width" of $p$. We will take the center $(x_0,y_0)$ to be $(0,R_0)$ for some $0<R_0<R$.
Then we have 
\begin{align}
     p(x,y)&=\frac{A}{2\pi\sigma}e^{-\frac{1}{2}\left(\left(\frac{x}{\sigma}\right)^2
     +\left(\frac{y-R_0}{\sigma}\right)^2\right)}
\end{align}



## Scaling the  equation

There are many physical parameters in this problem, and we can benefit from grouping them by means of scaling. Let us introduce dimensionless coordinates 
$\bar{x}=\frac{x}{R}$, $\bar{y}=\frac{y}{R}$, and a dimensionless deflection $w=\frac{D}{D_e}$, where $D_e$ is a characteristic size of the deflection. Introducing $\bar{R}_0=\frac{R_0}{R}$, we obtain
\begin{align}
    -\frac{\partial^2 w}{\partial \bar{x}^2} -\frac{\partial^2 w}{\partial \bar{y}^2}
    &=\frac{R^2A}{2\pi\sigma TD_e}e^{-\frac{R^2}{2\sigma^2}\left(\bar{x}^2+(\bar{y}-\bar{R}_0)^2\right)}\\
    &=\alpha e^{-\beta^2(\bar{x}^2+(\bar{y}-\bar{R}_0)^2}
\end{align}
for $\bar{x}^2+\bar{y}^2<1$ where $\alpha = \frac{R^2A}{2\pi\sigma TD_e}$ and $\beta=\frac{R}{\sqrt{2}\sigma}$.

With an appropriate scaling, $w$ and its derivatives are of size unity, so the left-hand side of the scaled PDE is about unity in size, while the right hand side has $\alpha$ as its characteristic size. This suggests choosing alpha to be unity, or around unity. In this particular case, we choose $\alpha=4$. (One can also find  the analytical solution in scaled coordinates and show that the maximum deflection $D(0,0)$ is $D_e$ if we choose $\alpha=4$ to determine $D_e$.)
With $D_e=\frac{AR^2}{8\pi\sigma T}$ and dropping the bars we obtain the scaled problem
\begin{align}
    -\nabla^2 w = 4e^{-\beta^2(x^2+(y-R_0)^2)}
\end{align}
to be solved over the unit disc with $w=0$ on the boundary. Now there are only two parameters to varyl the dimensionelss extent of the pressure, $\beta$, and the localization of the pressure peak, $R_0\in[0,1]$. As $\beta\to 0$, the solution will approach the special case $1-x^2-y^2$. Given a computed scaed solution $w$, the physical deflection can be computed by
\begin{align}
    D=\frac{AR^2}{8\pi\sigma T}w
\end{align}
