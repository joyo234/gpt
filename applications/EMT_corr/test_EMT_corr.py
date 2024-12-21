import gpt as g
import numpy as np


class AntisymmetricMatrix(np.ndarray):

    def __setitem__(self, indexes, value):
        (i, j) = indexes
        super().__setitem__((i, j), g.eval(value))
        super().__setitem__((j, i), g.eval(-1.0 * value))


def empty_antisymmetric(shape):
    return np.empty(shape, g.core.lattice).view(AntisymmetricMatrix)


class SymmetricMatrix(np.ndarray):

    def __setitem__(self, indexes, value):
        (i, j) = indexes
        super().__setitem__((i, j), g.eval(value))
        super().__setitem__((j, i), g.eval(value))



def empty_symmetric(shape):
    return np.empty(shape, g.core.lattice).view(SymmetricMatrix)


class DoubleSymmetricMatrix(np.ndarray):

    def __setitem__(self, indexes, value):
        (i, j, l, m) = indexes
        super().__setitem__((i, j, l, m), g.eval(value))
        super().__setitem__((j, i, l, m), g.eval(value))
        super().__setitem__((i, j, m, l), g.eval(value))
        super().__setitem__((j, i, m, l), g.eval(value))



def empty_doublesymmetric(shape):
    return np.empty(shape, g.core.lattice).view(DoubleSymmetricMatrix)


# from gpt.qcd.gluon.gluonic_objects from branch haitao
# F_munu is anti-symmetric, i.e. F01 = -F10. The diagonal parts are simply zeros
def get_Fmunu(U):
    
    # empty_field = g.complex(U[0].grid)
    # empty_field[:] = 0.0
    
    # Fmunu = antisymmetric(np.full((4,4), empty_field, dtype=g.core.lattice))
    
    Ex = g.qcd.gauge.field_strength(U, 0, 3) # FT03 # gpt.core.lattice.lattice
    Ey = g.qcd.gauge.field_strength(U, 1, 3) # FT13
    Ez = g.qcd.gauge.field_strength(U, 2, 3) # FT23
    Bx = g.qcd.gauge.field_strength(U, 1, 2) # FT12
    By = g.qcd.gauge.field_strength(U, 2, 0) # FT20
    Bz = g.qcd.gauge.field_strength(U, 0, 1) # FT01
    
    # g.message(f"{type(Ex) = }")
    
    # Fmunu[0,3] = Ex
    # g.message(f"{type(Fmunu[0,0]) = }")
    # g.message(f"{type(Fmunu[0,3]) = }")
    # g.message(f"{type(Fmunu[3,0]) = }")
    # g.message(f"{Fmunu[0,1][0,0,0,:] = }")
    # g.message(f"{Fmunu[1,0][0,0,0,:] = }")
    
    # g.message("Fmunu done")
    return Ex, Ey, Ez, Bx, By, Bz


# =1/4*F^a_{rho,sigma}*F^a_{rho,sigma}, Tr[t^i*t*j]=1/2*delta_{i,j}
def get_gluon_anomaly(U, Fmunu = None):
    grid = U[0].grid
    
    if Fmunu is None:
        Ex, Ey, Ez, Bx, By, Bz = get_Fmunu(U)
    else:
        Ex, Ey, Ez, Bx, By, Bz = Fmunu
    
    gluon_anomaly = g.complex(grid)
    gluon_anomaly[:] = 0
    gluon_anomaly = g.eval(g.color_trace(Ex*Ex+Ey*Ey+Ez*Ez+Bx*Bx+By*By+Bz*Bz))
    #g.message("Trace Anomaly done.")
    return gluon_anomaly


# The traceless part of gluonic T_munu, which we name U_munu, is symmetric.
# = F^a_{mu,rho}*F^a_{nu,rho}-delta_{mu,nu}*gluon_anomaly
def get_Umunu(U, Fmunu = None, gluon_anomaly = None):
    if Fmunu is None:
        Ex, Ey, Ez, Bx, By, Bz = get_Fmunu(U)
    else:
        Ex, Ey, Ez, Bx, By, Bz = Fmunu
    
    if gluon_anomaly is None:
        gluon_anomaly = get_gluon_anomaly(U)
    
    Umunu = empty_symmetric((4,4))
    
    Umunu[0,0] = g.eval(2.0*g.color_trace(g.eval(Bz*Bz+By*By+Ex*Ex)) - gluon_anomaly)
    # Umunu.append(U00)
    Umunu[0,1] = g.eval(2.0*g.color_trace(g.eval(-By*Bx+Ex*Ey)))
    # Umunu.append(U01)
    Umunu[0,2] = g.eval(2.0*g.color_trace(g.eval(-Bz*Bx+Ex*Ez)))
    # Umunu.append(U02)
    Umunu[0,3] = g.eval(2.0*g.color_trace(g.eval(-Bz*Ey+By*Ez)))
    # Umunu.append(U03)
    Umunu[1,1] = g.eval(2.0*g.color_trace(g.eval(Bz*Bz+Bx*Bx+Ey*Ey)) - gluon_anomaly)
    # Umunu.append(U11)
    Umunu[1,2] = g.eval(2.0*g.color_trace(g.eval(-Bz*By+Ey*Ez)))
    # Umunu.append(U12)
    Umunu[1,3] = g.eval(2.0*g.color_trace(g.eval(Bz*Ex-Bx*Ez)))
    # Umunu.append(U13)
    Umunu[2,2] = g.eval(2.0*g.color_trace(g.eval(By*By+Bx*Bx+Ez*Ez)) - gluon_anomaly)
    # Umunu.append(U22)
    Umunu[2,3] = g.eval(2.0*g.color_trace(g.eval(-By*Ex+Bx*Ey)))
    # Umunu.append(U23)
    Umunu[3,3] = g.eval(2.0*g.color_trace(g.eval(Ex*Ex+Ey*Ey+Ez*Ez)) - gluon_anomaly)
    # Umunu.append(U33)
    
    g.message(f"{type(Umunu[0,0]) = }")
    g.message(f"{type(Umunu[0,1]) = }")
    g.message(f"{Umunu[0,1][0,0,15,:] = }")
    g.message(f"{type(Umunu[1,0]) = }")
    g.message(f"{Umunu[1,0][0,0,15,:] = }")
    
    return Umunu


def get_U_FT(U):
    # L = U[0,0].grid.gdimensions
    
    U_FT = empty_symmetric((4,4))
    U_FT_adj = empty_symmetric((4,4))
    for i in range(0,4):
        for j in range(i,4):
            g.message(f"{i, j = }")
            U_FT[i,j] = g.eval(g.fft([0,1,2]) * U[i,j])
            U_FT_adj[i,j] = g.eval(g.adj(g.fft([0,1,2])) * U[i,j])
    
    g.message(f"{type(U_FT[0,0]) = }")
    g.message(f"{type(U_FT[0,1]) = }")
    g.message(f"{U_FT[0,1][0,0,15,:] = }")
    g.message(f"{type(U_FT[1,0]) = }")
    g.message(f"{U_FT[1,0][0,0,15,:] = }")
    
    return U_FT, U_FT_adj


def get_G(U, Umunu_FT=None, Umunu_FT_adj=None):
    
    if Umunu_FT is None or Umunu_FT_adj is None:
        Umunu_FT, Umunu_FT_adj = get_U_FT(U)
    
    G = empty_doublesymmetric((4,4,4,4))
    
    for i in range(4):
        for j in range(i, 4):
            for l in range(4):
                for m in range(l, 4):
                    G[i,j,l,m] = g.eval(g.adj(g.fft([0,1,2])) * Umunu_FT[i,j] * Umunu_FT_adj[l,m])
    
    g.message(f"{G.shape = }")
    g.message(f"{type(G[0,0,0,0]) = }")
    g.message(f"{G[0,0,0,0][0,0,15,:] = }")
    
    return G


def get_reduced_G(U, Umunu_FT=None, Umunu_FT_adj=None, G=None):
    
    if G is None:
        G = get_G(U, Umunu_FT=U_FT, Umunu_FT_adj=U_FT_adj)
    
    G_reduced = np.empty((4,4), g.core.lattice)
    
    for i, j in np.ndindex(G_reduced.shape):
        G_reduced[i,j] = G[i,3,j,3]
    
    g.message(f"{G_reduced.shape = }")
    g.message(f"{type(G_reduced[0,0]) = }")
    g.message(f"{G_reduced[0,0][0,0,15,:] = }")
    
    return G_reduced


def r2(r):
    return (r[0]**2 + r[1]**2 + r[2]**2)


def delta(i,j):
    if i == j:
        return 1
    else:
        return 0


def get_projector_L(i, j, r):
    if r[0] == 0 and r[1] == 0 and r[2] == 0:
        return 1
    
    return r[i] * r[j] / r2(r)


def get_projector_T(i, j, r):
    if r[0] == 0 and r[1] == 0 and r[2] == 0:
        return 1
    
    return (r2(r) * delta(i,j) - r[i] * r[j]) / (2 * r2(r))


def get_projector_LL(i, j, l, m, r):
    if r[0] == 0 and r[1] == 0 and r[2] == 0:
        return 1

    return 1/9 * (
        6 * (
            r[i] * r[j] * r[l] * r[m]
        ) / (r2(r)**2)
        - 3 * (
            r[l] * r[m] * delta(i, j)
            + r[i] * r[j] * delta(l, m)
        ) / r2(r)
        + delta(i, j) * delta(l, m)
    )


def get_projector_LT(i, j, l, m, r):
    if r[0] == 0 and r[1] == 0 and r[2] == 0:
        return 1

    return 0.5 * (
        (
            r[i] * r[l] * delta(j, m)
            + r[i] * r[m] * delta(j, l)
            + r[j] * r[l] * delta(i, m)
            + r[j] * r[m] * delta(i, l)
        ) / r2(r)
        - 4 * (
            r[i] * r[j] * r[l] * r[m]
        ) / (r2(r)**2)
    )


def get_projector_TT(i, j, l, m, r):
    if r[0] == 0 and r[1] == 0 and r[2] == 0:
        return 1
    
    return 0.5 * (
        delta(i, l) * delta(j, m)
        + delta(i, m) * delta(j, m)
        - delta(i, j) * delta(l, m)
        + (
            r[i] * r[j] * r[l] * r[m]
        ) / (r2(r)**2)
        - (
            r[i] * r[l] * delta(j, m)
            + r[i] * r[m] * delta(j, l)
            + r[j] * r[l] * delta(i, m)
            + r[j] * r[m] * delta(i, l)
        ) / r2(r)
        + (
            r[l] * r[m] * delta(i, j)
            + r[i] * r[j] * delta(l, m)
        ) / r2(r)
    )


def get_reduced_projector_field(grid, projector):
    
    shape = (3, 3)
    
    coordinates = g.coordinates(grid)
    
    projector_reduced_field = empty_symmetric(shape)
    
    for i, j in np.ndindex(shape):
        projector_reduced_field[i, j] = g.real(grid)
        
        projector_values = np.array([projector(i, j, r) for r in coordinates])
        
        g.coordinate_mask(projector_reduced_field[i, j], projector_values)
    
    return projector_reduced_field


def get_projector_field(grid, projector):
    
    shape = (3, 3, 3, 3)
    
    coordinates = g.coordinates(grid)
    
    projector_field = empty_doublesymmetric(shape)
    
    for i, j, l, m in np.ndindex(shape):
        projector_field[i, j, l, m] = g.real(grid)
        
        projector_values = np.array([projector(i, j, l, m, r) for r in coordinates])
        
        g.coordinate_mask(projector_field[i, j, l, m], projector_values)
    
    return projector_field


def get_projector_L_field(grid):
    return get_reduced_projector_field(grid, get_projector_L)


def get_projector_T_field(grid):
    return get_reduced_projector_field(grid, get_projector_T)


def get_projector_LL_field(grid):
    return get_projector_field(grid, get_projector_LL)


def get_projector_LT_field(grid):
    return get_projector_field(grid, get_projector_LT)


def get_projector_TT_field(grid):
    return get_projector_field(grid, get_projector_TT)


def get_G_component_reduced(G_reduced, projector_component_function):
    grid = G_reduced[0, 0].grid
    
    G_component = g.complex(grid)
    G_component[:] = 0
    
    projector_field = projector_component_function(grid)
    
    for i in range(3):
        for j in range(i, 3):
            G_component = g.eval(G_reduced[i, j] * projector_field[i, j] + G_component)
    
    return G_component


def get_G_component(G, projector_component_function):
    grid = G[0, 0, 0, 0].grid
    
    G_component = g.complex(grid)
    G_component[:] = 0
    
    projector_field = projector_component_function(grid)
    
    for i in range(3):
        for j in range(i, 3):
            for l in range(3):
                for m in range(l, 3):
                    G_component = g.eval(G[i, j, l, m] * projector_field[i, j, l, m] + G_component)
    
    return G_component


def get_G_L(G_reduced):
    return get_G_component_reduced(G_reduced, get_projector_L_field)


def get_G_T(G_reduced):
    return get_G_component_reduced(G_reduced, get_projector_T_field)


def get_G_LL(G):
    return get_G_component(G, get_projector_LL_field)


def get_G_LT(G):
    return get_G_component(G, get_projector_LT_field)


def get_G_TT(G):
    return get_G_component(G, get_projector_TT_field)


def time_average(lattice):
    
    if isinstance(lattice, np.ndarray):
        
        # g.message(f"{lattice.shape = }")
        
        spatial_lattice = np.empty(lattice.shape, lattice.dtype)
        
        for indices in np.ndindex(lattice.shape):
            spatial_lattice[indices] = time_average(lattice[indices])
        
        return spatial_lattice
    
    elif isinstance(lattice, g.core.lattice):
        # g.message(f"{lattice.grid.fdimensions = }")
        
        spatial_lattice = g.complex(g.grid(lattice.grid.fdimensions[0:3], g.single))
        # g.message(f"{spatial_lattice.grid.fdimensions = }")
        
        # terribly slow
        for indices in np.ndindex(tuple(spatial_lattice.grid.fdimensions)):
            spatial_lattice[indices] = lattice[*indices, 0]
        
        return spatial_lattice
    else:
        print("Unknown type!")


def main():
    g.message(f"{dir(g.qcd.gauge) = }")
    
    Rsq_list = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 35, 40, 60, 80, 100, 120, 140, 180, 220, 260, 320, 360, 400, 520, 640, 700, 800, 1000, 1400, 1800, 2200, 3072]
    
    print("Hello")
    U = g.convert(g.load("/work/CONF/jwinter/8x32x32x32_b7.544_shift0x0x0/standard_configuration_run/conf_s032t08_b0754400_a_U000000500"), g.double)
    g.message("Finished loading gauge configuration.")
    g.message(f"{type(U) = }")
    g.message(f"{len(U)}")
    g.message(U)
    
    g.message(f"{type(U[0]) = }")
    # g.message(U[0])
    
    grid = U[0].grid # gpt.core.grid.grid
    # g.message(f"{type(grid) = }")
    g.message(grid)
    
    L = U[0].grid.gdimensions
    g.message(f"{L = }")
    
    Ns = U[0].grid.gdimensions[0] # 32
    # g.message(f"{Ns = }")
    
    coordinates = g.coordinates(U[0]) # gpt.core.coordinates.local_coordinates
    # g.message(f"{type(coor) = }")
    # g.message(f"{coor = }") # basically all 4D coordinates
    # g.message(coor)
    
    # projector_1 = g.real(grid)
    
    # projector_1_values = np.array([
    #     get_projector_L(0, 0, r[0], r[1], r[2]) for r in coor
    # ])
    
    # g.coordinate_mask(projector_1, projector_1_values)
    
    # g.message(f"{type(projector_1) = }")
    # t = 0
    # for x,y,z in np.ndindex((4,4,4)):
    #     g.message(f"{x,y,z,t = } with {projector_1[x,y,z,t] = }")
    
    mask = g.complex(grid) # gpt.core.lattice.lattice
    rsq = g.complex(grid) # gpt.core.lattice.lattice
    # g.message(f"{type(mask) = }")
    
    # basically all 4D coordinates
    # g.message(f"{g.coordinates(rsq) = }")
    
    rsqs_1D_from_3D_spatial_lattice = np.array([
        (Ns-i[0] if i[0]>int(Ns/2) else i[0])**2
        + (Ns-i[1] if i[1]>int(Ns/2) else i[1])**2
        + (Ns-i[2] if i[2]>int(Ns/2) else i[2])**2 
        for i in coordinates
    ]) # all r^2 values for the 3D spatial lattice part with periodic boundary conditions
    # g.message(f"{haitao_array.shape = }") # 262144 = 8*32^3
    # g.message(f"{haitao_array[:64] = }")
    
    g.coordinate_mask(
        rsq,
        rsqs_1D_from_3D_spatial_lattice
    ) # loading r^2 values to 3D/4D lattice?
    
    Mask_all = list() # list[<gpt.core.object_type.complex_additive_group.ot_complex_additive_group]
    for ii in range(len(Rsq_list)):
        # g.message("Rsq: ", Rsq_list[ii])
        mask[:] = 0
        g.coordinate_mask(mask, rsq[:] <= float(Rsq_list[ii]))
        Mask_all.append(g.copy(mask))
    
    
    # for Mask in Mask_all:
    #     g.message(f"{Mask.otype = }")
    
    Fmunu = get_Fmunu(U)
    
    gluon_anomaly = get_gluon_anomaly(U, Fmunu)
    Umunu = get_Umunu(U, Fmunu=Fmunu, gluon_anomaly=gluon_anomaly)
    
    # Umunu_spatial = time_average(Umunu)
    g.message(f"{Umunu[0,0][0,0,0,:] = }")
    # g.message(f"{Umunu_spatial[0,0][0,0,:] = }")
    
    Umunu_FT, Umunu_FT_adj = get_U_FT(Umunu)
    
    G = get_G(U, Umunu_FT=Umunu_FT, Umunu_FT_adj=Umunu_FT_adj)
    G_reduced = get_reduced_G(U, G=G)
    
    G_L = get_G_L(G_reduced)
    g.message(f"{G_L[0,0,0,:] = }")
    G_T = get_G_T(G_reduced)
    g.message(f"{G_T[0,0,0,:] = }")
    
    G_LL = get_G_LL(G)
    g.message(f"{G_LL[0,0,0,:] = }")
    G_LT = get_G_LT(G)
    g.message(f"{G_LT[0,0,0,:] = }")
    G_TT = get_G_TT(G)
    g.message(f"{G_TT[0,0,0,:] = }")
    
    # r_grid = g.vreal(g.grid(L[0:3], g.single), 3)
    # r_grid[:] = 0
    # for i,j,k in np.ndindex(r_grid.grid.fdimensions):
    #     r_grid[i,j,k] = 
    
    # sliced = g.slice(Umunu[0,0], 3) # list[complex]
    # g.message(f"{sliced[0] = }")

    print("Finished script.")


if __name__=="__main__":
    main()
