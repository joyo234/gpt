import gpt as g
import numpy as np

# from gpt.qcd.gauge.gluonic_objects import get_gluonic_objects

parameters = {
    "placeholder" : [0]
#    "q" : [0,1,0,0],
}

def main():
    g.message(f"{dir(g.qcd.gauge) = }")
    
    Rsq_list = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 35, 40, 60, 80, 100, 120, 140, 180, 220, 260, 320, 360, 400, 520, 640, 700, 800, 1000, 1400, 1800, 2200, 3072]
    
    print("Hello")
    U = g.convert(g.load("/work/CONF/jwinter/8x32x32x32_b7.544_shift0x0x0/standard_configuration_run/conf_s032t08_b0754400_a_U000000500"), g.double)
    g.message("Finished loading gauge configuration.")
    g.message(f"{type(U) = }")
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
    
    coor=g.coordinates(U[0]) # gpt.core.coordinates.local_coordinates
    # g.message(f"{type(coor) = }")
    # g.message(f"{coor = }") # basically all 4D coordinates
    
    mask = g.complex(grid) # gpt.core.lattice.lattice
    rsq = g.complex(grid) # gpt.core.lattice.lattice
    # g.message(f"{type(mask) = }")
    
    # basically all 4D coordinates
    # g.message(f"{g.coordinates(rsq) = }")
    
    rsqs_1D_from_3D_spatial_lattice = np.array([
        (Ns-i[0] if i[0]>int(Ns/2) else i[0])**2
        + (Ns-i[1] if i[1]>int(Ns/2) else i[1])**2
        + (Ns-i[2] if i[2]>int(Ns/2) else i[2])**2 
        for i in coor
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
    
    
    for Mask in Mask_all:
        g.message(f"{Mask.otype = }")
    
    objects = g.qcd.gauge.gluonic_objects.get_gluonic_objects(parameters)
    
    Umunu = objects.get_Umunu(U)
    g.message(f"{type(Umunu) = }")

    print("Finished script.")


if __name__=="__main__":
    main()
