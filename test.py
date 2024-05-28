import glob
import numpy as np
from PIL import Image
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.gaussian as gaussian
import utils.masks as masks
import utils.image_loader as image_loader
import utils.vector_utils as vector_utils
import utils.quadratic_forms as quadratic_forms
import utils.sphere_deprojection as sphere_deprojection
import utils.camera as camera
import utils.intersections as intersections
import utils.photometry as photometry
import utils.geometric_fit as geometric_fit

folder_path = '/home/bcoupry/Data/DOME/ALL'
file_paths = glob.glob(folder_path + '/*.PNG')[::1]


max_image = image_loader.load_max_image(file_paths)
nu,nv, nc = max_image.shape

nspheres = 5
focal_mm = 35
matrix_size = 24
focal_pix = nu*focal_mm/matrix_size
K = camera.build_K_matrix(focal_pix,nu/2,nv/2)

pixels = np.reshape(max_image/255.0,(-1,3))

init_params = np.ones(2),np.broadcast_to(np.eye(3)*0.1,(2,3,3)),np.asarray([[0,0,0],[1,1,1]])
estimated_params = gaussian.gaussian_mixture_estimation(pixels,init_params, it=10)
classif = np.asarray(gaussian.maximum_likelihood(pixels,estimated_params),dtype=bool)

rectified_classif = masks.select_binary_mask(classif,lambda mask : np.mean(pixels[mask]))
sphere_masks = masks.get_greatest_components(np.reshape(rectified_classif,(nu, nv)),nspheres)
border_masks = np.vectorize(masks.get_mask_border,signature='(u,v)->(u,v)')(sphere_masks)

fit_on_mask = lambda border : quadratic_forms.fit_quadratic_form(vector_utils.to_homogeneous(np.argwhere(border)))
ellipse_quadratics = np.vectorize(fit_on_mask,signature='(u,v)->(t,t)')(border_masks)
calibrated_quadratics = np.swapaxes(K,-1,-2)@ellipse_quadratics@K

sphere_centers = sphere_deprojection.deproject_ellipse_to_sphere(calibrated_quadratics,1)
coordinates = np.stack(np.meshgrid(range(nu),range(nv),indexing='ij'),axis=-1)
rays = camera.get_camera_rays(coordinates,K)
sphere_points_map,sphere_geometric_masks = intersections.line_sphere_intersection(sphere_centers[:,np.newaxis,np.newaxis,:],1,rays[np.newaxis,:,:,:])
sphere_points = np.asarray([sphere_points_map[i,sphere_geometric_masks[i]] for i in range(nspheres)],dtype=object)
sphere_normals = np.vectorize(intersections.sphere_intersection_normal,signature='(v),()->()',otypes=[object])(sphere_centers,sphere_points)

grey_values = np.asarray(list(image_loader.load_map(file_paths,lambda image : [np.mean(image,axis=-1)[sphere_geometric_masks[i]]/255.0 for i in range(nspheres)])),dtype=object)
estimated_lights = np.vectorize(photometry.estimate_light,excluded=(2,),signature = '(),()->(k)',otypes=[float])(sphere_normals,grey_values,(0.1,0.9))

light_positions = intersections.lines_intersections(sphere_centers,estimated_lights)
dome_center,dome_radius = geometric_fit.sphere_parameters_from_points(light_positions)
principal_directions = vector_utils.norm_vector(dome_center-light_positions)[1]
flat_normals,flat_points,flat_grey = np.concatenate(sphere_normals),np.concatenate(sphere_points),np.vectorize(np.concatenate,signature='(u)->(v)')(grey_values)
mu,emitted_flux = photometry.estimate_anisotropy(light_positions[...,None,:], principal_directions[...,None,:], flat_points,flat_normals, flat_grey)

plane_normal,plane_alpha = geometric_fit.plane_parameters_from_points(sphere_centers)
plane_points = intersections.line_plane_intersection(plane_normal,plane_alpha,rays[::4,::4,:])
light_conditions = photometry.light_conditions(light_positions[:,np.newaxis,np.newaxis,:], principal_directions[:,np.newaxis,np.newaxis,:], plane_points, mu[:,np.newaxis,np.newaxis], emitted_flux[:,np.newaxis,np.newaxis])






fig = plt.figure()
axs = fig.add_subplot(projection='3d')
axs.scatter(light_positions[:,0],light_positions[:,1],light_positions[:,2],color='k')
axs.scatter(dome_center[0],dome_center[1],dome_center[2],color='r')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
for i in range(sphere_centers.shape[0]):
    x = sphere_centers[i,0] + 1*np.outer(np.cos(u), np.sin(v))
    y = sphere_centers[i,1] + 1*np.outer(np.sin(u), np.sin(v))
    z = sphere_centers[i,2] + 1*np.outer(np.ones(np.size(u)), np.cos(v))
    axs.plot_surface(x, y, z, color='b')
x = dome_center[0] + dome_radius*np.outer(np.cos(u), np.sin(v))
y = dome_center[1] + dome_radius*np.outer(np.sin(u), np.sin(v))
z = dome_center[2] + dome_radius*np.outer(np.ones(np.size(u)), np.cos(v))
axs.plot_surface(x, y, z, color='r',alpha=0.1)
rel_phi = emitted_flux/np.max(emitted_flux)
for i in range(principal_directions.shape[0]):
    axs.plot([light_positions[i,0],light_positions[i,0]+principal_directions[i,0]*rel_phi[i]*dome_radius*0.25],[light_positions[i,1],light_positions[i,1]+principal_directions[i,1]*rel_phi[i]*dome_radius*0.25],[light_positions[i,2],light_positions[i,2]+principal_directions[i,2]*rel_phi[i]*dome_radius*0.25],color='y')

p0 = np.linspace(np.min(light_positions[:,:2]),np.max(light_positions[:,:2]),2)
px,py = np.meshgrid(p0,p0,indexing='ij')
pz = -(plane_normal[0]*px+plane_normal[1]*py+plane_alpha)/plane_normal[2]
axs.plot_surface(px, py, pz, color='b',alpha=0.1)
axs.set_aspect('equal')

