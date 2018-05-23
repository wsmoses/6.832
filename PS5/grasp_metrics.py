from numpy import sin, cos
import numpy as np

# These are only for plotting
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


def achieves_force_closure(points, normals, mu):
    """
    Determines whether the given forces achieve force closure.

    Note that each of points and normals are lists of variable 
    length, but should be the same length.

    Here's an example valid set of input args:
        points  = [np.asarray([0.1, 0.2]), np.asarray([0.5,-0.1])]
        normals = [np.asarray([-1.0,0.0]), np.asarray([0.0, 1.0])]
        mu = 0.2
        achieves_force_closure(points, normals, mu)

    NOTE: your normals should be normalized (be unit vectors)

    :param points: a list of arrays where each array is the
        position of the force points relative to the center of mass.
    :type points: a list of 1-d numpy arrays of shape (2,)

    :param normals: a list of arrays where each array is the normal
        of the local surface of the object, pointing in towards
        the object
    :type normals: a list of 1-d numpy arrays of shape (2,)

    :param mu: coulombic static friction coefficient
    :type mu: float, greater than or equal to 0

    :return: whether or not the given parameters achieves force closure
    :rtype: bool (True or False)
    """
    assert len(points) == len(normals)
    assert mu >= 0.0
    
    G = get_G(points, normals)
    if np.linalg.matrix_rank(G) != 3:
        return False
   
    from pydrake.all import MathematicalProgram
    from pydrake.solvers import mathematicalprogram as mpa
    
    mp = MathematicalProgram()
    
    f = mp.NewContinuousVariables(2*len(points), "f")
    gamma = mp.NewContinuousVariables(1, "gamma")

    mp.AddLinearConstraint(gamma[0] <= 0)
    mp.AddLinearConstraint(gamma[0] >= -1)

    for i in range(len(points)):
        mp.AddLinearConstraint(f[2*i] <= mu * f[2*i+1] + gamma[0])
        mp.AddLinearConstraint(-f[2*i] <= mu * f[2*i+1] + gamma[0])
        mp.AddLinearConstraint(f[2*i] <= 1000)
        mp.AddLinearConstraint(f[2*i] >= -1000)
        mp.AddLinearConstraint(f[2*i+1] + gamma[0] >= 0)
        mp.AddLinearConstraint(f[2*i+1] <= 1000)
                
    gf = G.dot(f)
    #print(gf.shape)
    assert(gf.shape == (3,))
    for i in range(3):
        mp.AddLinearConstraint(gf[i] == 0)
    
    mp.AddLinearCost(gamma[0])
    #mp.AddQuadraticCost(np.sum(f**2))

    #print(mp)
    x = mp.Solve()
    #print(mp.Solve())
    if x == mpa.SolutionResult.kSolutionFound:
        #print(mp.GetSolution(f))
        g = mp.GetSolution(gamma)
        #print(g)
        #print(g[0] < 0)
        return g[0] < 0
    return False

def compute_convex_hull_volume(points):
    """
    Compute the volume of the convex hull formed by points.

    :param points: See achieves_force_closure() for documentation.
    """
    from scipy.spatial import ConvexHull
    if len(points) <= 2: return 0
    hull = ConvexHull(points)
    return hull.volume

def get_G(points, normals):
    """
    Builds G matrix.
    """
    assert len(points) == len(normals)

    tangents = [get_perpendicular2d(normal) for normal in normals]

    G = np.zeros((3,len(points)*2))
    for i in range(len(points)):
        idx = i*2
        G[0,idx  ] = tangents[i][0]
        G[0,idx+1] = normals [i][0]
        G[1,idx  ] = tangents[i][1]
        G[1,idx+1] = normals [i][1]
        G[2,idx  ] = get_cross2d(points[i],tangents[i])
        G[2,idx+1] = get_cross2d(points[i],normals[i])

    return G

def plot_points_with_normals(points, normals, mu, magnitudes=None):
    '''
    Makes a plot to help you visualize points and normals.

    Args points, normals, and mu are the same as inputs to 
    achieves_force_closure() -- see that function for documentation.
    '''
    assert_unit_normals(normals)
    
    # CoM
    fig, axes = plt.subplots(nrows=1,ncols=1)
    axes.text(0, 0.2, 'CoM', style='italic')
    circ = Circle((0,0), radius=0.1, facecolor='k', edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
    axes.add_patch(circ)
    for point in points:
        circ = Circle(point, radius=0.01, facecolor='b', edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
        axes.add_patch(circ)

    local_surface = []
    good_scale = (np.max(points) - np.min(points))/4.0
    for _ in normals:
        local_surface.append(good_scale)
    if magnitudes is None:
        magnitudes = local_surface
            
    for point, normal, m, s in zip(points, normals, magnitudes, local_surface):
        
        arrow_tip = point + normal*m

        # force
        axes.arrow(point[0], point[1], normal[0]*m, normal[1]*m, 
            head_width=m/5.0, head_length=m/5.0, fc='k', ec='k')
        circ = Circle(arrow_tip, radius=0.0001, facecolor='k', edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
        axes.add_patch(circ)

        # tangent surface
        tangent = get_perpendicular2d(normal)
        axes.arrow(point[0], point[1], tangent[0]*s/2, tangent[1]*s/2, 
            head_width=0.0, head_length=0.0, fc='r', ec='r')
        axes.arrow(point[0], point[1], -tangent[0]*s/2, -tangent[1]*s/2, 
            head_width=0.0, head_length=0.0, fc='r', ec='r')

        # friction cone
        c1 = point + (normal*m + mu*tangent*m)*0.5
        c2 = point + (normal*m - mu*tangent*m)*0.5
        xy = np.asarray([point,c1,c2,point])
        poly = Polygon(xy, closed=True, alpha=0.1)
        axes.add_patch(poly)


    axes.axis('equal')
    plt.show()

def assert_unit_normals(normals):
    """
    Simple function to assert normals are unit vectors.
    """
    for normal in normals:
        assert np.allclose(np.linalg.norm(normal), 1.0), "Your normals should be normalized."

def get_perpendicular2d(vector):
    """
    Simple implementation to get a vector in 2D perpendicular to any vector.

    :type vector: 1-d numpy array of shape (2,)
    """
    if vector[1] == 0:
        return np.asarray([0.,1.])
    v2_0 = 1.0
    v2_1 = -(vector[0]/vector[1])
    v2   = np.asarray([v2_0, v2_1])
    return v2 / np.linalg.norm(v2)


def get_cross2d(v1, v2):
    """
    Simple implementation of cross product in 2D.

    :type v1, v2: 1-d array-like with 2 elements (i.e., 2-element 1-d numpy array)
    """
    return v1[0]*v2[1] - v1[1]*v2[0]