#! python

import numpy as np
from scipy import linalg


def ellipsoid_fit(point_data, mode=''):
    """ Fit an ellipsoid to a cloud of points using linear least squares
        Based on Yury Petrov MATLAB algorithm: "ellipsoid_fit.m"
    """

    X = point_data[:, 0]
    Y = point_data[:, 1]
    Z = point_data[:, 2]

    # ALGEBRAIC EQUATION FOR ELLIPSOID, from CARTESIAN DATA
    if mode == '':  # 9-DOF MODE
        D = np.array([
            X ** 2 + Y ** 2 - 2 * Z ** 2,
            X ** 2 + Z ** 2 - 2 * Y ** 2,
            2 * X * Y,
            2 * X * Z,
            2 * Y * Z,
            2 * X,
            2 * Y,
            2 * Z,
            1 + 0 * X
        ])
    elif mode == '2D':
        D = np.array([
            X ** 2 + Y ** 2 - 2 * Z ** 2,
            X ** 2 + Z ** 2 - 2 * Y ** 2,
            2 * X * Y,
            2 * X * Z,
            2 * Y * Z,
            2 * X,
            2 * Y,
            2 * Z,
            1 + 0 * X
        ])
    elif mode == 0:  # 6-DOF MODE (no rotation)
        D = np.array([
            X ** 2 + Y ** 2 - 2 * Z ** 2,
            X ** 2 + Z ** 2 - 2 * Y ** 2,
            2 * X,
            2 * Y,
            2 * Z,
            1 + 0 * X
        ])

    # THE RIGHT-HAND-SIDE OF THE LLSQ PROBLEM
    d2 = np.array([X ** 2 + Y ** 2 + Z ** 2]).T

    # SOLUTION TO NORMAL SYSTEM OF EQUATIONS
    u = np.linalg.solve(D.dot(D), D.dot(d2))
    # chi2 = (1 - (D.dot(u)) / d2) ^ 2

    # CONVERT BACK TO ALGEBRAIC FORM
    if mode == '':  # 9-DOF-MODE
        a = np.array([u[0] + 1 * u[1] - 1])
        b = np.array([u[0] - 2 * u[1] - 1])
        c = np.array([u[1] - 2 * u[0] - 1])
        v = np.concatenate([a, b, c, u[2:, :]], axis=0).flatten()

    elif mode == 0:  # 6-DOF-MODE
        a = u[0] + 1 * u[1] - 1
        b = u[0] - 2 * u[1] - 1
        c = u[1] - 2 * u[0] - 1
        zs = np.array([0, 0, 0])
        v = np.hstack((a, b, c, zs, u[2:, :].flatten()))

    else:
        pass

    # PUT IN ALGEBRAIC FORM FOR ELLIPSOID
    A = np.array(
        [
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ]
    )

    # FIND center OF ELLIPSOID
    center = np.linalg.solve(-A[:3, :3], v[6:9])

    # FORM THE CORRESPONDING TRANSLATION MATRIX
    T = np.eye(4)
    T[3, :3] = center

    # TRANSLATE TO THE center, ROTATE
    R = T.dot(A).dot(T.T)

    # SOLVE THE EIGEN PROBLEM
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])

    # CALCULATE SCALE FACTORS AND SIGNS
    radii = np.sqrt(1 / abs(evals))
    sgns = np.sign(evals)
    radii *= sgns

    return center, evecs, radii, v


def ls_ellipse(xy):
    """
    Least squares fit to an ellipse
    A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
    where F = -1
    :param xy: 2 x N vector
    :return: Returns coefficients A..F
    """
    # Get N x 1 matrices, so we can use hstack
    x, y = xy[..., np.newaxis]

    J = np.hstack((x**2, x * y, y**2, x, y))
    K = np.ones_like(x)  # column of ones

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = linalg.inv(JTJ)
    ABC = np.dot(InvJTJ, np.dot(JT, K))
    # ABC has polynomial coefficients A..E
    # Move the 1 to the other side and return A..F
    # A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
    return np.append(ABC, -1)


def ellipse_params_from_poly(v, verbose):
    """
    Convert the polynomial form of the ellipse to parameters: center, axes, and tilt
    :param v: vector whose elements are the polynomial coefficients A..F
    :param verbose:
    :return: (center, axes, tilt degrees, rotation matrix)
    """

    # Algebraic form: X.T * Amat * X --> polynomial form
    Amat = np.array([
        [v[0],   v[1]/2, v[3]/2],
        [v[1]/2, v[2],   v[4]/2],
        [v[3]/2, v[4]/2, v[5]]
    ])

    if verbose:
        print('Algebraic form of polynomial', Amat)

    # See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A2 = Amat[:2, :2]
    A2Inv = linalg.inv(A2)
    ofs = v[3:5] / 2
    cc = -np.dot(A2Inv, ofs)
    if verbose:
        print('Center at:', cc)

    # Center the ellipse at the origin
    Tofs = np.eye(3)
    Tofs[2, :2] = cc
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))
    if verbose:
        print('Algebraic form translated to center:', R)

    R2 = R[:2, :2]
    s1 = -R[2, 2]
    RS = R2/s1
    el, ec = np.linalg.eig(RS)

    recip = 1 / np.abs(el)
    axes = np.sqrt(recip)
    if verbose:
        print('Axes are', axes)

    rads = np.arctan2(ec[1, 0], ec[0, 0])
    deg = np.degrees(rads)  # convert radians to degrees (r2d=180.0/np.pi)
    if verbose:
        print('Rotation is ', deg)

    inve = linalg.inv(ec)  # inverse is actually the transpose here
    if verbose:
        print('Rotation matrix', inve)
    return cc[:2], axes[:2], deg, inve


def ellipse_print_ans(pv, xin, yin, verbose=False):
    print('\nPolynomial coefficients, F term is -1:', pv)

    # Normalize and make first term positive
    nrm = linalg.norm(pv)
    enrm = pv / nrm
    if enrm[0] < 0:
        enrm = - enrm
    print('\nNormalized Polynomial Coefficients:', enrm)

    # Convert polynomial coefficients to parameterized ellipse (center, axes, and tilt)
    # also returns rotation matrix in last parameter
    # either pv or normalized parameter will work
    # params = ellipse_params_from_poly(enrm, verbose)
    center, gains, tilt, R = ellipse_params_from_poly(pv, verbose)
    print("\nCenter at  %10.4f,%10.4f (truth is 1.5,  1.5)" % center)
    print("Axes gains %10.4f,%10.4f (truth is 1.55, 1.0)" % gains)
    print("Tilt Degrees %10.4f (truth is 30.0)" % tilt)
    print('\nRotation Matrix\n', R)

    # Check solution
    # Convert to unit sphere centered at origin
    #  1) Subtract off center
    #  2) Rotate points so bulges are aligned with x, y axes (no xy term)
    #  3) Scale the points by the inverse of the axes gains
    #  4) Back rotate
    # Rotations and gains are collected into single transformation matrix M

    # subtract the offset so ellipse is centered at origin
    xc = xin - center[0]
    yc = yin - center[1]

    # create transformation matrix
    L = np.diag([1 / gains[0], 1 / gains[1]])
    M = np.dot(R.T, np.dot(L, R))
    print('\nTransformation Matrix\n', M)

    # apply the transformation matrix
    xm, ym = np.dot(M, [xc, yc])
    # Calculate distance from origin for each point (ideal = 1.0)
    rm = np.sqrt(xm*xm + ym*ym)

    print("\nAverage Radius  %10.4f (truth is 1.0)" % np.mean(rm))
    print("Stdev of Radius %10.4f " % np.std(rm))


if __name__ == "__main__":
    # Test of least squares fit to an ellipse
    # Samples have random noise added to both X and Y components
    # True center is at (1.5, 1.5);
    # X axis is 1.55, Y axis is 1.0, tilt is 30 degrees
    # (or -150 from symmetry)
    #
    # Polynomial coefficients, F term is -1:
    # A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
    #
    # A= -0.53968362, B=  0.50979868, C= -0.8285294
    # D=  0.87914926, E=  1.72765849, F= -1

    # Polynomial coefficients after normalization:
    # A x^2 + B xy + C y^2 + Dx + Ey + F = 0
    #
    # A=  0.22041087, B= -0.20820563, C=  0.33837767
    # D= -0.3590512,  E= -0.70558878  F=  0.40840756

    # Test data, no noise
    xy0 = np.array([
        [
            2.2255, 2.5995, 2.8634, 2.9163,
            2.6252, 2.1366, 1.6252, 1.1421,
            0.7084, 0.3479, 0.1094, 0.1072,
            0.4497, 0.9500, 1.4583, 1.9341],
        [
            0.7817, 1.1319, 1.5717, 2.0812,
            2.5027, 2.6578, 2.6150, 2.4418,
            2.1675, 1.8020, 1.3480, 0.8351,
            0.4534, 0.3381, 0.4061, 0.5973]
    ])

    # Test data with added noise
    xy0noisy = np.array([
        [
            2.2422, 2.5713, 2.8677, 2.9708,
            2.7462, 2.2695, 1.7423, 1.2501,
            0.8562, 0.4489, 0.0933, 0.0639,
            0.3024, 0.7666, 1.2813, 1.7860],
        [
            0.7216, 1.1190, 1.5447, 2.0398,
            2.4942, 2.7168, 2.6496, 2.5163,
            2.1730, 1.8725, 1.5018, 0.9970,
            0.5509, 0.3211, 0.3729, 0.5340]
    ])

    print('\n==============================')
    print('\nSolution for Perfect Data (to 4 decimal places)')

    ans0 = ls_ellipse(xy0)
    ellipse_print_ans(ans0, *xy0, 0)

    print('\n==============================')
    print('\nSolution for Noisy Data')

    ans = ls_ellipse(xy0noisy)
    ellipse_print_ans(ans, *xy0noisy, 0)
